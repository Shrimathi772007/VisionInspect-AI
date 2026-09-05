"""Database-driven inspection analytics for the monitoring dashboard.

Every metric here is computed by PostgreSQL aggregation (COUNT/AVG/FILTER/
GROUP BY) - no endpoint in this module ever loads every Inspection row into
Python just to count or average them.

Ground truth (Inspection.status / MVTec dataset labels) and AI prediction
(Inspection.ai_prediction and friends) are always aggregated through
separate columns/filters and never combined into a single figure - see
app.models.inspection for why that separation exists.
"""

from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel
from sqlalchemy import Date, cast, func, select
from sqlalchemy.orm import Session

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product

# How many trailing calendar days (including today) activity_by_day covers.
ACTIVITY_WINDOW_DAYS = 14


class StatusCounts(BaseModel):
    """Ground-truth status counts (Inspection.status)."""

    pending: int
    good: int
    defective: int


class SourceCounts(BaseModel):
    upload: int
    mvtec_ad: int


class AIPredictionCounts(BaseModel):
    """AI prediction counts (Inspection.ai_prediction) - independent of StatusCounts."""

    good: int
    defective: int


class DailyActivity(BaseModel):
    date: date
    total: int
    good: int
    defective: int
    pending: int


class ProductBreakdown(BaseModel):
    product_id: int
    product_name: str
    total: int
    ai_defective: int


class InspectionAnalyticsSummary(BaseModel):
    total_inspections: int
    by_status: StatusCounts
    by_source: SourceCounts
    ai_analyzed_count: int
    ai_prediction_counts: AIPredictionCounts
    ai_defect_rate: float | None
    avg_reconstruction_error: float | None
    activity_by_day: list[DailyActivity]
    by_product: list[ProductBreakdown]


def _utc_day(column):
    """The calendar day (UTC) a timestamptz column falls on, as a DATE.

    Explicitly normalizes to UTC before truncating so day boundaries are
    stable regardless of the database session's timezone setting.
    """
    return cast(func.timezone("UTC", column), Date)


def _get_totals(db: Session) -> dict:
    """One aggregate query for every scalar count/average in the summary."""
    row = db.execute(
        select(
            func.count().label("total"),
            func.count().filter(Inspection.status == "pending").label("pending"),
            func.count().filter(Inspection.status == "good").label("good"),
            func.count().filter(Inspection.status == "defective").label("defective"),
            func.count().filter(Inspection.source == InspectionSource.upload).label("upload"),
            func.count().filter(Inspection.source == InspectionSource.mvtec_ad).label("mvtec_ad"),
            func.count().filter(Inspection.ai_prediction.isnot(None)).label("ai_analyzed"),
            func.count().filter(Inspection.ai_prediction == GOOD_PREDICTION).label("ai_good"),
            func.count().filter(Inspection.ai_prediction == DEFECTIVE_PREDICTION).label("ai_defective"),
            func.avg(Inspection.ai_reconstruction_error)
            .filter(Inspection.ai_prediction.isnot(None))
            .label("avg_reconstruction_error"),
        )
    ).one()
    return row._mapping


def _get_activity_by_day(db: Session) -> list[DailyActivity]:
    """Ground-truth activity for the last ACTIVITY_WINDOW_DAYS calendar days.

    Days with no inspections are zero-filled so the series is always exactly
    ACTIVITY_WINDOW_DAYS long and ascending - the aggregate query only ever
    returns as many rows as there are days with data, never one per inspection.
    """
    today = datetime.now(timezone.utc).date()
    window_start = today - timedelta(days=ACTIVITY_WINDOW_DAYS - 1)

    day_col = _utc_day(Inspection.created_at)
    rows = db.execute(
        select(
            day_col.label("day"),
            func.count().label("total"),
            func.count().filter(Inspection.status == "good").label("good"),
            func.count().filter(Inspection.status == "defective").label("defective"),
            func.count().filter(Inspection.status == "pending").label("pending"),
        )
        .where(day_col >= window_start)
        .where(day_col <= today)
        .group_by(day_col)
    ).all()

    by_day = {row.day: row for row in rows}

    activity = []
    for offset in range(ACTIVITY_WINDOW_DAYS):
        day = window_start + timedelta(days=offset)
        row = by_day.get(day)
        activity.append(
            DailyActivity(
                date=day,
                total=row.total if row else 0,
                good=row.good if row else 0,
                defective=row.defective if row else 0,
                pending=row.pending if row else 0,
            )
        )
    return activity


def _get_by_product(db: Session) -> list[ProductBreakdown]:
    """One row per product that has at least one inspection.

    A product with zero inspections has nothing to monitor, so it is left
    out rather than padded in with zeros.
    """
    rows = db.execute(
        select(
            Product.id.label("product_id"),
            Product.product_name.label("product_name"),
            func.count(Inspection.id).label("total"),
            func.count().filter(Inspection.ai_prediction == DEFECTIVE_PREDICTION).label("ai_defective"),
        )
        .join(Inspection, Inspection.product_id == Product.id)
        .group_by(Product.id, Product.product_name)
        .order_by(func.count(Inspection.id).desc(), Product.product_name.asc(), Product.id.asc())
    ).all()

    return [
        ProductBreakdown(
            product_id=row.product_id,
            product_name=row.product_name,
            total=row.total,
            ai_defective=row.ai_defective,
        )
        for row in rows
    ]


def get_inspection_analytics_summary(db: Session) -> InspectionAnalyticsSummary:
    """Database-aggregated inspection metrics for the monitoring dashboard."""
    totals = _get_totals(db)

    ai_analyzed = totals["ai_analyzed"]
    ai_defective = totals["ai_defective"]
    ai_defect_rate = (ai_defective / ai_analyzed) if ai_analyzed else None
    avg_reconstruction_error = (
        float(totals["avg_reconstruction_error"]) if totals["avg_reconstruction_error"] is not None else None
    )

    return InspectionAnalyticsSummary(
        total_inspections=totals["total"],
        by_status=StatusCounts(pending=totals["pending"], good=totals["good"], defective=totals["defective"]),
        by_source=SourceCounts(upload=totals["upload"], mvtec_ad=totals["mvtec_ad"]),
        ai_analyzed_count=ai_analyzed,
        ai_prediction_counts=AIPredictionCounts(good=totals["ai_good"], defective=ai_defective),
        ai_defect_rate=ai_defect_rate,
        avg_reconstruction_error=avg_reconstruction_error,
        activity_by_day=_get_activity_by_day(db),
        by_product=_get_by_product(db),
    )
