"""Database-driven inspection analytics for the monitoring/defect-analytics dashboard.

Every metric here is computed by PostgreSQL aggregation (COUNT/AVG/FILTER/
GROUP BY) - no endpoint in this module ever loads every Inspection row into
Python just to count or average them.

Ground truth (Inspection.status / MVTec dataset labels), AI prediction
(Inspection.ai_prediction and friends), severity (Phase 2), and quality
assessment (Phase 3) are always aggregated through separate columns/filters
and never combined into a single figure - see app.models.inspection for why
that separation exists.

MILESTONE 3 PHASE 5 - DEFECT ANALYTICS
---------------------------------------
Extends the existing Milestone 2 summary (total/status/source/AI/activity/by_product)
with defect-category, quality-decision, and severity distributions, a richer per-product
breakdown, and deterministic operational insights - all computed from columns that
already exist (Phase 1-4), with zero new database tables or migrations. This is
deliberately the SAME endpoint (GET /inspections/analytics/summary) rather than a second
one, so there is exactly one coherent analytics API.

Units, by field-name convention (kept consistent with the pre-existing ai_defect_rate):
    *_rate       -> a ratio in [0, 1] (multiply by 100 to display as a percentage)
    percentage   -> already a percentage in [0, 100]

NULL-bucket handling (never fabricated, always an honest "not available" bucket):
    - defect_category: grouped as-is (raw, nullable) - a generic upload's NULL category
      surfaces as `category: null` in defect_categories, matching the existing frontend
      convention (badgeMaps.defectCategoryLabel(null) -> "Not categorized").
    - quality_decision: NULL only for a hypothetical pre-Phase-3 row; coalesced to the
      existing quality.NOT_ASSESSED value ("NOT_ASSESSED") for grouping, never a new
      vocabulary value.
    - severity_level: NULL whenever severity is not assessed (the normal case today,
      see app.inspections.severity's all-four-required policy); coalesced to the
      existing severity.NOT_ASSESSED value ("Not assessed") for grouping.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Date, cast, func, select
from sqlalchemy.orm import Session

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION
from app.inspections.quality import NOT_ASSESSED as QUALITY_NOT_ASSESSED
from app.inspections.severity import NOT_ASSESSED as SEVERITY_NOT_ASSESSED
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product

# How many trailing calendar days (including today) activity_by_day covers.
ACTIVITY_WINDOW_DAYS = 14

# Implementation threshold (NOT defined by the project specification): the minimum number
# of inspections a product must have before its ground-truth defect rate is treated as a
# meaningful basis for a "highest observed defect rate" insight. Below this, a single
# defective inspection out of very few would otherwise produce a misleading, statistically
# meaningless 100%-style figure. Documented here so it can be revisited independently of
# the rest of the aggregation logic.
MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT = 5

# "good" is a real defect_category value (the dataset's own ground truth for a
# non-defective sample) - excluded when looking for the "most common DEFECT category"
# since it does not represent a defect. Not the same concept as ai.inference's
# GOOD_PREDICTION (an AI prediction value); this is MVTec ground-truth category naming.
_NON_DEFECT_CATEGORY = "good"


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
    # Milestone 3 Phase 5 additions - ground-truth (Inspection.status), independent of
    # `ai_defective` above. `defect_rate` is a ratio in [0, 1]; always defined because a
    # product only appears here once it has at least one inspection (total >= 1).
    defective: int
    defect_rate: float


class DefectCategoryCount(BaseModel):
    """One row of the defect_category distribution (Milestone 3 Phase 5).

    `category` is Phase 1 MVTec ground truth, never an AI classification. `None` means
    "uncategorized" (e.g. a generic upload) - never inferred, matching the existing
    frontend convention (badgeMaps.defectCategoryLabel(null) -> "Not categorized").
    """

    category: Optional[str] = None
    count: int
    percentage: float  # of total_inspections, in [0, 100]


class QualityDecisionCount(BaseModel):
    """One row of the quality_decision distribution (Milestone 3 Phase 3 data)."""

    decision: str  # PASS / FAIL / NOT_ASSESSED - restates app.inspections.quality's own vocabulary
    count: int
    percentage: float


class SeverityLevelCount(BaseModel):
    """One row of the severity_level distribution (Milestone 3 Phase 2 data)."""

    level: str  # Critical / High / Medium / Low / "Not assessed"
    count: int
    percentage: float


class OperationalInsight(BaseModel):
    """One deterministic, evidence-based observation derived from the aggregates above -
    never a business claim (cost, downtime, root cause, safety, compliance) this system
    has no data to support."""

    type: str
    message: str


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
    # Milestone 3 Phase 5 additions
    defect_categories: list[DefectCategoryCount]
    quality_decisions: list[QualityDecisionCount]
    severity_distribution: list[SeverityLevelCount]
    operational_insights: list[OperationalInsight]


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
    out rather than padded in with zeros. `defective` is the ground-truth
    (Inspection.status) count, tracked independently of AI-derived `ai_defective`.
    """
    rows = db.execute(
        select(
            Product.id.label("product_id"),
            Product.product_name.label("product_name"),
            func.count(Inspection.id).label("total"),
            func.count().filter(Inspection.ai_prediction == DEFECTIVE_PREDICTION).label("ai_defective"),
            func.count().filter(Inspection.status == "defective").label("defective"),
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
            defective=row.defective,
            defect_rate=row.defective / row.total,  # total is always >= 1 here
        )
        for row in rows
    ]


def _percentage(count: int, total: int) -> float:
    return (count / total) * 100 if total else 0.0


def _get_defect_category_counts(db: Session, total: int) -> list[DefectCategoryCount]:
    """Distribution over the raw (nullable) defect_category column - grouped as-is, so a
    generic upload's NULL surfaces as its own `category: null` bucket rather than being
    coalesced away or guessed."""
    rows = db.execute(
        select(Inspection.defect_category.label("category"), func.count().label("count"))
        .group_by(Inspection.defect_category)
        .order_by(func.count().desc(), Inspection.defect_category.asc().nulls_last())
    ).all()
    return [
        DefectCategoryCount(category=row.category, count=row.count, percentage=_percentage(row.count, total))
        for row in rows
    ]


def _get_quality_decision_counts(db: Session, total: int) -> list[QualityDecisionCount]:
    """Distribution over quality_decision (Phase 3), NULL coalesced to the existing
    NOT_ASSESSED vocabulary value rather than left as an ungrouped null bucket."""
    decision_expr = func.coalesce(Inspection.quality_decision, QUALITY_NOT_ASSESSED)
    rows = db.execute(
        select(decision_expr.label("decision"), func.count().label("count"))
        .group_by(decision_expr)
        .order_by(func.count().desc())
    ).all()
    return [
        QualityDecisionCount(decision=row.decision, count=row.count, percentage=_percentage(row.count, total))
        for row in rows
    ]


def _get_severity_distribution(db: Session, total: int) -> list[SeverityLevelCount]:
    """Distribution over severity_level (Phase 2), NULL coalesced to the existing
    "Not assessed" label - the normal case today under Phase 2's all-four-required policy."""
    level_expr = func.coalesce(Inspection.severity_level, SEVERITY_NOT_ASSESSED)
    rows = db.execute(
        select(level_expr.label("level"), func.count().label("count"))
        .group_by(level_expr)
        .order_by(func.count().desc())
    ).all()
    return [
        SeverityLevelCount(level=row.level, count=row.count, percentage=_percentage(row.count, total))
        for row in rows
    ]


def _build_operational_insights(
    total: int,
    ai_analyzed: int,
    by_product: list[ProductBreakdown],
    defect_categories: list[DefectCategoryCount],
    quality_decisions: list[QualityDecisionCount],
) -> list[OperationalInsight]:
    """Deterministic, evidence-based observations from the aggregates already computed
    above - no additional queries. Each insight is omitted entirely (never fabricated with
    a placeholder) when its denominator/comparison would not be meaningful. Order is fixed
    so results are reproducible for the same underlying data.
    """
    if total == 0:
        return []

    insights: list[OperationalInsight] = []

    if by_product:
        top_volume = by_product[0]  # already ordered by total desc
        insights.append(
            OperationalInsight(
                type="highest_volume_product",
                message=f"Highest inspection volume: {top_volume.product_name} ({top_volume.total} inspections).",
            )
        )

    real_defect_categories = [
        c for c in defect_categories if c.category not in (None, _NON_DEFECT_CATEGORY) and c.count > 0
    ]
    if real_defect_categories:
        top_category = max(real_defect_categories, key=lambda c: (c.count, c.category))
        insights.append(
            OperationalInsight(
                type="most_common_defect_category",
                message=(
                    f"Most common recorded defect category: {top_category.category} "
                    f"({top_category.count} inspections)."
                ),
            )
        )

    eligible_products = [p for p in by_product if p.total >= MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT]
    if eligible_products:
        top_rate_product = max(eligible_products, key=lambda p: (p.defect_rate, -p.product_id))
        insights.append(
            OperationalInsight(
                type="highest_defect_rate_product",
                message=(
                    f"Highest observed ground-truth defect rate: {top_rate_product.product_name} "
                    f"({top_rate_product.defect_rate * 100:.1f}% of {top_rate_product.total} inspections)."
                ),
            )
        )

    insights.append(
        OperationalInsight(
            type="ai_analyzed_rate",
            message=f"{_percentage(ai_analyzed, total):.1f}% of inspections have an AI prediction.",
        )
    )

    not_assessed_count = next(
        (row.count for row in quality_decisions if row.decision == QUALITY_NOT_ASSESSED), 0
    )
    insights.append(
        OperationalInsight(
            type="not_assessed_rate",
            message=f"{_percentage(not_assessed_count, total):.1f}% of inspections are not yet quality-assessed.",
        )
    )

    return insights


def get_inspection_analytics_summary(db: Session) -> InspectionAnalyticsSummary:
    """Database-aggregated inspection metrics for the monitoring/defect-analytics dashboard."""
    totals = _get_totals(db)
    total = totals["total"]

    ai_analyzed = totals["ai_analyzed"]
    ai_defective = totals["ai_defective"]
    ai_defect_rate = (ai_defective / ai_analyzed) if ai_analyzed else None
    avg_reconstruction_error = (
        float(totals["avg_reconstruction_error"]) if totals["avg_reconstruction_error"] is not None else None
    )

    by_product = _get_by_product(db)
    defect_categories = _get_defect_category_counts(db, total)
    quality_decisions = _get_quality_decision_counts(db, total)
    severity_distribution = _get_severity_distribution(db, total)

    return InspectionAnalyticsSummary(
        total_inspections=total,
        by_status=StatusCounts(pending=totals["pending"], good=totals["good"], defective=totals["defective"]),
        by_source=SourceCounts(upload=totals["upload"], mvtec_ad=totals["mvtec_ad"]),
        ai_analyzed_count=ai_analyzed,
        ai_prediction_counts=AIPredictionCounts(good=totals["ai_good"], defective=ai_defective),
        ai_defect_rate=ai_defect_rate,
        avg_reconstruction_error=avg_reconstruction_error,
        activity_by_day=_get_activity_by_day(db),
        by_product=by_product,
        defect_categories=defect_categories,
        quality_decisions=quality_decisions,
        severity_distribution=severity_distribution,
        operational_insights=_build_operational_insights(
            total, ai_analyzed, by_product, defect_categories, quality_decisions
        ),
    )
