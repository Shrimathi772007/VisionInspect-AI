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

MILESTONE 3 PHASE 6 - DEFECT TRENDS & FINAL INTEGRATION
--------------------------------------------------------
Adds `trend_monitoring` to the same summary: historical (never predictive) aggregation
over a fixed trailing window, reusing ACTIVITY_WINDOW_DAYS (14 days) as TREND_WINDOW_DAYS
so the new charts share exactly the window already familiar from activity_by_day/
ActivityChart - one window definition for the whole dashboard, not a second one to keep
in sync.

Design choices, made explicit here so they can be revisited independently:
    - `trend_monitoring.daily` folds the "daily totals" (Step 3) and "quality decision
      trends" (Step 5) requirements into ONE per-day series (one GROUP BY day query,
      the same idiom as _get_activity_by_day) rather than two separate day-indexed lists -
      half the response size, and a client only ever has to walk one calendar axis.
    - `trend_monitoring.category_trends` is capped at MAX_CATEGORY_TRENDS categories (by
      volume within the window, "good"/None excluded - those are not defects) so the
      response cannot grow with however many MVTec categories a dataset happens to define.
    - `trend_monitoring.insights` are deterministic CURRENT-vs-PREVIOUS-window comparisons
      (two fixed, already-elapsed TREND_WINDOW_DAYS periods) - historical observations
      only, never forecasts. Gated by MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT on both
      sides of the comparison; below that, a single "insufficient_trend_history" insight
      is returned instead of a fabricated direction.
    - Product-level trend information (Step 6) is folded onto the existing `by_product`
      rows (`recent_defect_rate`, `recent_trend`) rather than a second product x day
      response - a directional indicator is what is operationally useful here, not a
      full time series per product.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Date, cast, func, select
from sqlalchemy.orm import Session

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION
from app.inspections.quality import FAIL, PASS
from app.inspections.quality import NOT_ASSESSED as QUALITY_NOT_ASSESSED
from app.inspections.severity import NOT_ASSESSED as SEVERITY_NOT_ASSESSED
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product

# How many trailing calendar days (including today) activity_by_day covers.
ACTIVITY_WINDOW_DAYS = 14

# Milestone 3 Phase 6 - trend monitoring uses the SAME window length as activity_by_day
# (documented above): one "recent history" window definition for the whole dashboard.
TREND_WINDOW_DAYS = ACTIVITY_WINDOW_DAYS

# Display/response-size cap (NOT a specification value): the number of distinct defect
# categories tracked in category_trends, chosen by volume within the trend window so the
# dashboard chart stays readable and the response stays small regardless of how many
# MVTec categories exist.
MAX_CATEGORY_TRENDS = 5

# A period-over-period rate change smaller than this (5 percentage points) is reported as
# "stable" rather than "up"/"down" - avoids noisy up/down flip-flopping from small swings
# that are not a meaningful change. Implementation threshold, not a spec value.
TREND_STABLE_THRESHOLD = 0.05

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
    # Milestone 3 Phase 6 additions - ground-truth defect rate within just the last
    # TREND_WINDOW_DAYS days, and how it compares with the previous window of equal
    # length. `recent_defect_rate` is None, and `recent_trend` is "insufficient_data",
    # whenever either window has fewer than MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT
    # inspections - never a fabricated rate or direction from a handful of rows.
    recent_defect_rate: Optional[float] = None
    recent_trend: str = "insufficient_data"  # "up" / "down" / "stable" / "insufficient_data"


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


class TrendDay(BaseModel):
    """One calendar day of the trend-monitoring window - ground truth (total/good/
    defective/pending), AI predictions, and quality decisions, each independently counted
    the same way as the rest of this module (never merged into one figure)."""

    date: date
    total: int
    good: int
    defective: int
    pending: int
    ai_analyzed: int
    ai_defective: int
    quality_pass: int
    quality_fail: int
    quality_not_assessed: int


class CategoryTrendPoint(BaseModel):
    date: date
    count: int


class CategoryTrend(BaseModel):
    """Daily history for one MVTec ground-truth defect category (Phase 1) - only the
    top MAX_CATEGORY_TRENDS categories by volume within the window are included."""

    category: str
    daily: list[CategoryTrendPoint]


class TrendMonitoring(BaseModel):
    """Milestone 3 Phase 6 - historical trend monitoring over the last `period_days`
    days, compared against the previous `period_days`-day period for `insights`. Every
    figure here is a count of what already happened; nothing here is a forecast."""

    period_days: int
    daily: list[TrendDay]
    category_trends: list[CategoryTrend]
    insights: list[OperationalInsight]


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
    # Milestone 3 Phase 6 addition
    trend_monitoring: TrendMonitoring


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


def _classify_rate_change(current_rate: float, previous_rate: float) -> str:
    """"up" / "down" / "stable" for a period-over-period rate comparison - callers are
    responsible for only calling this once both periods meet the minimum sample size."""
    diff = current_rate - previous_rate
    if abs(diff) < TREND_STABLE_THRESHOLD:
        return "stable"
    return "up" if diff > 0 else "down"


def _get_product_period_trends(db: Session, today: date, window_days: int) -> dict:
    """One aggregate query, grouped by product, splitting the last 2*window_days days into
    a "current" and "previous" window of equal length - the basis for each product's
    `recent_defect_rate`/`recent_trend` in _get_by_product. A product with no inspections
    in either window simply has no row here (never zero-padded into a fake trend)."""
    current_start = today - timedelta(days=window_days - 1)
    previous_start = today - timedelta(days=(2 * window_days) - 1)
    previous_end = current_start - timedelta(days=1)
    day_col = _utc_day(Inspection.created_at)

    rows = db.execute(
        select(
            Inspection.product_id.label("product_id"),
            func.count().filter(day_col.between(current_start, today)).label("current_total"),
            func.count()
            .filter(day_col.between(current_start, today), Inspection.status == "defective")
            .label("current_defective"),
            func.count().filter(day_col.between(previous_start, previous_end)).label("previous_total"),
            func.count()
            .filter(day_col.between(previous_start, previous_end), Inspection.status == "defective")
            .label("previous_defective"),
        )
        .where(day_col.between(previous_start, today))
        .group_by(Inspection.product_id)
    ).all()
    return {row.product_id: row for row in rows}


def _get_by_product(db: Session, today: date, window_days: int) -> list[ProductBreakdown]:
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

    period_trends = _get_product_period_trends(db, today, window_days)

    breakdown = []
    for row in rows:
        period = period_trends.get(row.product_id)
        recent_defect_rate = None
        recent_trend = "insufficient_data"
        if period is not None:
            has_current_sample = period.current_total >= MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT
            has_previous_sample = period.previous_total >= MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT
            if has_current_sample:
                recent_defect_rate = period.current_defective / period.current_total
            if has_current_sample and has_previous_sample:
                recent_trend = _classify_rate_change(
                    period.current_defective / period.current_total,
                    period.previous_defective / period.previous_total,
                )
        breakdown.append(
            ProductBreakdown(
                product_id=row.product_id,
                product_name=row.product_name,
                total=row.total,
                ai_defective=row.ai_defective,
                defective=row.defective,
                defect_rate=row.defective / row.total,  # total is always >= 1 here
                recent_defect_rate=recent_defect_rate,
                recent_trend=recent_trend,
            )
        )
    return breakdown


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


# ---------------------------------------------------------------------------
# Milestone 3 Phase 6 - trend monitoring
# ---------------------------------------------------------------------------


def _get_trend_daily(db: Session, today: date, window_days: int) -> list[TrendDay]:
    """Ground truth, AI, and quality-decision counts for each of the last `window_days`
    calendar days - one GROUP BY day query (same idiom as _get_activity_by_day), zero-
    filled so the series is always exactly `window_days` long and ascending."""
    window_start = today - timedelta(days=window_days - 1)
    day_col = _utc_day(Inspection.created_at)
    decision_expr = func.coalesce(Inspection.quality_decision, QUALITY_NOT_ASSESSED)

    rows = db.execute(
        select(
            day_col.label("day"),
            func.count().label("total"),
            func.count().filter(Inspection.status == "good").label("good"),
            func.count().filter(Inspection.status == "defective").label("defective"),
            func.count().filter(Inspection.status == "pending").label("pending"),
            func.count().filter(Inspection.ai_prediction.isnot(None)).label("ai_analyzed"),
            func.count().filter(Inspection.ai_prediction == DEFECTIVE_PREDICTION).label("ai_defective"),
            func.count().filter(decision_expr == PASS).label("quality_pass"),
            func.count().filter(decision_expr == FAIL).label("quality_fail"),
            func.count().filter(decision_expr == QUALITY_NOT_ASSESSED).label("quality_not_assessed"),
        )
        .where(day_col >= window_start)
        .where(day_col <= today)
        .group_by(day_col)
    ).all()

    by_day = {row.day: row for row in rows}

    daily = []
    for offset in range(window_days):
        day = window_start + timedelta(days=offset)
        row = by_day.get(day)
        daily.append(
            TrendDay(
                date=day,
                total=row.total if row else 0,
                good=row.good if row else 0,
                defective=row.defective if row else 0,
                pending=row.pending if row else 0,
                ai_analyzed=row.ai_analyzed if row else 0,
                ai_defective=row.ai_defective if row else 0,
                quality_pass=row.quality_pass if row else 0,
                quality_fail=row.quality_fail if row else 0,
                quality_not_assessed=row.quality_not_assessed if row else 0,
            )
        )
    return daily


def _get_category_trends(db: Session, today: date, window_days: int) -> list[CategoryTrend]:
    """Daily history for the top MAX_CATEGORY_TRENDS defect categories (by volume within
    the window) - two bounded aggregate queries total, never one per category or per row.
    "good" and NULL are excluded: this tracks DEFECT categories, not the absence of one."""
    window_start = today - timedelta(days=window_days - 1)
    day_col = _utc_day(Inspection.created_at)

    top_rows = db.execute(
        select(Inspection.defect_category.label("category"), func.count().label("count"))
        .where(day_col >= window_start)
        .where(day_col <= today)
        .where(Inspection.defect_category.isnot(None))
        .where(Inspection.defect_category != _NON_DEFECT_CATEGORY)
        .group_by(Inspection.defect_category)
        .order_by(func.count().desc(), Inspection.defect_category.asc())
        .limit(MAX_CATEGORY_TRENDS)
    ).all()
    categories = [row.category for row in top_rows]
    if not categories:
        return []

    rows = db.execute(
        select(
            day_col.label("day"),
            Inspection.defect_category.label("category"),
            func.count().label("count"),
        )
        .where(day_col >= window_start)
        .where(day_col <= today)
        .where(Inspection.defect_category.in_(categories))
        .group_by(day_col, Inspection.defect_category)
    ).all()
    by_cat_day = {(row.category, row.day): row.count for row in rows}

    trends = []
    for category in categories:  # preserve the by-volume order computed above
        daily = []
        for offset in range(window_days):
            day = window_start + timedelta(days=offset)
            daily.append(CategoryTrendPoint(date=day, count=by_cat_day.get((category, day), 0)))
        trends.append(CategoryTrend(category=category, daily=daily))
    return trends


def _get_previous_period_totals(db: Session, today: date, window_days: int) -> dict:
    """Aggregate (non-daily) totals for the window immediately BEFORE the current trend
    window - one query, used only as the "previous period" side of _build_trend_insights'
    comparisons. Not itself part of the response."""
    previous_start = today - timedelta(days=(2 * window_days) - 1)
    previous_end = today - timedelta(days=window_days)
    day_col = _utc_day(Inspection.created_at)
    decision_expr = func.coalesce(Inspection.quality_decision, QUALITY_NOT_ASSESSED)

    row = db.execute(
        select(
            func.count().label("total"),
            func.count().filter(Inspection.status == "defective").label("defective"),
            func.count().filter(decision_expr == FAIL).label("fail"),
        )
        .where(day_col >= previous_start)
        .where(day_col <= previous_end)
    ).one()
    return row._mapping


def _build_trend_insights(
    period_days: int,
    current_total: int,
    current_defective: int,
    current_fail: int,
    previous_total: int,
    previous_defective: int,
    previous_fail: int,
    category_trends: list[CategoryTrend],
    by_product: list[ProductBreakdown],
) -> list[OperationalInsight]:
    """Deterministic CURRENT-period-vs-PREVIOUS-period observations - a comparison of two
    fixed, already-elapsed `period_days`-day windows. Never a forecast: every message
    describes what already happened. Gated by MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT on
    both sides so a handful of inspections cannot produce a misleading swing; below that,
    a single explicit "insufficient data" observation is returned instead.
    """
    if (
        current_total < MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT
        or previous_total < MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT
    ):
        return [
            OperationalInsight(
                type="insufficient_trend_history",
                message=(
                    f"Insufficient historical data yet to compare the last {period_days} days "
                    f"with the previous {period_days} days."
                ),
            )
        ]

    insights: list[OperationalInsight] = []

    current_rate = current_defective / current_total
    previous_rate = previous_defective / previous_total
    rate_direction = _classify_rate_change(current_rate, previous_rate)
    if rate_direction == "stable":
        insights.append(
            OperationalInsight(
                type="defect_rate_trend",
                message=(
                    f"Ground-truth defect rate has remained stable at approximately "
                    f"{current_rate * 100:.1f}% over the last {period_days} days."
                ),
            )
        )
    else:
        insights.append(
            OperationalInsight(
                type="defect_rate_trend",
                message=(
                    f"Ground-truth defect rate {'increased' if rate_direction == 'up' else 'decreased'} "
                    f"from {previous_rate * 100:.1f}% to {current_rate * 100:.1f}% compared with the "
                    f"previous {period_days} days."
                ),
            )
        )

    current_fail_rate = current_fail / current_total
    previous_fail_rate = previous_fail / previous_total
    fail_direction = _classify_rate_change(current_fail_rate, previous_fail_rate)
    if fail_direction != "stable":
        insights.append(
            OperationalInsight(
                type="quality_fail_trend",
                message=(
                    f"Quality FAIL decisions {'increased' if fail_direction == 'up' else 'decreased'} "
                    f"from {previous_fail_rate * 100:.1f}% to {current_fail_rate * 100:.1f}% compared "
                    f"with the previous {period_days} days."
                ),
            )
        )

    half = period_days // 2
    for trend in category_trends:  # already ordered by volume - first match wins
        first_half = sum(point.count for point in trend.daily[:half])
        second_half = sum(point.count for point in trend.daily[half:])
        window_total = first_half + second_half
        if window_total < MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT:
            continue
        if second_half > first_half and (second_half - first_half) / window_total >= TREND_STABLE_THRESHOLD:
            insights.append(
                OperationalInsight(
                    type="category_became_more_frequent",
                    message=(
                        f"'{trend.category}' defects became more frequent in the second half of the "
                        f"last {period_days} days ({first_half} -> {second_half})."
                    ),
                )
            )
            break

    elevated_products = [
        product
        for product in by_product
        if product.recent_trend == "up" and product.recent_defect_rate is not None
    ]
    if elevated_products:
        top_elevated = max(elevated_products, key=lambda product: (product.recent_defect_rate, product.product_id))
        insights.append(
            OperationalInsight(
                type="product_elevated_recent_activity",
                message=(
                    f"{top_elevated.product_name} shows elevated recent defect activity "
                    f"({top_elevated.recent_defect_rate * 100:.1f}% over the last {period_days} days)."
                ),
            )
        )

    return insights


def _get_trend_monitoring(db: Session, today: date, by_product: list[ProductBreakdown]) -> TrendMonitoring:
    daily = _get_trend_daily(db, today, TREND_WINDOW_DAYS)
    category_trends = _get_category_trends(db, today, TREND_WINDOW_DAYS)
    previous_totals = _get_previous_period_totals(db, today, TREND_WINDOW_DAYS)

    # Derived from the already-fetched `daily` (a fixed TREND_WINDOW_DAYS-long list), not a
    # separate DB query - summing a bounded, already-materialized list, not looping rows.
    current_total = sum(day.total for day in daily)
    current_defective = sum(day.defective for day in daily)
    current_fail = sum(day.quality_fail for day in daily)

    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total,
        current_defective,
        current_fail,
        previous_totals["total"],
        previous_totals["defective"],
        previous_totals["fail"],
        category_trends,
        by_product,
    )

    return TrendMonitoring(
        period_days=TREND_WINDOW_DAYS,
        daily=daily,
        category_trends=category_trends,
        insights=insights,
    )


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

    today = datetime.now(timezone.utc).date()

    by_product = _get_by_product(db, today, TREND_WINDOW_DAYS)
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
        trend_monitoring=_get_trend_monitoring(db, today, by_product),
    )
