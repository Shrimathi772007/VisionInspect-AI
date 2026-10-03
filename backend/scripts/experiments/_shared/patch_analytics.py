from pathlib import Path

p = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend\app\inspections\analytics.py")
s = p.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


# ---- module docstring: document the new window + performance semantics ----
sub('''    - Product-level trend information (Step 6) is folded onto the existing `by_product`
      rows (`recent_defect_rate`, `recent_trend`) rather than a second product x day
      response - a directional indicator is what is operationally useful here, not a
      full time series per product.
"""
''',
'''    - Product-level trend information (Step 6) is folded onto the existing `by_product`
      rows (`recent_defect_rate`, `recent_trend`) rather than a second product x day
      response - a directional indicator is what is operationally useful here, not a
      full time series per product.

MILESTONE 4 - PERFORMANCE METRICS & SELECTABLE WINDOW
-----------------------------------------------------
Adds `performance` to the same summary and an optional trailing window (`window_days`, one
of ALLOWED_WINDOW_DAYS, default TREND_WINDOW_DAYS = 14 so a request without it behaves
exactly as before).

Which sections use the window, and which do not - stated explicitly so the dashboard can
label each one truthfully:
    - WINDOWED (follow `window_days`): `trend_monitoring` (daily series, category trends,
      current-vs-previous insights), each product's `recent_defect_rate`/`recent_trend`, and
      `performance`.
    - FIXED: `activity_by_day` stays the 14-day overview (ACTIVITY_WINDOW_DAYS) and every
      all-time aggregate (totals, distributions, per-product totals) stays all-time.
The window is always applied in SQL (a WHERE on the created_at UTC day), never by filtering
rows in Python.

`performance` is computed only from what is actually stored: Inspection.processing_time_ms
and Inspection.ai_inference_time_ms are NULL for every inspection recorded before they
existed, and aggregates ignore NULLs. `count` on each TimingStats is how many inspections in
the window actually have a measurement, and avg/median/min/max are None (never 0) when that
count is 0 - so a window with no timed inspections reports "no data", not a fabricated speed.
"""
''')

# ---- constants ----
sub('''# Milestone 3 Phase 6 - trend monitoring uses the SAME window length as activity_by_day
# (documented above): one "recent history" window definition for the whole dashboard.
TREND_WINDOW_DAYS = ACTIVITY_WINDOW_DAYS
''',
'''# Milestone 3 Phase 6 - trend monitoring's DEFAULT window equals ACTIVITY_WINDOW_DAYS, so an
# un-parameterized request behaves exactly as it did before windows became selectable.
TREND_WINDOW_DAYS = ACTIVITY_WINDOW_DAYS

# Milestone 4 - the trailing windows a client may request for the time-windowed sections.
# A small fixed set (rather than any integer) keeps the response size bounded and every
# supported value covered by tests.
ALLOWED_WINDOW_DAYS = (7, 14, 30)
''')

# ---- response models ----
sub('''class InspectionAnalyticsSummary(BaseModel):
    total_inspections: int''',
'''class TimingStats(BaseModel):
    """Summary of one timing column (milliseconds) over the inspections in the window that
    actually have a measurement. `count` is that number; the other fields are None when it
    is 0 - never a fabricated zero."""

    count: int
    avg_ms: Optional[float] = None
    median_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Milestone 4 - operational performance over the last `window_days` days.

    Terminology (kept deliberately literal): `processing_time` is the server-side handling
    time of the inspection request, `ai_inference_time` is the time predict_image reported
    (it includes model loading and threshold derivation) - see app.models.inspection.
    `ai_analyzed_rate` is the share of inspections in the window that received an AI
    prediction, a ratio in [0, 1]; it says nothing about AI accuracy or recall.
    """

    window_days: int
    inspections_in_window: int
    ai_analyzed_in_window: int
    ai_analyzed_rate: Optional[float] = None
    processing_time: TimingStats
    ai_inference_time: TimingStats


class InspectionAnalyticsSummary(BaseModel):
    total_inspections: int''')

sub('''    # Milestone 3 Phase 6 addition
    trend_monitoring: TrendMonitoring
''',
'''    # Milestone 3 Phase 6 addition
    trend_monitoring: TrendMonitoring
    # Milestone 4 addition
    performance: PerformanceMetrics
''')

# ---- trend monitoring takes the window ----
sub('''def _get_trend_monitoring(db: Session, today: date, by_product: list[ProductBreakdown]) -> TrendMonitoring:
    daily = _get_trend_daily(db, today, TREND_WINDOW_DAYS)
    category_trends = _get_category_trends(db, today, TREND_WINDOW_DAYS)
    previous_totals = _get_previous_period_totals(db, today, TREND_WINDOW_DAYS)

    # Derived from the already-fetched `daily` (a fixed TREND_WINDOW_DAYS-long list), not a
    # separate DB query - summing a bounded, already-materialized list, not looping rows.''',
'''def _get_trend_monitoring(
    db: Session, today: date, by_product: list[ProductBreakdown], window_days: int = TREND_WINDOW_DAYS
) -> TrendMonitoring:
    daily = _get_trend_daily(db, today, window_days)
    category_trends = _get_category_trends(db, today, window_days)
    previous_totals = _get_previous_period_totals(db, today, window_days)

    # Derived from the already-fetched `daily` (a `window_days`-long list), not a
    # separate DB query - summing a bounded, already-materialized list, not looping rows.''')

sub('''    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total,''',
'''    insights = _build_trend_insights(
        window_days,
        current_total,''')

sub('''    return TrendMonitoring(
        period_days=TREND_WINDOW_DAYS,''',
'''    return TrendMonitoring(
        period_days=window_days,''')

# ---- performance metrics + summary entry point ----
sub('''def get_inspection_analytics_summary(db: Session) -> InspectionAnalyticsSummary:
    """Database-aggregated inspection metrics for the monitoring/defect-analytics dashboard."""
    totals = _get_totals(db)''',
'''# ---------------------------------------------------------------------------
# Milestone 4 - performance metrics
# ---------------------------------------------------------------------------


def _timing_aggregates(column, prefix: str) -> tuple:
    """The SQL aggregates behind one TimingStats. COUNT(column) counts only non-NULL values,
    and AVG/MIN/MAX/percentile_cont all ignore NULLs, so inspections with no measurement
    (every historical row) contribute nothing rather than being read as zero."""
    return (
        func.count(column).label(f"{prefix}_count"),
        func.avg(column).label(f"{prefix}_avg"),
        func.percentile_cont(0.5).within_group(column).label(f"{prefix}_median"),
        func.min(column).label(f"{prefix}_min"),
        func.max(column).label(f"{prefix}_max"),
    )


def _timing_stats(row, prefix: str) -> TimingStats:
    count = row[f"{prefix}_count"]

    def _value(name: str) -> Optional[float]:
        value = row[f"{prefix}_{name}"]
        return float(value) if count and value is not None else None

    return TimingStats(
        count=count,
        avg_ms=_value("avg"),
        median_ms=_value("median"),
        min_ms=_value("min"),
        max_ms=_value("max"),
    )


def _get_performance_metrics(db: Session, today: date, window_days: int) -> PerformanceMetrics:
    """One aggregate query over the inspections created in the last `window_days` calendar
    days - the same UTC-day window idiom as the trend queries. Never loads a row."""
    window_start = today - timedelta(days=window_days - 1)
    day_col = _utc_day(Inspection.created_at)

    row = db.execute(
        select(
            func.count().label("inspections"),
            func.count().filter(Inspection.ai_prediction.isnot(None)).label("ai_analyzed"),
            *_timing_aggregates(Inspection.processing_time_ms, "processing"),
            *_timing_aggregates(Inspection.ai_inference_time_ms, "ai_inference"),
        )
        .where(day_col >= window_start)
        .where(day_col <= today)
    ).one()._mapping

    inspections = row["inspections"]
    ai_analyzed = row["ai_analyzed"]
    return PerformanceMetrics(
        window_days=window_days,
        inspections_in_window=inspections,
        ai_analyzed_in_window=ai_analyzed,
        ai_analyzed_rate=(ai_analyzed / inspections) if inspections else None,
        processing_time=_timing_stats(row, "processing"),
        ai_inference_time=_timing_stats(row, "ai_inference"),
    )


def get_inspection_analytics_summary(
    db: Session, window_days: int = TREND_WINDOW_DAYS
) -> InspectionAnalyticsSummary:
    """Database-aggregated inspection metrics for the monitoring/defect-analytics dashboard.

    `window_days` selects the trailing window for the time-windowed sections only (see the
    module docstring); it must be one of ALLOWED_WINDOW_DAYS, otherwise ValueError is raised.
    """
    if window_days not in ALLOWED_WINDOW_DAYS:
        raise ValueError(f"days must be one of {list(ALLOWED_WINDOW_DAYS)}, got {window_days}")

    totals = _get_totals(db)''')

sub('''    by_product = _get_by_product(db, today, TREND_WINDOW_DAYS)''',
    '''    by_product = _get_by_product(db, today, window_days)''')

sub('''        trend_monitoring=_get_trend_monitoring(db, today, by_product),
    )''',
'''        trend_monitoring=_get_trend_monitoring(db, today, by_product, window_days),
        performance=_get_performance_metrics(db, today, window_days),
    )''')

p.write_text(s, encoding="utf-8")
print("analytics.py updated")
