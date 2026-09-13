"""Milestone 3 Phase 6: defect trends & final integration.

Extends the existing GET /inspections/analytics/summary (same endpoint as Phase 5 - see
app.inspections.analytics's module docstring) with a `trend_monitoring` field: historical
(never predictive) aggregation over a fixed trailing window.

As in test_inspection_analytics.py / test_defect_analytics.py, the test database is a real,
shared PostgreSQL instance used across the whole test session, so endpoint-level tests use
before/after deltas on the CURRENT DAY's bucket rather than fixed absolute counts.

Deterministic-date isolation: several tests below call the private aggregation helpers
directly with a fixed, far-past `today` (year 2000) where no real inspection could possibly
exist. This makes "zero-data window" behavior (zero-fill, empty category list, no
fabricated recent-trend) verifiable without depending on the current system clock or on how
much data happens to already be in the shared database.
"""

from datetime import date, timezone, datetime

from app.inspections.analytics import (
    MAX_CATEGORY_TRENDS,
    MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT,
    TREND_STABLE_THRESHOLD,
    TREND_WINDOW_DAYS,
    CategoryTrend,
    CategoryTrendPoint,
    ProductBreakdown,
    _build_trend_insights,
    _classify_rate_change,
    _get_by_product,
    _get_category_trends,
    _get_previous_period_totals,
    _get_trend_daily,
)
from app.inspections.quality import FAIL, NOT_ASSESSED, PASS
from tests.conftest import make_image_bytes

ANALYTICS_URL = "/inspections/analytics/summary"

# A date far enough in the past that no real inspection (created "now", during test runs)
# can possibly fall inside its trend/previous windows - used to exercise zero-data behavior
# deterministically, independent of the current system clock or existing DB contents.
FAR_PAST = date(2000, 1, 1)


def _get_summary(client, headers):
    response = client.get(ANALYTICS_URL, headers=headers)
    assert response.status_code == 200
    return response.json()


def _upload(client, headers, product_id):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=headers,
        data={"product_id": str(product_id)},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    return response.json()


def _import(client, headers, product_id, defect_type, filename="000.png", category="bottle", split="test"):
    response = client.post(
        "/inspections/import",
        headers=headers,
        json={
            "product_id": product_id,
            "category": category,
            "split": split,
            "defect_type": defect_type,
            "filename": filename,
        },
    )
    assert response.status_code == 201
    return response.json()


def _import_with_fake_ai(client, headers, product_id, monkeypatch, *, prediction, defect_type, filename):
    from app.ai.inference.schemas import PredictionResult

    fake_result = PredictionResult(
        category="bottle", prediction=prediction, reconstruction_error=0.001, threshold=0.01, model_name="autoencoder"
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)
    return _import(client, headers, product_id, defect_type, filename=filename)


def _create_or_get_product(client, headers, product_name, product_code):
    response = client.post(
        "/products", json={"product_name": product_name, "product_code": product_code}, headers=headers
    )
    if response.status_code == 409:
        listing = client.get("/products", headers=headers)
        assert listing.status_code == 200
        for product in listing.json():
            if product["product_code"] == product_code:
                return product
        raise AssertionError("Product reported as duplicate but not found in listing")
    assert response.status_code == 201
    return response.json()


def _today_bucket(trend_monitoring):
    today_iso = datetime.now(timezone.utc).date().isoformat()
    for day in trend_monitoring["daily"]:
        if day["date"] == today_iso:
            return day
    raise AssertionError("today's date is missing from trend_monitoring.daily")


# ---------------------------------------------------------------------------
# Authorization (same endpoint, re-asserted for this test file's own completeness)
# ---------------------------------------------------------------------------


def test_trend_requires_authentication(client):
    assert client.get(ANALYTICS_URL).status_code == 401


def test_trend_accessible_to_quality_engineer(client, qe_headers):
    assert client.get(ANALYTICS_URL, headers=qe_headers).status_code == 200


def test_trend_accessible_to_factory_supervisor(client, supervisor_headers):
    assert client.get(ANALYTICS_URL, headers=supervisor_headers).status_code == 200


# ---------------------------------------------------------------------------
# trend_monitoring response shape
# ---------------------------------------------------------------------------


def test_trend_monitoring_shape(client, qe_headers):
    body = _get_summary(client, qe_headers)
    trend = body["trend_monitoring"]

    assert trend["period_days"] == TREND_WINDOW_DAYS
    assert isinstance(trend["daily"], list)
    assert len(trend["daily"]) == TREND_WINDOW_DAYS
    for day in trend["daily"]:
        assert set(day.keys()) == {
            "date", "total", "good", "defective", "pending",
            "ai_analyzed", "ai_defective", "quality_pass", "quality_fail", "quality_not_assessed",
        }

    assert isinstance(trend["category_trends"], list)
    assert len(trend["category_trends"]) <= MAX_CATEGORY_TRENDS
    for category_trend in trend["category_trends"]:
        assert set(category_trend.keys()) == {"category", "daily"}
        assert len(category_trend["daily"]) == TREND_WINDOW_DAYS
        for point in category_trend["daily"]:
            assert set(point.keys()) == {"date", "count"}

    assert isinstance(trend["insights"], list)
    for insight in trend["insights"]:
        assert set(insight.keys()) == {"type", "message"}
        assert isinstance(insight["type"], str)
        assert isinstance(insight["message"], str)


def test_trend_daily_dates_ascending_and_end_today(client, qe_headers):
    body = _get_summary(client, qe_headers)
    dates = [day["date"] for day in body["trend_monitoring"]["daily"]]
    assert dates == sorted(dates)
    assert dates[-1] == datetime.now(timezone.utc).date().isoformat()


def test_trend_category_trends_categories_are_never_good_or_null(client, qe_headers):
    body = _get_summary(client, qe_headers)
    categories = [entry["category"] for entry in body["trend_monitoring"]["category_trends"]]
    assert "good" not in categories
    assert None not in categories


# ---------------------------------------------------------------------------
# Daily trend counts - before/after deltas on TODAY's bucket
# ---------------------------------------------------------------------------


def test_trend_daily_total_increases_on_upload(client, qe_headers, test_product):
    before = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])
    _upload(client, qe_headers, test_product["id"])
    after = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])
    assert after["total"] == before["total"] + 1


def test_trend_daily_tracks_ground_truth_status_independently(client, qe_headers, test_product):
    before = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])

    _import(client, qe_headers, test_product["id"], "good", filename="001.png")
    _import(client, qe_headers, test_product["id"], "broken_large", filename="002.png")
    _upload(client, qe_headers, test_product["id"])  # generic upload -> pending

    after = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])
    assert after["good"] == before["good"] + 1
    assert after["defective"] == before["defective"] + 1
    assert after["pending"] == before["pending"] + 1


def test_trend_daily_tracks_ai_predictions_independently_of_status(client, qe_headers, test_product, monkeypatch):
    before = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])

    _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", defect_type="good", filename="003.png",
    )

    after = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])
    assert after["ai_analyzed"] == before["ai_analyzed"] + 1
    assert after["ai_defective"] == before["ai_defective"] + 1


def test_trend_daily_tracks_quality_decisions(client, qe_headers, test_product, monkeypatch):
    # Fake AI predictions that AGREE with ground truth, so the quality decision is
    # deterministic (rule 4 / rule 3 in app.inspections.quality) rather than depending on
    # what the real autoencoder happens to predict for these specific dataset images.
    before = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])

    upload_body = _upload(client, qe_headers, test_product["id"])
    assert upload_body["quality_decision"] == NOT_ASSESSED
    good_body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", defect_type="good", filename="004.png",
    )
    assert good_body["quality_decision"] == PASS
    fail_body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", defect_type="broken_large", filename="005.png",
    )
    assert fail_body["quality_decision"] == FAIL

    after = _today_bucket(_get_summary(client, qe_headers)["trend_monitoring"])
    assert after["quality_not_assessed"] == before["quality_not_assessed"] + 1
    assert after["quality_pass"] == before["quality_pass"] + 1
    assert after["quality_fail"] == before["quality_fail"] + 1


# ---------------------------------------------------------------------------
# Zero-data window behavior - deterministic (far-past `today`, never the live clock)
# ---------------------------------------------------------------------------


def test_trend_daily_zero_fills_a_window_with_no_data(client, qe_headers):
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        daily = _get_trend_daily(db, FAR_PAST, TREND_WINDOW_DAYS)
    finally:
        db.close()

    assert len(daily) == TREND_WINDOW_DAYS
    assert daily[-1].date == FAR_PAST
    for day in daily:
        assert day.total == 0
        assert day.good == 0
        assert day.defective == 0
        assert day.pending == 0
        assert day.ai_analyzed == 0
        assert day.ai_defective == 0
        assert day.quality_pass == 0
        assert day.quality_fail == 0
        assert day.quality_not_assessed == 0


def test_category_trends_empty_for_a_window_with_no_data(client, qe_headers):
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        trends = _get_category_trends(db, FAR_PAST, TREND_WINDOW_DAYS)
    finally:
        db.close()

    assert trends == []


def test_previous_period_totals_zero_for_a_window_with_no_data(client, qe_headers):
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        totals = _get_previous_period_totals(db, FAR_PAST, TREND_WINDOW_DAYS)
    finally:
        db.close()

    assert totals["total"] == 0
    assert totals["defective"] == 0
    assert totals["fail"] == 0


def test_recent_defect_rate_none_for_all_products_in_a_window_with_no_data(client, qe_headers, test_product):
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        by_product = _get_by_product(db, FAR_PAST, TREND_WINDOW_DAYS)
    finally:
        db.close()

    assert len(by_product) > 0  # real products exist (all-time), just no data in this window
    for product in by_product:
        assert product.recent_defect_rate is None
        assert product.recent_trend == "insufficient_data"


# ---------------------------------------------------------------------------
# by_product recent-trend fields - live endpoint, type-level only (no exact-value
# assumptions about the shared database's current state)
# ---------------------------------------------------------------------------


def test_by_product_entries_have_valid_recent_trend_fields(client, qe_headers):
    body = _get_summary(client, qe_headers)
    for product in body["by_product"]:
        assert product["recent_trend"] in {"up", "down", "stable", "insufficient_data"}
        assert product["recent_defect_rate"] is None or isinstance(product["recent_defect_rate"], float)
        if product["recent_trend"] == "insufficient_data":
            pass  # recent_defect_rate may still be a float here (current sample OK, previous not)
        else:
            assert product["recent_defect_rate"] is not None


# ---------------------------------------------------------------------------
# _classify_rate_change - pure function, no DB
# ---------------------------------------------------------------------------


def test_classify_rate_change_stable_within_threshold():
    assert _classify_rate_change(0.20, 0.20 + TREND_STABLE_THRESHOLD / 2) == "stable"


def test_classify_rate_change_up_beyond_threshold():
    assert _classify_rate_change(0.30, 0.20) == "up"


def test_classify_rate_change_down():
    assert _classify_rate_change(0.10, 0.30) == "down"


# ---------------------------------------------------------------------------
# _build_trend_insights - pure function, no DB, fully deterministic inputs
# ---------------------------------------------------------------------------


def test_trend_insights_insufficient_when_current_below_minimum():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT - 1, current_defective=1, current_fail=0,
        previous_total=50, previous_defective=10, previous_fail=5,
        category_trends=[], by_product=[],
    )
    assert len(insights) == 1
    assert insights[0].type == "insufficient_trend_history"


def test_trend_insights_insufficient_when_previous_below_minimum():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=5,
        previous_total=MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT - 1, previous_defective=1, previous_fail=0,
        category_trends=[], by_product=[],
    )
    assert len(insights) == 1
    assert insights[0].type == "insufficient_trend_history"


def test_trend_insights_stable_defect_rate_reported_as_stable():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=20, current_defective=4, current_fail=4,   # 20% both periods
        previous_total=20, previous_defective=4, previous_fail=4,
        category_trends=[], by_product=[],
    )
    rate_insight = next(i for i in insights if i.type == "defect_rate_trend")
    assert "stable" in rate_insight.message
    assert not any(i.type == "quality_fail_trend" for i in insights)


def test_trend_insights_increasing_defect_rate():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=20, current_defective=10, current_fail=0,   # 50%
        previous_total=20, previous_defective=2, previous_fail=0,  # 10%
        category_trends=[], by_product=[],
    )
    rate_insight = next(i for i in insights if i.type == "defect_rate_trend")
    assert "increased" in rate_insight.message


def test_trend_insights_decreasing_defect_rate():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=20, current_defective=2, current_fail=0,    # 10%
        previous_total=20, previous_defective=10, previous_fail=0,  # 50%
        category_trends=[], by_product=[],
    )
    rate_insight = next(i for i in insights if i.type == "defect_rate_trend")
    assert "decreased" in rate_insight.message


def test_trend_insights_quality_fail_trend_included_when_changed():
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=20, current_defective=5, current_fail=10,   # fail rate 50%
        previous_total=20, previous_defective=5, previous_fail=2,  # fail rate 10%
        category_trends=[], by_product=[],
    )
    fail_insight = next(i for i in insights if i.type == "quality_fail_trend")
    assert "increased" in fail_insight.message


def test_trend_insights_category_became_more_frequent():
    daily = [CategoryTrendPoint(date=FAR_PAST, count=0) for _ in range(7)] + [
        CategoryTrendPoint(date=FAR_PAST, count=2) for _ in range(7)
    ]
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=10,
        previous_total=50, previous_defective=10, previous_fail=10,
        category_trends=[CategoryTrend(category="broken_large", daily=daily)],
        by_product=[],
    )
    category_insight = next(i for i in insights if i.type == "category_became_more_frequent")
    assert "broken_large" in category_insight.message


def test_trend_insights_category_insight_omitted_below_sample_size():
    # Total window count (3) is below MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT (5) - too few
    # data points for a "became more frequent" claim to be meaningful, regardless of the
    # first-half/second-half split.
    daily = [CategoryTrendPoint(date=FAR_PAST, count=0) for _ in range(13)] + [
        CategoryTrendPoint(date=FAR_PAST, count=3)
    ]
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=10,
        previous_total=50, previous_defective=10, previous_fail=10,
        category_trends=[CategoryTrend(category="broken_small", daily=daily)],
        by_product=[],
    )
    assert not any(i.type == "category_became_more_frequent" for i in insights)


def test_trend_insights_product_elevated_activity_reported():
    elevated = ProductBreakdown(
        product_id=1, product_name="Elevated Line", total=30, ai_defective=5, defective=10,
        defect_rate=0.33, recent_defect_rate=0.6, recent_trend="up",
    )
    calm = ProductBreakdown(
        product_id=2, product_name="Calm Line", total=30, ai_defective=1, defective=3,
        defect_rate=0.1, recent_defect_rate=0.1, recent_trend="stable",
    )
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=10,
        previous_total=50, previous_defective=10, previous_fail=10,
        category_trends=[], by_product=[elevated, calm],
    )
    product_insight = next(i for i in insights if i.type == "product_elevated_recent_activity")
    assert "Elevated Line" in product_insight.message


def test_trend_insights_product_elevated_activity_omitted_when_none_up():
    calm = ProductBreakdown(
        product_id=1, product_name="Calm Line", total=30, ai_defective=1, defective=3,
        defect_rate=0.1, recent_defect_rate=0.1, recent_trend="stable",
    )
    down = ProductBreakdown(
        product_id=2, product_name="Improving Line", total=30, ai_defective=1, defective=1,
        defect_rate=0.03, recent_defect_rate=0.02, recent_trend="down",
    )
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=10,
        previous_total=50, previous_defective=10, previous_fail=10,
        category_trends=[], by_product=[calm, down],
    )
    assert not any(i.type == "product_elevated_recent_activity" for i in insights)


def test_trend_insights_picks_highest_rate_among_elevated_products():
    lower = ProductBreakdown(
        product_id=1, product_name="Lower Elevated", total=30, ai_defective=1, defective=6,
        defect_rate=0.2, recent_defect_rate=0.3, recent_trend="up",
    )
    higher = ProductBreakdown(
        product_id=2, product_name="Higher Elevated", total=30, ai_defective=1, defective=15,
        defect_rate=0.5, recent_defect_rate=0.7, recent_trend="up",
    )
    insights = _build_trend_insights(
        TREND_WINDOW_DAYS,
        current_total=50, current_defective=10, current_fail=10,
        previous_total=50, previous_defective=10, previous_fail=10,
        category_trends=[], by_product=[lower, higher],
    )
    product_insight = next(i for i in insights if i.type == "product_elevated_recent_activity")
    assert "Higher Elevated" in product_insight.message


# ---------------------------------------------------------------------------
# No N+1 query pattern for the trend-monitoring aggregation itself
# ---------------------------------------------------------------------------


def test_trend_monitoring_issues_a_bounded_number_of_queries(client, qe_headers, test_product):
    from sqlalchemy import event

    from app.database import SessionLocal, engine
    from app.inspections.analytics import _get_trend_monitoring

    for _ in range(3):
        _upload(client, qe_headers, test_product["id"])

    query_count = 0

    def _count_queries(*args, **kwargs):
        nonlocal query_count
        query_count += 1

    db = SessionLocal()
    try:
        by_product = _get_by_product(db, datetime.now(timezone.utc).date(), TREND_WINDOW_DAYS)
    finally:
        db.close()

    event.listen(engine, "before_cursor_execute", _count_queries)
    try:
        db = SessionLocal()
        try:
            _get_trend_monitoring(db, datetime.now(timezone.utc).date(), by_product)
        finally:
            db.close()
    finally:
        event.remove(engine, "before_cursor_execute", _count_queries)

    # Fixed aggregate queries only: daily trend (1), category trends (2: top categories +
    # their daily breakdown), previous-period totals (1) - never one per inspection row.
    assert query_count <= 6


# ---------------------------------------------------------------------------
# Regression: Phase 5 fields remain intact alongside the new trend_monitoring field
# ---------------------------------------------------------------------------


def test_phase5_fields_unaffected_by_trend_monitoring(client, qe_headers):
    body = _get_summary(client, qe_headers)
    assert "defect_categories" in body
    assert "quality_decisions" in body
    assert "severity_distribution" in body
    assert "operational_insights" in body
    assert "by_product" in body
    assert "trend_monitoring" in body
