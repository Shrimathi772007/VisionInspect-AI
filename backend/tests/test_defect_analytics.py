"""Milestone 3 Phase 5: defect analytics dashboard.

Extends the existing GET /inspections/analytics/summary (Phase 8/Milestone 2) rather than
adding a second endpoint - see app.inspections.analytics's module docstring. As in
test_inspection_analytics.py, the test database is a real, shared PostgreSQL instance used
across the whole test session, so every test here compares a "before" snapshot to an
"after" snapshot once a known number of new records exist, and asserts the *delta* - never
a fixed global number.

Covers: defect_categories, quality_decisions, severity_distribution, the extended
by_product (ground-truth `defective`/`defect_rate`), and operational_insights.
"""

from app.inspections.analytics import (
    MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT,
    DefectCategoryCount,
    ProductBreakdown,
    QualityDecisionCount,
    _build_operational_insights,
    _percentage,
)
from app.inspections.quality import FAIL, NOT_ASSESSED, PASS
from tests.conftest import make_image_bytes

ANALYTICS_URL = "/inspections/analytics/summary"


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


def _find_category_entry(summary, category):
    for entry in summary["defect_categories"]:
        if entry["category"] == category:
            return entry
    return None


def _find_decision_entry(summary, decision):
    for entry in summary["quality_decisions"]:
        if entry["decision"] == decision:
            return entry
    return None


def _find_severity_entry(summary, level):
    for entry in summary["severity_distribution"]:
        if entry["level"] == level:
            return entry
    return None


def _find_product_entry(summary, product_id):
    for entry in summary["by_product"]:
        if entry["product_id"] == product_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Authorization (same endpoint, re-asserted for this test file's own completeness)
# ---------------------------------------------------------------------------


def test_analytics_requires_authentication(client):
    assert client.get(ANALYTICS_URL).status_code == 401


def test_analytics_accessible_to_quality_engineer(client, qe_headers):
    assert client.get(ANALYTICS_URL, headers=qe_headers).status_code == 200


def test_analytics_accessible_to_factory_supervisor(client, supervisor_headers):
    assert client.get(ANALYTICS_URL, headers=supervisor_headers).status_code == 200


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_response_includes_phase5_sections(client, qe_headers):
    body = _get_summary(client, qe_headers)
    assert "defect_categories" in body
    assert "quality_decisions" in body
    assert "severity_distribution" in body
    assert "operational_insights" in body

    for entry in body["defect_categories"]:
        assert set(entry.keys()) == {"category", "count", "percentage"}
    for entry in body["quality_decisions"]:
        assert set(entry.keys()) == {"decision", "count", "percentage"}
    for entry in body["severity_distribution"]:
        assert set(entry.keys()) == {"level", "count", "percentage"}
    for entry in body["operational_insights"]:
        assert set(entry.keys()) == {"type", "message"}
    for entry in body["by_product"]:
        # `recent_defect_rate`/`recent_trend` are a Milestone 3 Phase 6 addition - see
        # test_defect_trends.py for their behavior.
        assert set(entry.keys()) == {
            "product_id", "product_name", "total", "ai_defective", "defective", "defect_rate",
            "recent_defect_rate", "recent_trend",
        }


def test_distribution_counts_sum_to_total_inspections(client, qe_headers):
    body = _get_summary(client, qe_headers)
    total = body["total_inspections"]
    assert sum(e["count"] for e in body["defect_categories"]) == total
    assert sum(e["count"] for e in body["quality_decisions"]) == total
    assert sum(e["count"] for e in body["severity_distribution"]) == total


# ---------------------------------------------------------------------------
# Defect category counts - ground truth only, never inferred
# ---------------------------------------------------------------------------


def test_defect_category_counts_reflect_known_categories(client, qe_headers, test_product):
    before = _get_summary(client, qe_headers)
    before_large = (_find_category_entry(before, "broken_large") or {"count": 0})["count"]
    before_contam = (_find_category_entry(before, "contamination") or {"count": 0})["count"]

    _import(client, qe_headers, test_product["id"], "broken_large", filename="016.png")
    _import(client, qe_headers, test_product["id"], "contamination", filename="016.png")

    after = _get_summary(client, qe_headers)
    assert _find_category_entry(after, "broken_large")["count"] == before_large + 1
    assert _find_category_entry(after, "contamination")["count"] == before_contam + 1


def test_generic_upload_counts_as_uncategorized_null_bucket(client, qe_headers, test_product):
    before = _get_summary(client, qe_headers)
    before_null = (_find_category_entry(before, None) or {"count": 0})["count"]

    body = _upload(client, qe_headers, test_product["id"])
    assert body["defect_category"] is None

    after = _get_summary(client, qe_headers)
    assert _find_category_entry(after, None)["count"] == before_null + 1


# ---------------------------------------------------------------------------
# Quality decision counts - deterministic via monkeypatched AI
# ---------------------------------------------------------------------------


def test_quality_decision_counts_reflect_pass_fail_and_conflict(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_pass = (_find_decision_entry(before, PASS) or {"count": 0})["count"]
    before_fail = (_find_decision_entry(before, FAIL) or {"count": 0})["count"]
    before_not_assessed = (_find_decision_entry(before, NOT_ASSESSED) or {"count": 0})["count"]

    pass_body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch, prediction="good", defect_type="good", filename="017.png"
    )
    assert pass_body["quality_decision"] == PASS

    fail_body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", defect_type="broken_large", filename="017.png",
    )
    assert fail_body["quality_decision"] == FAIL

    conflict_body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", defect_type="good", filename="018.png",
    )
    assert conflict_body["quality_decision"] == NOT_ASSESSED

    after = _get_summary(client, qe_headers)
    assert _find_decision_entry(after, PASS)["count"] == before_pass + 1
    assert _find_decision_entry(after, FAIL)["count"] == before_fail + 1
    assert _find_decision_entry(after, NOT_ASSESSED)["count"] == before_not_assessed + 1


def test_generic_upload_counts_as_not_assessed_quality_decision(client, qe_headers, test_product):
    before = _get_summary(client, qe_headers)
    before_not_assessed = (_find_decision_entry(before, NOT_ASSESSED) or {"count": 0})["count"]

    body = _upload(client, qe_headers, test_product["id"])
    assert body["quality_decision"] == NOT_ASSESSED

    after = _get_summary(client, qe_headers)
    assert _find_decision_entry(after, NOT_ASSESSED)["count"] == before_not_assessed + 1


# ---------------------------------------------------------------------------
# Severity distribution - honest "Not assessed" under the current Phase 2 policy
# ---------------------------------------------------------------------------


def test_severity_distribution_counts_new_inspections_as_not_assessed(client, qe_headers, test_product):
    """Every inspection is currently "Not assessed" (Phase 2's all-four-required policy -
    Location/Confidence are always unavailable), so new inspections must land there, never
    fabricated into a real level."""
    before = _get_summary(client, qe_headers)
    before_not_assessed = _find_severity_entry(before, "Not assessed")["count"]

    _upload(client, qe_headers, test_product["id"])
    _import(client, qe_headers, test_product["id"], "broken_small", filename="019.png")

    after = _get_summary(client, qe_headers)
    assert _find_severity_entry(after, "Not assessed")["count"] == before_not_assessed + 2
    # No fabricated Critical/High/Medium/Low bucket appears from these two new rows.
    for level in ("Critical", "High", "Medium", "Low"):
        before_level = _find_severity_entry(before, level)
        after_level = _find_severity_entry(after, level)
        before_count = before_level["count"] if before_level else 0
        after_count = after_level["count"] if after_level else 0
        assert after_count == before_count


# ---------------------------------------------------------------------------
# by_product - ground-truth `defective`/`defect_rate`, independent of `ai_defective`
# ---------------------------------------------------------------------------


def test_by_product_ground_truth_defective_and_rate(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_entry = _find_product_entry(before, test_product["id"])
    before_total = before_entry["total"] if before_entry else 0
    before_defective = before_entry["defective"] if before_entry else 0

    # Ground truth defective, but AI (monkeypatched) says good - defective/ai_defective
    # must move independently.
    body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", defect_type="broken_large", filename="007.png",
    )
    assert body["status"] == "defective"
    assert body["ai_prediction"] == "good"

    after = _get_summary(client, qe_headers)
    after_entry = _find_product_entry(after, test_product["id"])
    assert after_entry["total"] == before_total + 1
    assert after_entry["defective"] == before_defective + 1
    assert after_entry["defect_rate"] == after_entry["defective"] / after_entry["total"]


def _create_or_get_product(client, headers, product_name, product_code):
    """Same create-or-fetch pattern as conftest.test_product - the shared test database
    persists across runs, so a rerun must not fail on an already-registered product_code."""
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


def test_multiple_products_tracked_independently(client, qe_headers):
    product_a = _create_or_get_product(client, qe_headers, "Analytics Widget A", "ANALYTICS-A")
    product_b = _create_or_get_product(client, qe_headers, "Analytics Widget B", "ANALYTICS-B")

    before = _get_summary(client, qe_headers)
    before_a = (_find_product_entry(before, product_a["id"]) or {"total": 0})["total"]
    before_b = (_find_product_entry(before, product_b["id"]) or {"total": 0})["total"]

    _upload(client, qe_headers, product_a["id"])
    _upload(client, qe_headers, product_b["id"])
    _upload(client, qe_headers, product_b["id"])

    after = _get_summary(client, qe_headers)
    entry_a = _find_product_entry(after, product_a["id"])
    entry_b = _find_product_entry(after, product_b["id"])
    assert entry_a["total"] == before_a + 1
    assert entry_b["total"] == before_b + 2


# ---------------------------------------------------------------------------
# Operational insights - direct, deterministic unit tests (no DB needed)
# ---------------------------------------------------------------------------


def test_operational_insights_empty_for_zero_total():
    assert _build_operational_insights(0, 0, [], [], []) == []


def test_operational_insights_highest_volume_product():
    by_product = [
        ProductBreakdown(product_id=1, product_name="A", total=10, ai_defective=1, defective=2, defect_rate=0.2),
        ProductBreakdown(product_id=2, product_name="B", total=5, ai_defective=0, defective=1, defect_rate=0.2),
    ]
    insights = _build_operational_insights(15, 5, by_product, [], [])
    volume_insight = next(i for i in insights if i.type == "highest_volume_product")
    assert "A" in volume_insight.message
    assert "10" in volume_insight.message


def test_operational_insights_most_common_defect_category_excludes_good_and_null():
    categories = [
        DefectCategoryCount(category=None, count=50, percentage=50.0),
        DefectCategoryCount(category="good", count=30, percentage=30.0),
        DefectCategoryCount(category="contamination", count=15, percentage=15.0),
        DefectCategoryCount(category="broken_small", count=5, percentage=5.0),
    ]
    insights = _build_operational_insights(100, 20, [], categories, [])
    category_insight = next(i for i in insights if i.type == "most_common_defect_category")
    assert "contamination" in category_insight.message
    assert "good" not in category_insight.message  # "good" (a non-defect category) never chosen as the answer


def test_operational_insights_no_defect_category_insight_when_only_good_and_null():
    categories = [
        DefectCategoryCount(category=None, count=10, percentage=50.0),
        DefectCategoryCount(category="good", count=10, percentage=50.0),
    ]
    insights = _build_operational_insights(20, 5, [], categories, [])
    assert not any(i.type == "most_common_defect_category" for i in insights)


def test_operational_insights_defect_rate_requires_minimum_sample_size():
    small_sample_product = ProductBreakdown(
        product_id=1, product_name="TinySample", total=MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT - 1,
        ai_defective=0, defective=MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT - 1, defect_rate=1.0,
    )
    insights = _build_operational_insights(10, 5, [small_sample_product], [], [])
    assert not any(i.type == "highest_defect_rate_product" for i in insights)

    eligible_product = ProductBreakdown(
        product_id=2, product_name="EligibleSample", total=MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT,
        ai_defective=0, defective=1, defect_rate=1 / MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT,
    )
    insights = _build_operational_insights(10, 5, [eligible_product], [], [])
    assert any(i.type == "highest_defect_rate_product" for i in insights)


def test_operational_insights_ai_and_not_assessed_rates_always_present_when_total_positive():
    decisions = [QualityDecisionCount(decision=NOT_ASSESSED, count=7, percentage=70.0)]
    insights = _build_operational_insights(10, 3, [], [], decisions)
    types = {i.type for i in insights}
    assert "ai_analyzed_rate" in types
    assert "not_assessed_rate" in types

    not_assessed_insight = next(i for i in insights if i.type == "not_assessed_rate")
    assert "70.0%" in not_assessed_insight.message


def test_percentage_helper_handles_zero_total():
    assert _percentage(0, 0) == 0.0
    assert _percentage(5, 10) == 50.0


# ---------------------------------------------------------------------------
# No N+1 query pattern: the summary issues a small, fixed number of SQL statements
# regardless of how many inspections exist.
# ---------------------------------------------------------------------------


def test_analytics_summary_issues_a_bounded_number_of_queries(client, qe_headers, test_product):
    from sqlalchemy import event

    from app.database import SessionLocal, engine

    for _ in range(3):
        _upload(client, qe_headers, test_product["id"])

    query_count = 0

    def _count_queries(*args, **kwargs):
        nonlocal query_count
        query_count += 1

    event.listen(engine, "before_cursor_execute", _count_queries)
    try:
        from app.inspections.analytics import get_inspection_analytics_summary

        db = SessionLocal()
        try:
            get_inspection_analytics_summary(db)
        finally:
            db.close()
    finally:
        event.remove(engine, "before_cursor_execute", _count_queries)

    # A handful of fixed aggregate queries (totals, activity, by_product + its per-product
    # period-trend query, defect categories, quality decisions, severity distribution, plus
    # Milestone 3 Phase 6's trend-monitoring queries: daily trend, category trends [2
    # queries], previous-period totals) - 11 today, never one per inspection row or scaling
    # with row count. See test_defect_trends.py for Phase 6's own bounded-query assertion.
    assert query_count <= 20


# ---------------------------------------------------------------------------
# Ground truth vs AI vs severity vs quality remain distinguishable end to end
# ---------------------------------------------------------------------------


def test_ai_metrics_remain_separate_from_ground_truth_in_summary(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_status_defective = before["by_status"]["defective"]
    before_ai_good = before["ai_prediction_counts"]["good"]

    body = _import_with_fake_ai(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", defect_type="broken_small", filename="021.png",
    )
    assert body["status"] == "defective"
    assert body["ai_prediction"] == "good"

    after = _get_summary(client, qe_headers)
    assert after["by_status"]["defective"] == before_status_defective + 1
    assert after["ai_prediction_counts"]["good"] == before_ai_good + 1
