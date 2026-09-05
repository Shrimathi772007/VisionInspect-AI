"""Phase 8: GET /inspections/analytics/summary.

The test database is a real, shared PostgreSQL instance used across the whole
test session (see tests/conftest.py) - other test files create inspections
too, and this file's own tests run alongside them. So instead of asserting
fixed/global numbers (which would be flaky or outright wrong depending on
what ran before), every test here compares a "before" snapshot of the
summary to an "after" snapshot once we've created a known number of new
records, and asserts the *delta* matches what we just created. This is the
only way to test real database aggregation without seeding fake/demo rows.
"""

from datetime import datetime, timezone

import pytest

from app.ai.inference import PredictionResult
from tests.conftest import make_image_bytes

ANALYTICS_URL = "/inspections/analytics/summary"


def _get_summary(client, headers):
    response = client.get(ANALYTICS_URL, headers=headers)
    assert response.status_code == 200
    return response.json()


def _upload_plain_inspection(client, headers, product_id):
    """A user-uploaded image with no derivable MVTec category - ai_* stays null."""
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=headers,
        data={"product_id": str(product_id)},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    return response.json()


def _import_with_fake_prediction(
    client,
    headers,
    product_id,
    monkeypatch,
    *,
    prediction,
    reconstruction_error,
    category="bottle",
    split="test",
    defect_type="good",
    filename="000.png",
    threshold=0.003212,
    model_name="autoencoder",
):
    """Import an MVTec inspection with a deterministic, monkeypatched AI prediction.

    Mirrors the pattern already used in test_inspection_ai_integration.py:
    predict_image itself is not touched, only the seam
    app.inspections.service.predict_image is monkeypatched for this call, so
    the real inference engine (Phase 6) is never modified or bypassed in
    production code - just stubbed for this one test's determinism.
    """
    fake_result = PredictionResult(
        category=category,
        prediction=prediction,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        model_name=model_name,
        input_size=(128, 128),
        processing_time_ms=1.0,
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)

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


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def test_analytics_summary_requires_authentication(client):
    response = client.get(ANALYTICS_URL)
    assert response.status_code == 401


def test_analytics_summary_accessible_to_quality_engineer(client, qe_headers):
    response = client.get(ANALYTICS_URL, headers=qe_headers)
    assert response.status_code == 200


def test_analytics_summary_accessible_to_factory_supervisor(client, supervisor_headers):
    response = client.get(ANALYTICS_URL, headers=supervisor_headers)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Response shape - structure/types only, no fixed values
# ---------------------------------------------------------------------------


def test_analytics_summary_response_shape(client, qe_headers):
    body = _get_summary(client, qe_headers)

    assert isinstance(body["total_inspections"], int)
    assert set(body["by_status"].keys()) == {"pending", "good", "defective"}
    assert set(body["by_source"].keys()) == {"upload", "mvtec_ad"}
    assert isinstance(body["ai_analyzed_count"], int)
    assert set(body["ai_prediction_counts"].keys()) == {"good", "defective"}
    assert body["ai_defect_rate"] is None or isinstance(body["ai_defect_rate"], float)
    assert body["avg_reconstruction_error"] is None or isinstance(body["avg_reconstruction_error"], float)
    assert isinstance(body["activity_by_day"], list)
    assert isinstance(body["by_product"], list)

    for day in body["activity_by_day"]:
        assert set(day.keys()) == {"date", "total", "good", "defective", "pending"}

    for product in body["by_product"]:
        assert set(product.keys()) == {"product_id", "product_name", "total", "ai_defective"}


# ---------------------------------------------------------------------------
# total_inspections
# ---------------------------------------------------------------------------


def test_total_inspections_increases_by_exact_created_count(client, qe_headers, test_product):
    before = _get_summary(client, qe_headers)

    for _ in range(3):
        _upload_plain_inspection(client, qe_headers, test_product["id"])

    after = _get_summary(client, qe_headers)
    assert after["total_inspections"] == before["total_inspections"] + 3


# ---------------------------------------------------------------------------
# AI-analyzed vs non-AI
# ---------------------------------------------------------------------------


def test_plain_upload_does_not_count_as_ai_analyzed(client, qe_headers, test_product):
    before = _get_summary(client, qe_headers)

    body = _upload_plain_inspection(client, qe_headers, test_product["id"])
    assert body["ai_prediction"] is None

    after = _get_summary(client, qe_headers)
    assert after["ai_analyzed_count"] == before["ai_analyzed_count"]
    assert after["total_inspections"] == before["total_inspections"] + 1


def test_mvtec_bottle_import_counts_as_ai_analyzed(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)

    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", reconstruction_error=0.001,
    )

    after = _get_summary(client, qe_headers)
    assert after["ai_analyzed_count"] == before["ai_analyzed_count"] + 1
    assert after["total_inspections"] == before["total_inspections"] + 1


# ---------------------------------------------------------------------------
# AI prediction counts - good/defective tracked independently
# ---------------------------------------------------------------------------


def test_ai_prediction_counts_track_good_and_defective_independently(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)

    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", reconstruction_error=0.001,
    )
    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", reconstruction_error=0.01,
    )

    after = _get_summary(client, qe_headers)
    assert after["ai_prediction_counts"]["good"] == before["ai_prediction_counts"]["good"] + 1
    assert after["ai_prediction_counts"]["defective"] == before["ai_prediction_counts"]["defective"] + 1
    assert after["ai_analyzed_count"] == before["ai_analyzed_count"] + 2


# ---------------------------------------------------------------------------
# Ground truth (status) and AI prediction remain independent
# ---------------------------------------------------------------------------


def test_ground_truth_status_and_ai_prediction_counted_independently(client, qe_headers, test_product, monkeypatch):
    """Import a known-defective MVTec image (ground truth = defective) but stub the AI
    prediction as 'good' - by_status must reflect ground truth, ai_prediction_counts
    must reflect the AI result, and neither may leak into or overwrite the other."""
    before = _get_summary(client, qe_headers)

    body = _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", reconstruction_error=0.0005,
        defect_type="broken_large",
    )
    assert body["status"] == "defective"  # ground truth, untouched
    assert body["ai_prediction"] == "good"  # independently derived AI result

    after = _get_summary(client, qe_headers)
    assert after["by_status"]["defective"] == before["by_status"]["defective"] + 1
    assert after["ai_prediction_counts"]["good"] == before["ai_prediction_counts"]["good"] + 1
    # The mismatched AI "good" call must not have been counted as ground-truth good,
    # nor as an AI "defective" prediction.
    assert after["by_status"]["good"] == before["by_status"]["good"]
    assert after["ai_prediction_counts"]["defective"] == before["ai_prediction_counts"]["defective"]


# ---------------------------------------------------------------------------
# ai_defect_rate
# ---------------------------------------------------------------------------


def test_ai_defect_rate_matches_computed_ratio(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_analyzed = before["ai_analyzed_count"]
    before_defective = before["ai_prediction_counts"]["defective"]

    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="good", reconstruction_error=0.001,
    )
    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", reconstruction_error=0.02,
    )
    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", reconstruction_error=0.03,
    )

    after = _get_summary(client, qe_headers)
    expected_analyzed = before_analyzed + 3
    expected_defective = before_defective + 2
    expected_rate = expected_defective / expected_analyzed

    assert after["ai_analyzed_count"] == expected_analyzed
    assert after["ai_defect_rate"] == expected_rate


def test_zero_ai_analyzed_yields_null_defect_rate_and_avg_error(monkeypatch):
    """Unit-level check of the null-guard: with zero AI-analyzed rows,
    ai_defect_rate and avg_reconstruction_error must be null, never 0 or NaN.

    The shared test database always has AI-analyzed rows by this point in the
    suite, so this exercises the guard directly by stubbing the totals query
    rather than trying to contrive a globally-empty table.
    """
    from app.database import SessionLocal
    from app.inspections import analytics as analytics_module

    fake_totals = {
        "total": 10,
        "pending": 2,
        "good": 5,
        "defective": 3,
        "upload": 4,
        "mvtec_ad": 6,
        "ai_analyzed": 0,
        "ai_good": 0,
        "ai_defective": 0,
        "avg_reconstruction_error": None,
    }
    monkeypatch.setattr(analytics_module, "_get_totals", lambda db: fake_totals)

    db = SessionLocal()
    try:
        summary = analytics_module.get_inspection_analytics_summary(db)
    finally:
        db.close()

    assert summary.ai_analyzed_count == 0
    assert summary.ai_defect_rate is None
    assert summary.avg_reconstruction_error is None


# ---------------------------------------------------------------------------
# avg_reconstruction_error - only averaged over AI-analyzed rows
# ---------------------------------------------------------------------------


def test_avg_reconstruction_error_only_averages_ai_analyzed_rows(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_analyzed = before["ai_analyzed_count"]
    before_avg = before["avg_reconstruction_error"] or 0.0
    before_sum = before_avg * before_analyzed

    # A handful of non-AI uploads must NOT shift the average at all.
    for _ in range(2):
        _upload_plain_inspection(client, qe_headers, test_product["id"])

    new_errors = [0.0011, 0.0022, 0.0033]
    for error in new_errors:
        _import_with_fake_prediction(
            client, qe_headers, test_product["id"], monkeypatch,
            prediction="good", reconstruction_error=error,
        )

    after = _get_summary(client, qe_headers)
    expected_analyzed = before_analyzed + len(new_errors)
    expected_avg = (before_sum + sum(new_errors)) / expected_analyzed

    assert after["ai_analyzed_count"] == expected_analyzed
    assert after["avg_reconstruction_error"] == pytest.approx(expected_avg, abs=1e-6)


# ---------------------------------------------------------------------------
# activity_by_day
# ---------------------------------------------------------------------------


def test_activity_by_day_covers_last_14_days_ascending_and_reflects_new_inspection(
    client, qe_headers, test_product
):
    before = _get_summary(client, qe_headers)
    activity_before = before["activity_by_day"]
    assert len(activity_before) == 14

    dates = [datetime.strptime(day["date"], "%Y-%m-%d").date() for day in activity_before]
    assert dates == sorted(dates)
    assert dates[-1] == datetime.now(timezone.utc).date()
    assert (dates[-1] - dates[0]).days == 13

    today_total_before = activity_before[-1]["total"]

    _upload_plain_inspection(client, qe_headers, test_product["id"])

    after = _get_summary(client, qe_headers)
    today_bucket_after = after["activity_by_day"][-1]
    assert today_bucket_after["date"] == dates[-1].isoformat()
    assert today_bucket_after["total"] == today_total_before + 1


# ---------------------------------------------------------------------------
# by_product
# ---------------------------------------------------------------------------


def _find_product_entry(summary, product_id):
    for entry in summary["by_product"]:
        if entry["product_id"] == product_id:
            return entry
    return None


def test_by_product_totals_and_ai_defective_are_correct(client, qe_headers, test_product, monkeypatch):
    before = _get_summary(client, qe_headers)
    before_entry = _find_product_entry(before, test_product["id"])
    before_total = before_entry["total"] if before_entry else 0
    before_ai_defective = before_entry["ai_defective"] if before_entry else 0

    _upload_plain_inspection(client, qe_headers, test_product["id"])
    _upload_plain_inspection(client, qe_headers, test_product["id"])
    _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", reconstruction_error=0.05,
    )

    after = _get_summary(client, qe_headers)
    after_entry = _find_product_entry(after, test_product["id"])
    assert after_entry is not None
    assert after_entry["product_name"] == test_product["product_name"]
    assert after_entry["total"] == before_total + 3
    assert after_entry["ai_defective"] == before_ai_defective + 1


def test_by_product_is_sorted_descending_by_total(client, qe_headers):
    body = _get_summary(client, qe_headers)
    totals = [entry["total"] for entry in body["by_product"]]
    assert totals == sorted(totals, reverse=True)


# ---------------------------------------------------------------------------
# Existing inspection detail keeps exposing AI fields alongside ground truth
# ---------------------------------------------------------------------------


def test_inspection_detail_still_exposes_ai_and_ground_truth_fields(
    client, qe_headers, test_product, monkeypatch
):
    body = _import_with_fake_prediction(
        client, qe_headers, test_product["id"], monkeypatch,
        prediction="defective", reconstruction_error=0.0099,
        defect_type="good",
    )

    detail_response = client.get(f"/inspections/{body['id']}", headers=qe_headers)
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert detail["status"] == "good"  # ground truth, from dataset_defect_type "good"
    assert detail["ai_prediction"] == "defective"  # independent AI result
    assert detail["ai_reconstruction_error"] == 0.0099
    assert detail["ai_model_name"] == "autoencoder"
