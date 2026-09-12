"""Milestone 3 Phase 2 (+ correction review): severity scoring and quality risk assessment.

Two layers under test here (dataset mask I/O has its own test_mvtec_ground_truth.py, and
service-layer wiring has its own test_severity_service_integration.py):

1. app.inspections.severity - the pure, framework/DB-free scoring engine (weighting,
   boundaries, risk mapping, factor resolution). Exercised directly with explicit inputs
   so behavior is deterministic and independent of the database/AI pipeline/filesystem.
2. The inspection API (upload/import) - confirms severity fields are wired in without
   fabricating evidence, and stay independent of status/defect_category/ai_prediction.

This module's chosen missing-input strategy (see app/inspections/severity.py docstring):
a severity_score is produced ONLY when all four weighted factors have real evidence. The
Phase 2 correction review found that Defect Size CAN legitimately be derived from MVTec's
own ground-truth defect masks (see app.dataset.ground_truth) - so it is no longer always
None. Defect Location and Detection Confidence remain unavailable by explicit, documented
decision (no defensible location-to-severity rule exists, and the autoencoder has no
calibrated confidence). Since this scorer requires all four factors, assess_severity()
therefore still always returns "not assessed" for every real inspection today - that is
the correct, honest outcome given the evidence this project actually has, not a leftover
bug, and is exactly what several tests below assert.
"""

from sqlalchemy import inspect as sa_inspect

from app.inspections.severity import (
    CONFIDENCE_WEIGHT,
    DEFECT_TYPE_WEIGHT,
    LOCATION_WEIGHT,
    NOT_ASSESSED,
    SIZE_SATURATION_RATIO,
    SIZE_WEIGHT,
    assess_severity,
    calculate_severity_score,
    quality_risk_for_level,
    resolve_confidence_score,
    resolve_defect_type_score,
    resolve_location_score,
    resolve_size_score,
    severity_level_for_score,
)
from app.models.inspection import Inspection
from tests.conftest import make_image_bytes


def _import(client, headers, product_id, defect_type, filename="000.png", category="bottle", split="test"):
    return client.post(
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


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def test_factor_weights_match_specification():
    assert SIZE_WEIGHT == 30.0
    assert LOCATION_WEIGHT == 25.0
    assert DEFECT_TYPE_WEIGHT == 25.0
    assert CONFIDENCE_WEIGHT == 20.0
    assert SIZE_WEIGHT + LOCATION_WEIGHT + DEFECT_TYPE_WEIGHT + CONFIDENCE_WEIGHT == 100.0


# ---------------------------------------------------------------------------
# calculate_severity_score - sum, min, max
# ---------------------------------------------------------------------------

def test_score_reaches_100_when_all_factors_at_max():
    score = calculate_severity_score(
        size_score=SIZE_WEIGHT,
        location_score=LOCATION_WEIGHT,
        defect_type_score=DEFECT_TYPE_WEIGHT,
        confidence_score=CONFIDENCE_WEIGHT,
    )
    assert score == 100.0


def test_score_reaches_0_when_all_factors_zero():
    score = calculate_severity_score(
        size_score=0.0, location_score=0.0, defect_type_score=0.0, confidence_score=0.0
    )
    assert score == 0.0


def test_score_sums_individual_factor_contributions():
    score = calculate_severity_score(
        size_score=10.0, location_score=5.0, defect_type_score=20.0, confidence_score=2.0
    )
    assert score == 37.0


def test_score_cannot_exceed_100_even_with_out_of_range_factor_inputs():
    score = calculate_severity_score(
        size_score=SIZE_WEIGHT * 10,
        location_score=LOCATION_WEIGHT * 10,
        defect_type_score=DEFECT_TYPE_WEIGHT * 10,
        confidence_score=CONFIDENCE_WEIGHT * 10,
    )
    assert score == 100.0


def test_score_cannot_go_below_0_even_with_negative_factor_inputs():
    score = calculate_severity_score(
        size_score=-50.0, location_score=-50.0, defect_type_score=-50.0, confidence_score=-50.0
    )
    assert score == 0.0


def test_individual_factor_clamped_to_its_own_weight():
    """An oversized single factor must not leak points into the total beyond its weight."""
    score = calculate_severity_score(
        size_score=SIZE_WEIGHT * 5, location_score=0.0, defect_type_score=0.0, confidence_score=0.0
    )
    assert score == SIZE_WEIGHT


# ---------------------------------------------------------------------------
# Missing evidence -> no fabricated score
# ---------------------------------------------------------------------------

def test_score_is_none_when_any_single_factor_is_missing():
    assert calculate_severity_score(None, LOCATION_WEIGHT, DEFECT_TYPE_WEIGHT, CONFIDENCE_WEIGHT) is None
    assert calculate_severity_score(SIZE_WEIGHT, None, DEFECT_TYPE_WEIGHT, CONFIDENCE_WEIGHT) is None
    assert calculate_severity_score(SIZE_WEIGHT, LOCATION_WEIGHT, None, CONFIDENCE_WEIGHT) is None
    assert calculate_severity_score(SIZE_WEIGHT, LOCATION_WEIGHT, DEFECT_TYPE_WEIGHT, None) is None


def test_score_is_none_when_all_factors_missing():
    assert calculate_severity_score(None, None, None, None) is None


# ---------------------------------------------------------------------------
# severity_level_for_score - boundaries
# ---------------------------------------------------------------------------

def test_critical_boundary():
    assert severity_level_for_score(80) == "Critical"
    assert severity_level_for_score(100) == "Critical"


def test_high_boundary():
    assert severity_level_for_score(79) == "High"
    assert severity_level_for_score(60) == "High"


def test_medium_boundary():
    assert severity_level_for_score(59) == "Medium"
    assert severity_level_for_score(40) == "Medium"


def test_low_boundary():
    assert severity_level_for_score(39) == "Low"
    assert severity_level_for_score(0) == "Low"


def test_severity_level_is_none_for_none_score():
    assert severity_level_for_score(None) is None


# ---------------------------------------------------------------------------
# quality_risk_for_level - mapping
# ---------------------------------------------------------------------------

def test_quality_risk_mapping():
    assert quality_risk_for_level("Critical") == "Critical Risk"
    assert quality_risk_for_level("High") == "High Risk"
    assert quality_risk_for_level("Medium") == "Medium Risk"
    assert quality_risk_for_level("Low") == "Low Risk"


def test_quality_risk_not_assessed_when_level_missing():
    assert quality_risk_for_level(None) == NOT_ASSESSED == "Not assessed"


# ---------------------------------------------------------------------------
# Factor resolvers - real evidence only
# ---------------------------------------------------------------------------

def test_resolve_defect_type_score_known_categories():
    assert resolve_defect_type_score("good") == 0.0
    assert resolve_defect_type_score("broken_large") == DEFECT_TYPE_WEIGHT
    assert resolve_defect_type_score("broken_small") == 15.0
    assert resolve_defect_type_score("contamination") == 10.0


def test_resolve_defect_type_score_none_for_unknown_or_missing_category():
    assert resolve_defect_type_score(None) is None
    assert resolve_defect_type_score("some_future_mvtec_category") is None


def test_location_and_confidence_are_always_unavailable_today():
    """No location-to-severity rule and no calibrated detection confidence exists
    anywhere in this system - these must stay None, never a guessed default."""
    assert resolve_location_score() is None
    assert resolve_confidence_score() is None


# ---------------------------------------------------------------------------
# resolve_size_score - real mask area ratio -> 0-30, per the Phase 2 correction review
# ---------------------------------------------------------------------------

def test_resolve_size_score_none_when_ratio_unavailable():
    """No mask evidence (e.g. a generic upload, or an MVTec image with no matching
    ground-truth file) means the factor is unavailable, not zero."""
    assert resolve_size_score(None) is None


def test_resolve_size_score_zero_for_zero_ratio():
    """A real, measured ratio of exactly 0.0 (e.g. a "good" image - no defect to segment)
    is a real zero, not missing evidence - distinct from the None case above."""
    assert resolve_size_score(0.0) == 0.0


def test_resolve_size_score_scales_linearly_below_saturation():
    half_saturation_ratio = SIZE_SATURATION_RATIO / 2
    assert resolve_size_score(half_saturation_ratio) == SIZE_WEIGHT / 2


def test_resolve_size_score_saturates_at_max_weight():
    assert resolve_size_score(SIZE_SATURATION_RATIO) == SIZE_WEIGHT
    assert resolve_size_score(SIZE_SATURATION_RATIO * 2) == SIZE_WEIGHT
    assert resolve_size_score(1.0) == SIZE_WEIGHT  # a ratio of 1.0 (whole image) must not exceed SIZE_WEIGHT


def test_resolve_size_score_clamped_for_negative_ratio():
    assert resolve_size_score(-0.5) == 0.0


def test_resolve_size_score_always_within_0_and_weight():
    for ratio in (-10.0, -0.01, 0.0, 0.001, SIZE_SATURATION_RATIO, 0.5, 1.0, 10.0):
        score = resolve_size_score(ratio)
        assert score is not None
        assert 0.0 <= score <= SIZE_WEIGHT


# ---------------------------------------------------------------------------
# assess_severity - end-to-end engine behavior on real defect_category/size values
# ---------------------------------------------------------------------------

def test_assess_severity_not_assessed_for_known_defect_category_without_size_evidence():
    """A known defect_category (e.g. broken_large) alone is not enough: Location and
    Confidence are still unavailable, so the score stays unassessed rather than being
    computed from partial evidence (which would understate real severity)."""
    result = assess_severity("broken_large")
    assert result.score is None
    assert result.level is None
    assert result.quality_risk == "Not assessed"


def test_assess_severity_still_not_assessed_even_with_real_size_evidence():
    """This is the key correction-review assertion: even now that Defect Size can be a
    real, measured value (defect_area_ratio from an actual MVTec mask - see
    app.dataset.ground_truth), Location and Confidence are still unavailable, so the
    overall score must still be "not assessed", never a partial/misleading number."""
    result = assess_severity("broken_large", defect_area_ratio=0.109)
    assert result.score is None
    assert result.level is None
    assert result.quality_risk == "Not assessed"


def test_assess_severity_not_assessed_for_null_defect_category():
    result = assess_severity(None)
    assert result.score is None
    assert result.level is None
    assert result.quality_risk == "Not assessed"


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------

SEVERITY_COLUMNS = {"severity_score", "severity_level", "quality_risk"}


def test_inspection_model_has_severity_columns():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    assert SEVERITY_COLUMNS.issubset(columns.keys())


def test_severity_columns_are_nullable():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    for name in SEVERITY_COLUMNS:
        assert columns[name].nullable, f"{name} must be nullable"


def test_new_inspection_row_defaults_severity_fields_to_null():
    inspection = Inspection(product_id=1, image_path="bottle/test/good/000.png")
    assert inspection.severity_score is None
    assert inspection.severity_level is None
    assert inspection.quality_risk is None


# ---------------------------------------------------------------------------
# Generic upload - no fabricated severity
# ---------------------------------------------------------------------------

def test_upload_does_not_receive_fabricated_severity(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["severity_score"] is None
    assert body["severity_level"] is None
    assert body["quality_risk"] == "Not assessed"  # explicit label, per spec - never NULL


# ---------------------------------------------------------------------------
# MVTec import - severity present in response, honestly unassessed, independent of
# status / defect_category / ai_prediction
# ---------------------------------------------------------------------------

def test_import_severity_fields_present_and_unassessed(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "broken_large", filename="010.png")
    assert response.status_code == 201
    body = response.json()

    # Fields are present in the response shape (Phase 2 requirement) ...
    assert "severity_score" in body
    assert "severity_level" in body
    assert "quality_risk" in body
    # ... and honestly null/"Not assessed" per this system's current evidence limits.
    assert body["severity_score"] is None
    assert body["severity_level"] is None
    assert body["quality_risk"] == "Not assessed"


def test_severity_independent_of_status_defect_category_and_ai_prediction(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "broken_small", filename="011.png")
    assert response.status_code == 201
    body = response.json()

    assert body["status"] == "defective"  # existing ground-truth-derived status, untouched
    assert body["defect_category"] == "broken_small"  # Phase 1 field, untouched
    assert body["ai_prediction"] in ("good", "defective", None)  # independently derived
    assert body["severity_score"] is None  # not fabricated from any of the above


def test_import_good_status_and_defect_category_unaffected_by_severity(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "good", filename="012.png")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "good"
    assert body["defect_category"] == "good"


# ---------------------------------------------------------------------------
# Authorization unchanged
# ---------------------------------------------------------------------------

def test_import_still_requires_quality_engineer_role_with_severity_fields_present(
    client, supervisor_headers, test_product
):
    response = _import(client, supervisor_headers, test_product["id"], "good", filename="013.png")
    assert response.status_code == 403


def test_get_inspection_requires_authentication_with_severity_fields_present(
    client, qe_headers, test_product
):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="014.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}")
    assert response.status_code == 401


def test_supervisor_can_read_severity_fields(client, qe_headers, supervisor_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "contamination", filename="015.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}", headers=supervisor_headers)
    assert response.status_code == 200
    body = response.json()
    assert "severity_score" in body
    assert "severity_level" in body
    assert "quality_risk" in body


# ---------------------------------------------------------------------------
# Route surface unchanged
# ---------------------------------------------------------------------------

def test_phase2_adds_zero_api_routes(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operations = sum(
        1 for methods in paths.values() for m in methods if m.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert operations == 19
