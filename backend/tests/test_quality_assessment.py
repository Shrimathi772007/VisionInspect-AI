"""Milestone 3 Phase 3: quality assessment and recommendations.

Three layers under test:

1. app.inspections.quality.assess_quality - the pure, framework/DB-free decision engine
   (evidence-precedence rules, recommendation selection). Exercised directly with
   explicit inputs so behavior is deterministic and independent of the database/AI/
   severity pipelines.
2. app.inspections.service.apply_quality_assessment - service-layer wiring, exercised
   with constructed Inspection objects (same pattern as
   test_severity_service_integration.py) so ai_prediction/severity_level can be
   controlled without needing a real model run.
3. The inspection API (upload/import) - confirms quality fields are wired into the
   response and stay independent of status/ai_prediction/defect_category/severity.
"""

from sqlalchemy import inspect as sa_inspect

from app.inspections.quality import FAIL, NOT_ASSESSED, PASS, assess_quality
from app.inspections.service import apply_quality_assessment
from app.models.inspection import Inspection, InspectionSource
from tests.conftest import make_image_bytes

QUALITY_COLUMNS = {"quality_decision", "quality_assessment", "quality_recommendation"}


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
# Decision values
# ---------------------------------------------------------------------------

def test_decision_constants_are_exact_strings():
    assert PASS == "PASS"
    assert FAIL == "FAIL"
    assert NOT_ASSESSED == "NOT_ASSESSED"


def test_decision_is_not_a_copy_of_status():
    """status="pending" is a real, valid status value - it must never itself equal a
    quality decision string, proving this is a distinct vocabulary, not a passthrough."""
    result = assess_quality(status="pending", ai_prediction=None, defect_category=None, severity_level=None)
    assert result.decision != "pending"


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------

def test_pass_for_good_ground_truth_alone():
    result = assess_quality(status="good", ai_prediction=None, defect_category="good", severity_level=None)
    assert result.decision == PASS


def test_pass_for_good_ground_truth_with_agreeing_ai():
    result = assess_quality(status="good", ai_prediction="good", defect_category="good", severity_level=None)
    assert result.decision == PASS


def test_pass_recommendation_is_generic_proceed_message():
    result = assess_quality(status="good", ai_prediction="good", defect_category="good", severity_level=None)
    assert result.recommendation == "Product can proceed to the next quality-control stage."


# ---------------------------------------------------------------------------
# FAIL
# ---------------------------------------------------------------------------

def test_fail_for_defective_ground_truth_alone():
    result = assess_quality(
        status="defective", ai_prediction=None, defect_category="broken_large", severity_level=None
    )
    assert result.decision == FAIL


def test_fail_for_ai_defective_alone_no_ground_truth():
    """A pending/no-ground-truth inspection where the AI flags an anomaly is not PASS -
    treated as FAIL, since it is the system's only defect signal and nothing contradicts
    it (see app.inspections.quality's rule 3 docstring)."""
    result = assess_quality(status="pending", ai_prediction="defective", defect_category=None, severity_level=None)
    assert result.decision == FAIL


def test_fail_for_ground_truth_and_ai_agreeing_defective():
    result = assess_quality(
        status="defective", ai_prediction="defective", defect_category="contamination", severity_level=None
    )
    assert result.decision == FAIL


def test_fail_assessment_message_matches_expected_wording():
    result = assess_quality(
        status="defective", ai_prediction="defective", defect_category="broken_large", severity_level=None
    )
    assert result.assessment == "Defective product requires review."


# ---------------------------------------------------------------------------
# NOT_ASSESSED - insufficient evidence
# ---------------------------------------------------------------------------

def test_not_assessed_when_no_evidence_at_all():
    result = assess_quality(status="pending", ai_prediction=None, defect_category=None, severity_level=None)
    assert result.decision == NOT_ASSESSED
    assert result.assessment == "Insufficient evidence."
    assert result.recommendation == "Additional inspection evidence is required before making a quality decision."


def test_not_assessed_for_ai_good_alone_without_ground_truth():
    """An AI "good" prediction alone, with no ground truth, is not sufficient for PASS -
    this autoencoder has known poor recall on real defects (see the module docstring)."""
    result = assess_quality(status="pending", ai_prediction="good", defect_category=None, severity_level=None)
    assert result.decision == NOT_ASSESSED


# ---------------------------------------------------------------------------
# NOT_ASSESSED - conflicting evidence
# ---------------------------------------------------------------------------

def test_not_assessed_for_conflicting_good_status_defective_ai():
    result = assess_quality(status="good", ai_prediction="defective", defect_category="good", severity_level=None)
    assert result.decision == NOT_ASSESSED
    assert result.assessment == "Conflicting inspection evidence requires review."
    assert result.recommendation == "Conflicting inspection evidence requires quality review."


def test_not_assessed_for_conflicting_defective_status_good_ai():
    result = assess_quality(
        status="defective", ai_prediction="good", defect_category="broken_small", severity_level=None
    )
    assert result.decision == NOT_ASSESSED
    assert result.assessment == "Conflicting inspection evidence requires review."


def test_conflicting_evidence_never_becomes_pass():
    for status, ai in (("good", "defective"), ("defective", "good")):
        result = assess_quality(status=status, ai_prediction=ai, defect_category=None, severity_level=None)
        assert result.decision != PASS


# ---------------------------------------------------------------------------
# Severity overrides (rule 1 - highest precedence)
# ---------------------------------------------------------------------------

def test_critical_severity_forces_fail_even_with_agreeing_good_evidence():
    result = assess_quality(status="good", ai_prediction="good", defect_category="good", severity_level="Critical")
    assert result.decision == FAIL
    assert result.assessment == "High-severity defect evidence requires review."


def test_high_severity_forces_fail():
    result = assess_quality(status="good", ai_prediction="good", defect_category="good", severity_level="High")
    assert result.decision == FAIL


def test_medium_and_low_severity_do_not_force_fail():
    for level in ("Medium", "Low"):
        result = assess_quality(status="good", ai_prediction="good", defect_category="good", severity_level=level)
        assert result.decision == PASS


# ---------------------------------------------------------------------------
# Recommendations - defect-category-specific text
# ---------------------------------------------------------------------------

def test_structural_defect_recommendation_for_broken_large():
    result = assess_quality(
        status="defective", ai_prediction=None, defect_category="broken_large", severity_level=None
    )
    assert "structural damage" in result.recommendation


def test_structural_defect_recommendation_for_broken_small():
    result = assess_quality(
        status="defective", ai_prediction=None, defect_category="broken_small", severity_level=None
    )
    assert "structural damage" in result.recommendation


def test_contamination_recommendation():
    result = assess_quality(
        status="defective", ai_prediction=None, defect_category="contamination", severity_level=None
    )
    assert "contamination" in result.recommendation


def test_unknown_defect_category_falls_back_to_generic_recommendation():
    result = assess_quality(
        status="defective", ai_prediction=None, defect_category="some_future_defect_type", severity_level=None
    )
    assert result.decision == FAIL
    assert result.recommendation  # never empty


def test_ai_review_recommendation_included_when_ai_flags_defective():
    result = assess_quality(
        status="defective", ai_prediction="defective", defect_category="broken_large", severity_level=None
    )
    assert "Review the detected anomaly" in result.recommendation
    assert "structural damage" in result.recommendation  # both apply, neither hidden


def test_recommendation_priority_severity_escalation_not_hidden_by_defect_specific_text():
    """The exact scenario the phase brief calls out: High/Critical severity must not be
    accidentally hidden by a more generic defect-specific recommendation."""
    result = assess_quality(
        status="defective", ai_prediction="defective", defect_category="contamination", severity_level="Critical"
    )
    assert result.recommendation.startswith("Escalate the inspection for quality review.")
    assert "contamination" in result.recommendation


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_recommendations_are_deterministic():
    args = dict(status="defective", ai_prediction="defective", defect_category="broken_large", severity_level="High")
    assert assess_quality(**args) == assess_quality(**args)


# ---------------------------------------------------------------------------
# Model / schema shape
# ---------------------------------------------------------------------------

def test_inspection_model_has_quality_columns():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    assert QUALITY_COLUMNS.issubset(columns.keys())


def test_quality_columns_are_nullable():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    for name in QUALITY_COLUMNS:
        assert columns[name].nullable, f"{name} must be nullable"


def test_new_inspection_row_defaults_quality_fields_to_null():
    inspection = Inspection(product_id=1, image_path="bottle/test/good/000.png")
    assert inspection.quality_decision is None
    assert inspection.quality_assessment is None
    assert inspection.quality_recommendation is None


# ---------------------------------------------------------------------------
# apply_quality_assessment - service-layer wiring (controlled inputs, no real AI/severity)
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal stand-in for a SQLAlchemy Session - only what apply_quality_assessment calls."""

    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass


def test_apply_quality_assessment_writes_all_three_fields_and_commits():
    inspection = Inspection(
        id=1,
        product_id=1,
        image_path="bottle/test/good/000.png",
        source=InspectionSource.mvtec_ad,
        status="good",
        defect_category="good",
        ai_prediction="good",
    )
    session = _FakeSession()
    apply_quality_assessment(inspection, session)

    assert inspection.quality_decision == PASS
    assert inspection.quality_assessment
    assert inspection.quality_recommendation
    assert session.committed is True


def test_apply_quality_assessment_never_touches_status_ai_prediction_or_defect_category():
    inspection = Inspection(
        id=2,
        product_id=1,
        image_path="bottle/test/broken_large/000.png",
        source=InspectionSource.mvtec_ad,
        status="defective",
        defect_category="broken_large",
        ai_prediction="defective",
    )
    session = _FakeSession()
    apply_quality_assessment(inspection, session)

    assert inspection.status == "defective"
    assert inspection.defect_category == "broken_large"
    assert inspection.ai_prediction == "defective"
    assert inspection.quality_decision == FAIL


def test_apply_quality_assessment_for_upload_with_no_evidence():
    inspection = Inspection(
        id=3,
        product_id=1,
        image_path="1/abcd1234.png",
        source=InspectionSource.upload,
        status="pending",
        defect_category=None,
        ai_prediction=None,
        severity_level=None,
    )
    session = _FakeSession()
    apply_quality_assessment(inspection, session)

    assert inspection.quality_decision == NOT_ASSESSED
    assert inspection.quality_assessment == "Insufficient evidence."


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def test_upload_quality_fields_present_and_not_assessed(client, qe_headers, test_product):
    """Generic uploads never get an AI prediction (no derivable category - see
    app.inspections.service._resolve_category) and never have a defect_category, so the
    quality decision is deterministically NOT_ASSESSED - never a fabricated PASS/FAIL."""
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["quality_decision"] == NOT_ASSESSED
    assert body["quality_assessment"] == "Insufficient evidence."
    assert body["ai_prediction"] is None
    assert body["defect_category"] is None


def test_import_quality_fields_present(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "good", filename="005.png")
    assert response.status_code == 201
    body = response.json()
    assert "quality_decision" in body
    assert "quality_assessment" in body
    assert "quality_recommendation" in body
    assert body["quality_decision"] in (PASS, FAIL, NOT_ASSESSED)


def test_import_quality_decision_conflict_when_ai_disagrees_with_ground_truth(
    client, qe_headers, test_product, monkeypatch
):
    """Force a known conflict (ground truth good, AI says defective) and confirm the API
    surfaces NOT_ASSESSED rather than silently trusting either side."""
    from app.ai.inference.schemas import DEFECTIVE_PREDICTION, PredictionResult

    fake_result = PredictionResult(
        category="bottle",
        prediction=DEFECTIVE_PREDICTION,
        reconstruction_error=0.5,
        threshold=0.1,
        model_name="autoencoder",
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)

    response = _import(client, qe_headers, test_product["id"], "good", filename="006.png")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "good"
    assert body["ai_prediction"] == "defective"
    assert body["quality_decision"] == NOT_ASSESSED
    assert body["quality_assessment"] == "Conflicting inspection evidence requires review."


def test_import_quality_decision_pass_when_ai_agrees_with_ground_truth(
    client, qe_headers, test_product, monkeypatch
):
    from app.ai.inference.schemas import GOOD_PREDICTION, PredictionResult

    fake_result = PredictionResult(
        category="bottle", prediction=GOOD_PREDICTION, reconstruction_error=0.01, threshold=0.1, model_name="autoencoder"
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)

    response = _import(client, qe_headers, test_product["id"], "good", filename="007.png")
    assert response.status_code == 201
    assert response.json()["quality_decision"] == PASS


def test_import_quality_decision_fail_for_defective_category_without_ai(
    client, qe_headers, test_product, monkeypatch
):
    from app.ai.inference.errors import ModelArtifactNotFoundError

    def _raise(*args, **kwargs):
        raise ModelArtifactNotFoundError("no model for test")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise)

    response = _import(client, qe_headers, test_product["id"], "contamination", filename="008.png")
    assert response.status_code == 201
    body = response.json()
    assert body["ai_prediction"] is None  # AI unavailable in this test
    assert body["status"] == "defective"
    assert body["quality_decision"] == FAIL
    assert "contamination" in body["quality_recommendation"]


# ---------------------------------------------------------------------------
# Regression: existing fields unaffected by quality assessment
# ---------------------------------------------------------------------------

def test_status_ai_prediction_defect_category_unchanged_by_quality_assessment(
    client, qe_headers, test_product, monkeypatch
):
    from app.ai.inference.errors import ModelArtifactNotFoundError

    def _raise(*args, **kwargs):
        raise ModelArtifactNotFoundError("no model for test")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise)

    response = _import(client, qe_headers, test_product["id"], "broken_small", filename="009.png")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "defective"
    assert body["defect_category"] == "broken_small"
    assert body["ai_prediction"] is None


def test_severity_fields_unchanged_by_quality_assessment(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "broken_large", filename="010.png")
    assert response.status_code == 201
    body = response.json()
    assert body["severity_score"] is None
    assert body["quality_risk"] == "Not assessed"


# ---------------------------------------------------------------------------
# Authorization unchanged
# ---------------------------------------------------------------------------

def test_import_still_requires_quality_engineer_role_with_quality_fields_present(
    client, supervisor_headers, test_product
):
    response = _import(client, supervisor_headers, test_product["id"], "good", filename="011.png")
    assert response.status_code == 403


def test_get_inspection_requires_authentication_with_quality_fields_present(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="012.png")
    inspection_id = import_response.json()["id"]
    response = client.get(f"/inspections/{inspection_id}")
    assert response.status_code == 401


def test_supervisor_can_read_quality_fields(client, qe_headers, supervisor_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "contamination", filename="013.png")
    inspection_id = import_response.json()["id"]
    response = client.get(f"/inspections/{inspection_id}", headers=supervisor_headers)
    assert response.status_code == 200
    body = response.json()
    assert "quality_decision" in body
    assert "quality_assessment" in body
    assert "quality_recommendation" in body


# ---------------------------------------------------------------------------
# Route surface unchanged
# ---------------------------------------------------------------------------

def test_phase3_adds_zero_api_routes(client):
    # Milestone 3 Phase 4 later adds exactly one new operation (GET
    # /inspections/{id}/report) - the count below reflects that, not a Phase 3 regression.
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operations = sum(
        1 for methods in paths.values() for m in methods if m.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert operations == 20
