"""Milestone 3 Phase 2 correction review: service-layer severity wiring.

Exercises app.inspections.service.apply_severity_assessment / _resolve_mvtec_parts
directly (same pattern as test_inspection_ai_integration.py's run_ai_inference tests) -
confirms real MVTec ground-truth mask evidence is picked up for mvtec_ad inspections and
never fabricated for uploads, without needing the full HTTP roundtrip for every case.
"""

from app.inspections.service import _resolve_mvtec_parts, apply_severity_assessment
from app.models.inspection import Inspection, InspectionSource


class _FakeSession:
    """Minimal stand-in for a SQLAlchemy Session - only what apply_severity_assessment calls."""

    def __init__(self):
        self.committed = False
        self.refreshed = None

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        self.refreshed = obj


# ---------------------------------------------------------------------------
# _resolve_mvtec_parts
# ---------------------------------------------------------------------------

def test_resolve_mvtec_parts_for_mvtec_import():
    inspection = Inspection(
        id=1, product_id=1, image_path="bottle/test/broken_large/000.png", source=InspectionSource.mvtec_ad
    )
    assert _resolve_mvtec_parts(inspection) == ("bottle", "broken_large", "000.png")


def test_resolve_mvtec_parts_none_for_upload():
    """Uploads never carry the <category>/<split>/<defect_type>/<filename> structure -
    this must return None, never a guessed tuple."""
    inspection = Inspection(id=2, product_id=1, image_path="1/abcd1234.png", source=InspectionSource.upload)
    assert _resolve_mvtec_parts(inspection) is None


# ---------------------------------------------------------------------------
# apply_severity_assessment - real mask evidence for MVTec imports
# ---------------------------------------------------------------------------

def test_apply_severity_assessment_uses_real_mask_for_broken_large():
    """End-to-end (minus the DB/HTTP layers): a real Bottle broken_large inspection must
    resolve real ground-truth mask evidence, yet still land on "not assessed" overall
    because Location/Confidence remain unavailable - this is the central Phase 2
    correction-review assertion, exercised at the service-wiring layer."""
    inspection = Inspection(
        id=3,
        product_id=1,
        image_path="bottle/test/broken_large/000.png",
        source=InspectionSource.mvtec_ad,
        defect_category="broken_large",
    )
    session = _FakeSession()
    apply_severity_assessment(inspection, session)

    assert inspection.severity_score is None
    assert inspection.severity_level is None
    assert inspection.quality_risk == "Not assessed"
    assert session.committed is True


def test_apply_severity_assessment_for_good_mvtec_import():
    inspection = Inspection(
        id=4,
        product_id=1,
        image_path="bottle/test/good/000.png",
        source=InspectionSource.mvtec_ad,
        defect_category="good",
    )
    session = _FakeSession()
    apply_severity_assessment(inspection, session)

    assert inspection.severity_score is None
    assert inspection.quality_risk == "Not assessed"


def test_apply_severity_assessment_never_fabricates_for_upload():
    inspection = Inspection(
        id=5, product_id=1, image_path="1/abcd1234.png", source=InspectionSource.upload, defect_category=None
    )
    session = _FakeSession()
    apply_severity_assessment(inspection, session)

    assert inspection.severity_score is None
    assert inspection.severity_level is None
    assert inspection.quality_risk == "Not assessed"


def test_apply_severity_assessment_handles_missing_mask_gracefully():
    """An MVTec-shaped inspection referencing a category/defect_type/filename with no
    matching ground-truth file must not raise - the missing mask is "no evidence", not an
    error, and severity assessment must never block inspection creation."""
    inspection = Inspection(
        id=6,
        product_id=1,
        image_path="bottle/test/broken_large/does_not_exist.png",
        source=InspectionSource.mvtec_ad,
        defect_category="broken_large",
    )
    session = _FakeSession()
    apply_severity_assessment(inspection, session)  # must not raise

    assert inspection.severity_score is None
    assert inspection.quality_risk == "Not assessed"
