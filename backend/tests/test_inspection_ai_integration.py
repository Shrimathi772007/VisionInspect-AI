"""Phase 7: AI prediction integration into the inspection workflow.

Most tests here monkeypatch app.inspections.service.predict_image so they stay fast and
deterministic (Phase 6's own test suite already covers predict_image's internals). A
handful of import-flow tests use the real, trained bottle autoencoder against real MVTec
images to prove the wiring actually works end to end - they do not assert anything about
model accuracy, only that AI fields get populated (or safely don't) as expected.
"""

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from app.inspections.service import run_ai_inference
from app.models.inspection import Inspection, InspectionSource
from tests.conftest import make_image_bytes

AI_COLUMNS = {"ai_prediction", "ai_reconstruction_error", "ai_threshold", "ai_model_name"}


# ---------------------------------------------------------------------------
# Model / schema shape
# ---------------------------------------------------------------------------

def test_inspection_model_has_ai_columns():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    assert AI_COLUMNS.issubset(columns.keys())


def test_ai_columns_are_nullable():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    for name in AI_COLUMNS:
        assert columns[name].nullable, f"{name} must be nullable"


def test_new_inspection_row_defaults_ai_fields_to_null():
    inspection = Inspection(product_id=1, image_path="bottle/test/good/000.png", source=InspectionSource.mvtec_ad)
    assert inspection.ai_prediction is None
    assert inspection.ai_reconstruction_error is None
    assert inspection.ai_threshold is None
    assert inspection.ai_model_name is None


# ---------------------------------------------------------------------------
# run_ai_inference - category resolution / no-op paths (monkeypatched predict_image)
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal stand-in for a SQLAlchemy Session - only what run_ai_inference calls."""

    def __init__(self):
        self.committed = False
        self.refreshed = None

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        self.refreshed = obj


def test_upload_source_never_calls_predict_image(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("predict_image must not be called for an upload with no derivable category")

    monkeypatch.setattr("app.inspections.service.predict_image", _fail_if_called)

    inspection = Inspection(id=1, product_id=1, image_path="1/abcd.png", source=InspectionSource.upload)
    run_ai_inference(inspection, _FakeSession())

    assert inspection.ai_prediction is None
    assert inspection.ai_reconstruction_error is None
    assert inspection.ai_threshold is None
    assert inspection.ai_model_name is None


def test_mvtec_import_with_unsupported_category_leaves_fields_null(monkeypatch):
    # Simulate "no trained model" the same way predict_image itself would report it -
    # by making the load raise, exactly as it does for a real untrained category.
    from app.ai.inference.errors import ModelArtifactNotFoundError

    def _raise_not_found(*args, **kwargs):
        raise ModelArtifactNotFoundError("no trained model for category 'cable'")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise_not_found)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path",
        lambda inspection: __import__("pathlib").Path("unused"),
    )

    inspection = Inspection(
        id=2, product_id=1, image_path="cable/test/good/000.png", source=InspectionSource.mvtec_ad
    )
    run_ai_inference(inspection, _FakeSession())

    assert inspection.ai_prediction is None
    assert inspection.ai_reconstruction_error is None
    assert inspection.ai_threshold is None
    assert inspection.ai_model_name is None


def test_run_ai_inference_populates_fields_on_success(monkeypatch):
    from app.ai.inference import PredictionResult

    fake_result = PredictionResult(
        category="bottle",
        prediction="defective",
        reconstruction_error=0.0041,
        threshold=0.003212,
        model_name="autoencoder",
        input_size=(128, 128),
        processing_time_ms=12.3,
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path",
        lambda inspection: __import__("pathlib").Path("unused"),
    )

    inspection = Inspection(
        id=3, product_id=1, image_path="bottle/test/broken_large/000.png", source=InspectionSource.mvtec_ad
    )
    session = _FakeSession()
    run_ai_inference(inspection, session)

    assert inspection.ai_prediction == "defective"
    assert inspection.ai_reconstruction_error == 0.0041
    assert inspection.ai_threshold == 0.003212
    assert inspection.ai_model_name == "autoencoder"
    assert session.committed is True


def test_run_ai_inference_failure_does_not_overwrite_existing_values(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path",
        lambda inspection: __import__("pathlib").Path("unused"),
    )

    inspection = Inspection(
        id=4, product_id=1, image_path="bottle/test/good/000.png", source=InspectionSource.mvtec_ad
    )
    inspection.ai_prediction = "good"
    inspection.ai_reconstruction_error = 0.001
    inspection.ai_threshold = 0.003212
    inspection.ai_model_name = "autoencoder"

    session = _FakeSession()
    run_ai_inference(inspection, session)  # must not raise

    assert inspection.ai_prediction == "good"
    assert inspection.ai_reconstruction_error == 0.001
    assert inspection.ai_threshold == 0.003212
    assert inspection.ai_model_name == "autoencoder"
    assert session.committed is False


# ---------------------------------------------------------------------------
# Router integration - upload path (no derivable category)
# ---------------------------------------------------------------------------

def test_upload_still_succeeds_with_null_ai_fields(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["ai_prediction"] is None
    assert body["ai_reconstruction_error"] is None
    assert body["ai_threshold"] is None
    assert body["ai_model_name"] is None
    assert "image_path" not in body


def test_router_import_succeeds_even_if_ai_inference_raises(monkeypatch, client, qe_headers, test_product):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise)

    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "000.png",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["ai_prediction"] is None

    # The inspection itself must still be there - inference failure did not roll it back.
    get_response = client.get(f"/inspections/{body['id']}", headers=qe_headers)
    assert get_response.status_code == 200


# ---------------------------------------------------------------------------
# Router integration - MVTec import, REAL bottle model (sanity check, no mocking)
# ---------------------------------------------------------------------------

def test_mvtec_bottle_import_runs_real_ai_inference(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "000.png",
        },
    )
    assert response.status_code == 201
    body = response.json()

    assert body["ai_prediction"] in ("good", "defective")
    assert isinstance(body["ai_reconstruction_error"], float)
    assert isinstance(body["ai_threshold"], float)
    assert body["ai_reconstruction_error"] >= 0.0
    assert body["ai_model_name"] == "autoencoder"


def test_mvtec_import_without_trained_model_succeeds_with_null_ai_fields(client, qe_headers, test_product):
    """cable has no trained artifact under ai_models/ - import must still succeed."""
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "cable",
            "split": "test",
            "defect_type": "good",
            "filename": "000.png",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["ai_prediction"] is None
    assert body["ai_reconstruction_error"] is None
    assert body["ai_threshold"] is None
    assert body["ai_model_name"] is None


def test_ground_truth_and_ai_prediction_remain_independent(client, qe_headers, test_product):
    """A known-defective MVTec image may legitimately get an AI 'good' prediction (Phase 5
    measured 46% recall for this baseline model) - the API must represent both values
    independently, never reconciling one with the other."""
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "broken_large",
            "filename": "000.png",
        },
    )
    assert response.status_code == 201
    body = response.json()

    assert body["dataset_defect_type"] == "broken_large"  # MVTec ground truth, untouched
    assert body["status"] == "defective"  # existing ground-truth-derived status, untouched
    assert body["ai_prediction"] in ("good", "defective")  # independently derived - not forced to match


# ---------------------------------------------------------------------------
# Route surface unchanged
# ---------------------------------------------------------------------------

def test_phase7_added_zero_api_routes(client):
    # Phase 8 adds exactly one new operation (GET /inspections/analytics/summary), and
    # Milestone 3 Phase 4 adds exactly one more (GET /inspections/{id}/report) - the count
    # below reflects both, not a Phase 7 regression.
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operations = sum(
        1 for methods in paths.values() for m in methods if m.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert operations == 20
    assert set(paths.keys()) == {
        "/auth/register",
        "/auth/login",
        "/auth/me",
        "/products",
        "/products/{product_id}",
        "/inspections",
        "/inspections/analytics/summary",
        "/inspections/upload",
        "/inspections/import",
        "/inspections/{inspection_id}",
        "/inspections/{inspection_id}/report",
        "/inspections/{inspection_id}/image",
        "/dataset/categories",
        "/dataset/categories/{category}",
        "/dataset/categories/{category}/images",
        "/dataset/categories/{category}/preview",
        "/health",
        "/health/db",
    }
