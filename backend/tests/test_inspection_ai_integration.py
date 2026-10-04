"""Phase 7: AI prediction integration into the inspection workflow.

Most tests here monkeypatch app.inspections.service.predict_image so they stay fast and
deterministic (Phase 6's own test suite already covers predict_image's internals). A
handful of import-flow tests use the real, trained bottle autoencoder against real MVTec
images to prove the wiring actually works end to end - they do not assert anything about
model accuracy, only that AI fields get populated (or safely don't) as expected.
"""

from pathlib import Path

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


# Uploads get an AI prediction only through their product's MVTec category (see
# app.inspections.service._resolve_category). These two replace the earlier
# test_upload_source_never_calls_predict_image, which asserted that uploads were never scored.

def test_upload_for_product_without_category_never_calls_predict_image(
    client, category_products, monkeypatch
):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("predict_image must not be called for a product with no category")

    monkeypatch.setattr("app.inspections.service.predict_image", _fail_if_called)
    product = category_products.create(category=None)

    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    body = response.json()
    for field in AI_COLUMNS | {"ai_inference_time_ms"}:
        assert body[field] is None, field
    assert body["product_category"] is None
    assert body["status"] == "pending"
    assert body["quality_decision"] == "NOT_ASSESSED"


def test_upload_for_product_with_category_is_scored_and_assessed(client, category_products, monkeypatch):
    from app.ai.inference import PredictionResult
    from app.database import SessionLocal
    from app.inspections import router as inspections_router
    from app.inspections.storage import STORAGE_ROOT

    calls = []

    def _fake_predict(image_path, category, *args, **kwargs):
        calls.append({"path": Path(image_path), "category": category, "existed": Path(image_path).is_file()})
        return PredictionResult(
            category=category,
            prediction="defective",
            reconstruction_error=2.5,
            threshold=1.7393077017650718,
            model_name="knn_l23_256",
            input_size=(256, 256),
            processing_time_ms=42.0,
        )

    steps = []
    for name in ("apply_severity_assessment", "apply_quality_assessment"):
        original = getattr(inspections_router, name)

        def _spy(inspection, db, _original=original, _name=name):
            steps.append(_name)
            return _original(inspection, db)

        monkeypatch.setattr(inspections_router, name, _spy)
    monkeypatch.setattr("app.inspections.service.predict_image", _fake_predict)

    product = category_products.create(category="tile")
    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    body = response.json()

    # Called exactly once, with the product's category and the stored upload file.
    assert len(calls) == 1
    assert calls[0]["category"] == "tile"
    with SessionLocal() as session:
        stored = session.get(Inspection, body["id"])
        expected_path = (STORAGE_ROOT / stored.image_path).resolve()
        assert stored.ai_prediction == "defective"
        assert stored.ai_reconstruction_error == 2.5
        assert stored.ai_threshold == 1.7393077017650718
        assert stored.ai_model_name == "knn_l23_256"
        assert stored.ai_inference_time_ms == 42.0
    assert calls[0]["path"] == expected_path
    assert calls[0]["path"].is_relative_to(STORAGE_ROOT)
    assert calls[0]["existed"] is True

    # The same post-processing as imports, in order, after inference.
    assert steps == ["apply_severity_assessment", "apply_quality_assessment"]
    assert body["ai_prediction"] == "defective"
    assert body["product_category"] == "tile"
    assert body["status"] == "pending"
    assert body["severity_level"] is None  # uploads have no defect_category -> "not assessed"
    assert body["quality_decision"] == "FAIL"
    assert body["quality_assessment"]
    assert body["processing_time_ms"] is not None


def test_mvtec_import_with_unsupported_category_leaves_fields_null(monkeypatch):
    # Simulate "no trained model" the same way predict_image itself would report it -
    # by making the load raise, exactly as it does for a real untrained category.
    from app.ai.inference.errors import ModelArtifactNotFoundError

    def _raise_not_found(*args, **kwargs):
        raise ModelArtifactNotFoundError("no trained model for category 'capsule'")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise_not_found)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path",
        lambda inspection: __import__("pathlib").Path("unused"),
    )

    inspection = Inspection(
        id=2, product_id=1, image_path="capsule/test/good/000.png", source=InspectionSource.mvtec_ad
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
# Updated deliberately (all-categories registration): Bottle is served by its WRN-50 PatchCore model, and every
# category's test/ split is its consumed final test - so these imports use train/good images.
# ---------------------------------------------------------------------------

def test_mvtec_bottle_import_runs_real_ai_inference(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "train",
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
    assert body["ai_model_name"] == "wrn50_patchcore_crop224"
    # Served from the locked WRN-50 configuration (app.ai.inference.serving), not a threshold
    # recomputed from train/good.
    assert body["ai_threshold"] == 1.6858729828595898


def test_mvtec_import_without_trained_model_succeeds_with_null_ai_fields(client, qe_headers, test_product, monkeypatch):
    """A category with no served model - import must still succeed. Every MVTec category is served since the
    all-categories registration, so capsule's registry entry is removed for this test (updated deliberately;
    previously capsule simply had no model). train/good: test/ images are each category's consumed final test."""
    from app.ai.inference.serving import SERVING_CONFIGS

    monkeypatch.delitem(SERVING_CONFIGS, "capsule")
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "capsule",
            "split": "train",
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


def test_ground_truth_and_ai_prediction_remain_independent(client, qe_headers, test_product, monkeypatch):
    """A known-defective MVTec image may legitimately get an AI 'good' prediction (Phase 5
    measured 46% recall for the baseline model) - the API must represent both values
    independently, never reconciling one with the other.

    Updated deliberately (all-categories registration): this image is part of Bottle's consumed final test, which
    no served model may score again, so the AI result is stubbed as 'good' - the disagreement this test is about."""
    from app.ai.inference import PredictionResult

    stub = PredictionResult(category="bottle", prediction="good", reconstruction_error=1.0, threshold=1.6858729828595898,
                            model_name="wrn50_patchcore_crop224", input_size=(224, 224), processing_time_ms=1.0)
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: stub)
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
    assert body["ai_prediction"] == "good"  # independently derived - not forced to match


# ---------------------------------------------------------------------------
# Route surface unchanged
# ---------------------------------------------------------------------------

def test_phase7_added_zero_api_routes(client):
    # Phase 8 adds exactly one new operation (GET /inspections/analytics/summary), and
    # Milestone 3 Phase 4 adds exactly one more (GET /inspections/{id}/report) - the count
    # below reflects both, not a Phase 7 regression.
    # The role-escalation fix adds exactly two more (GET /users, PATCH /users/{user_id}/role).
    # The upload-AI change adds exactly one more (PATCH /products/{product_id}/category).
    # The all-categories serving change adds exactly one more (GET /ai/models).
    # The localization change adds exactly one more (GET /inspections/{inspection_id}/heatmap).
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operations = sum(
        1 for methods in paths.values() for m in methods if m.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert operations == 25
    assert set(paths.keys()) == {
        "/ai/models",
        "/auth/register",
        "/auth/login",
        "/auth/me",
        "/products",
        "/products/{product_id}",
        "/products/{product_id}/category",
        "/inspections",
        "/inspections/analytics/summary",
        "/inspections/upload",
        "/inspections/import",
        "/inspections/{inspection_id}",
        "/inspections/{inspection_id}/report",
        "/inspections/{inspection_id}/image",
        "/inspections/{inspection_id}/heatmap",
        "/dataset/categories",
        "/dataset/categories/{category}",
        "/dataset/categories/{category}/images",
        "/dataset/categories/{category}/preview",
        "/health",
        "/health/db",
        "/users",
        "/users/{user_id}/role",
    }
