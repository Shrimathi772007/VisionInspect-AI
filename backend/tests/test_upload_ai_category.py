"""AI predictions for uploaded images via the product's MVTec category.

Covers the real Tile model end to end, the best-effort failure paths (no served model, a
raising predictor), import-vs-product category precedence, the read-only
InspectionOut.product_category field (and that the list endpoint loads it without N+1
queries), and the products.category migration.

The real-model test uses the Git-ignored Tile artifact, the ResNet-18 backbone and an MVTec
tile image, and skips when any of them is absent, like the other AI serving tests. Dataset
images are only ever read; uploads go through a copy of their bytes.
"""

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import event, text

from app.ai.inference import PredictionResult
from app.ai.inference.serving import SERVING_CONFIGS, get_serving_config, get_supported_categories
from app.ai.models.resnet18 import pretrained_weights_path
from app.ai.training.artifacts import ARTIFACTS_ROOT
from app.database import SessionLocal, engine
from app.dataset.categories import MVTEC_CATEGORIES
from app.dataset.service import list_categories
from app.inspections.storage import DATASET_ROOT
from app.models.inspection import Inspection
from tests.conftest import make_image_bytes

BACKEND_DIR = Path(__file__).resolve().parent.parent
AI_FIELDS = ("ai_prediction", "ai_reconstruction_error", "ai_threshold", "ai_model_name", "ai_inference_time_ms")

TILE_MODEL_PATH = ARTIFACTS_ROOT / "tile" / "model_family_study" / "selected_candidate" / "model_state.pt"
TILE_TEST_GOOD_IMAGE = DATASET_ROOT / "tile" / "test" / "good" / "000.png"
BOTTLE_TEST_GOOD_IMAGE = DATASET_ROOT / "bottle" / "test" / "good" / "000.png"

requires_real_tile = pytest.mark.skipif(
    not (TILE_MODEL_PATH.is_file() and pretrained_weights_path().is_file() and TILE_TEST_GOOD_IMAGE.is_file()),
    reason="Tile model-family artifact, ResNet-18 backbone or MVTec tile test image not present in this environment",
)


def _fake_result(category: str, prediction: str = "good") -> PredictionResult:
    return PredictionResult(
        category=category,
        prediction=prediction,
        reconstruction_error=1.0,
        threshold=2.0,
        model_name="fake-model",
        input_size=(256, 256),
        processing_time_ms=5.0,
    )


def _assert_no_ai_result(body: dict) -> None:
    for field in AI_FIELDS:
        assert body[field] is None, field


# ---------------------------------------------------------------------------
# Real Tile model, end to end
# ---------------------------------------------------------------------------

@requires_real_tile
def test_real_tile_model_scores_an_uploaded_tile_image(category_products, tmp_path):
    # Upload a copy of the dataset image's bytes; the dataset file itself is never modified.
    copy = tmp_path / "tile_upload.png"
    copy.write_bytes(TILE_TEST_GOOD_IMAGE.read_bytes())
    product = category_products.create(category="tile")

    response = category_products.upload(product, copy.read_bytes(), filename="tile_upload.png")

    assert response.status_code == 201
    body = response.json()
    config = get_serving_config("tile")
    assert body["ai_prediction"] in {"good", "defective"}
    assert body["ai_model_name"] == config.model_name
    assert body["ai_threshold"] == config.threshold
    assert body["ai_reconstruction_error"] is not None
    assert body["ai_inference_time_ms"] > 0
    assert body["product_category"] == "tile"
    assert body["status"] == "pending"
    # Quality rules are unchanged: with no ground truth, an AI "good" alone is not sufficient
    # (NOT_ASSESSED) and an AI "defective" is a FAIL.
    assert body["quality_decision"] == ("FAIL" if body["ai_prediction"] == "defective" else "NOT_ASSESSED")


# ---------------------------------------------------------------------------
# Best-effort failure paths
# ---------------------------------------------------------------------------

def test_upload_for_category_without_served_model_leaves_ai_fields_null(client, category_products, monkeypatch):
    # Updated deliberately (all-categories registration): every MVTec category is served, so screw's registry
    # entry is removed for this test instead of skipping it.
    unserved = [c for c in MVTEC_CATEGORIES if c not in get_supported_categories()]
    if not unserved:
        monkeypatch.delitem(SERVING_CONFIGS, "screw")
        unserved = ["screw"]
    category = "screw" if "screw" in unserved else unserved[0]
    product = category_products.create(category=category)

    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    body = response.json()
    _assert_no_ai_result(body)
    assert body["product_category"] == category
    assert body["quality_decision"] == "NOT_ASSESSED"


def test_upload_survives_a_raising_predictor_without_leaking_details(category_products, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError(r"model load failed at C:\secret\ai_models\tile\model_state.pt")

    monkeypatch.setattr("app.inspections.service.predict_image", _boom)
    product = category_products.create(category="tile")

    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    _assert_no_ai_result(response.json())
    assert "secret" not in response.text
    assert "model load failed" not in response.text


def test_upload_survives_model_artifact_not_found(category_products, monkeypatch):
    # The real predictor's own error types are caught the same way as any other exception.
    from app.ai.inference import ModelArtifactNotFoundError

    def _missing(*args, **kwargs):
        raise ModelArtifactNotFoundError("No AI model is configured for category 'tile'.")

    monkeypatch.setattr("app.inspections.service.predict_image", _missing)
    product = category_products.create(category="tile")

    response = category_products.upload(product, make_image_bytes("JPEG"), filename="sample.jpg")

    assert response.status_code == 201
    _assert_no_ai_result(response.json())


# ---------------------------------------------------------------------------
# Imports: the dataset path's category wins over the product's
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not BOTTLE_TEST_GOOD_IMAGE.is_file(), reason="MVTec bottle test image not present")
def test_import_uses_dataset_path_category_not_product_category(client, qe_headers, category_products, monkeypatch):
    calls = []

    def _fake_predict(image_path, category, *args, **kwargs):
        calls.append((Path(image_path), category))
        return _fake_result(category)

    monkeypatch.setattr("app.inspections.service.predict_image", _fake_predict)
    product = category_products.create(category="tile")

    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "000.png",
        },
    )

    assert response.status_code == 201
    assert len(calls) == 1
    path, category = calls[0]
    assert category == "bottle"
    assert path == BOTTLE_TEST_GOOD_IMAGE.resolve()
    body = response.json()
    assert body["dataset_category"] == "bottle"
    assert body["product_category"] == "tile"


# ---------------------------------------------------------------------------
# InspectionOut.product_category
# ---------------------------------------------------------------------------

def test_product_category_on_detail_and_list(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda path, category, *a, **k: _fake_result(category))
    with_category = category_products.create(category="tile")
    without_category = category_products.create(category=None)
    tile_id = category_products.upload(with_category, make_image_bytes("PNG")).json()["id"]
    plain_id = category_products.upload(without_category, make_image_bytes("PNG")).json()["id"]

    detail = client.get(f"/inspections/{tile_id}", headers=qe_headers).json()
    assert detail["product_category"] == "tile"
    assert client.get(f"/inspections/{plain_id}", headers=qe_headers).json()["product_category"] is None

    listing = {i["id"]: i for i in client.get("/inspections?limit=10", headers=qe_headers).json()}
    assert listing[tile_id]["product_category"] == "tile"
    assert listing[plain_id]["product_category"] is None
    for item in listing.values():
        assert "product_category" in item
        assert "image_path" not in item


def test_inspection_list_loads_products_without_n_plus_1_queries(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda path, category, *a, **k: _fake_result(category))
    for category in ("tile", "cable", None):
        category_products.upload(category_products.create(category=category), make_image_bytes("PNG"))

    statements = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        response = client.get("/inspections?limit=50", headers=qe_headers)
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    assert response.status_code == 200
    assert len(response.json()) >= 3
    product_selects = [s for s in statements if "FROM products" in s]
    assert len(product_selects) <= 1, product_selects


# ---------------------------------------------------------------------------
# Category constant and migration
# ---------------------------------------------------------------------------

def test_mvtec_categories_constant_lists_the_fifteen_dataset_folders():
    assert len(MVTEC_CATEGORIES) == 15
    assert len(set(MVTEC_CATEGORIES)) == 15
    assert all(c == c.strip().lower() for c in MVTEC_CATEGORIES)
    if DATASET_ROOT.is_dir() and list_categories():
        assert set(list_categories()) == set(MVTEC_CATEGORIES)


def test_products_category_column_migration():
    head = ScriptDirectory.from_config(Config(str(BACKEND_DIR / "alembic.ini"))).get_current_head()
    with SessionLocal() as session:
        column = session.execute(
            text(
                "SELECT data_type, is_nullable, character_maximum_length, column_default "
                "FROM information_schema.columns WHERE table_name = 'products' AND column_name = 'category'"
            )
        ).one()
        current = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        # Every product that existed before this feature still has NULL. Products with a category
        # can only come from this feature's tests (PYTEST-CAT-*, deleted after each test).
        categorised_other = session.execute(
            text("SELECT count(*) FROM products WHERE category IS NOT NULL AND product_code NOT LIKE 'PYTEST-CAT-%'")
        ).scalar_one()

    assert current == head
    assert column.data_type == "character varying"
    assert column.is_nullable == "YES"
    assert column.character_maximum_length == 50
    assert column.column_default is None
    assert categorised_other == 0


def test_test_product_fixture_predates_categories_and_has_none(test_product):
    with SessionLocal() as session:
        from app.models.product import Product

        assert session.get(Product, test_product["id"]).category is None
    assert test_product["category"] is None


def test_inspection_row_has_no_category_column():
    # The category lives only on the product; inspections never store a copy.
    assert "category" not in Inspection.__table__.columns
