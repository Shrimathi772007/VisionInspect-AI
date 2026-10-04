"""Localization, confidence and manual review end to end: score identity with the real served models, uploads
through the API, GET /inspections/{id}/heatmap, heatmap deletion, best-effort failure paths and the migration.

Real-model tests use ONLY dataset/<category>/train/good/ images, copied to temporary files first (never the
test/ split), and skip when the artifacts, backbones or images are absent, like the other serving tests.
"""

import shutil
from pathlib import Path

import pytest
import torch
from alembic.config import Config
from alembic.script import ScriptDirectory
from PIL import Image
from sqlalchemy import text

from app.ai.inference import PredictionResult, predict_image
from app.ai.inference.serving import SERVING_CONFIGS, load_serving_model
from app.ai.models.resnet18 import pretrained_weights_path
from app.ai.models.wide_resnet50 import wide_resnet50_weights_path
from app.ai.training.artifacts import get_model_path
from app.database import SessionLocal
from app.inspections.storage import DATASET_ROOT, HEATMAP_ROOT
from app.models.inspection import Inspection
from tests.conftest import make_image_bytes

BACKEND_DIR = Path(__file__).resolve().parent.parent
NEW_FIELDS = ("ai_confidence", "ai_reliability", "review_required", "review_reason", "localization", "model_gate")
NEW_COLUMNS = ("ai_confidence", "review_required", "review_reason", "localization", "heatmap_path")


def _train_good(category: str) -> Path:
    return Path(DATASET_ROOT) / category / "train" / "good" / "000.png"


def _present(category: str) -> bool:
    config = SERVING_CONFIGS[category]
    path = get_model_path(category, config.artifact_name)
    return (path.is_file() and (path.parent / "model_config.json").is_file() and _train_good(category).is_file()
            and pretrained_weights_path().is_file() and wide_resnet50_weights_path().is_file())


def _requires(*categories):
    return pytest.mark.skipif(not all(_present(c) for c in categories),
                              reason=f"model artifacts / backbones / train/good images for {categories} not present")


def _copy_train_good(category: str, tmp_path: Path) -> Path:
    copy = tmp_path / f"{category}_train_good_000.png"
    shutil.copyfile(_train_good(category), copy)
    return copy


# ---------------------------------------------------------------------------
# Score identity: the image score and prediction are bit-identical with and without the patch grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", [
    pytest.param("tile", marks=_requires("tile")),      # ResNet-18 kNN, resize 256
    pytest.param("leather", marks=_requires("leather")),  # ResNet-18 Gaussian, resize 128
    pytest.param("bottle", marks=_requires("bottle")),  # WRN-50 crop224
    pytest.param("zipper", marks=_requires("zipper")),  # WRN-50 full256
    pytest.param("grid", marks=_requires("grid")),      # WRN-50 full320 (Addendum 1 loader)
])
def test_score_and_prediction_are_bit_identical_to_the_pre_change_path(category, tmp_path):
    image = _copy_train_good(category, tmp_path)
    plain = predict_image(image, category)
    with_grid = predict_image(image, category, return_patch_scores=True)

    assert plain.patch_scores is None and plain.input_mode is None
    assert with_grid.reconstruction_error == plain.reconstruction_error  # exact float equality
    assert type(with_grid.reconstruction_error) is type(plain.reconstruction_error) is float
    assert with_grid.prediction == plain.prediction
    assert with_grid.threshold == plain.threshold == SERVING_CONFIGS[category].threshold

    config = SERVING_CONFIGS[category]
    grid = with_grid.patch_scores
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1]
    if config.model_family == "patch_anomaly":
        # ...and equal to the detector's own score_images on the same tensor (the pre-change ResNet-18 call).
        from app.ai.inference.predict import _image_to_tensor
        model = load_serving_model(config)
        assert model.score_images(_image_to_tensor(image, config.input_size).unsqueeze(0))[0] == plain.reconstruction_error
        assert with_grid.input_mode == "resize"
    else:
        assert with_grid.input_mode == config.input_mode
        assert grid.shape[0] == config.input_size[0] // 8
        assert float(grid.max()) == plain.reconstruction_error  # WRN-50 aggregation is "max"


# ---------------------------------------------------------------------------
# Uploads through the API
# ---------------------------------------------------------------------------

def _heatmap_file(inspection_id: int) -> Path:
    with SessionLocal() as session:
        relative = session.get(Inspection, inspection_id).heatmap_path
    return HEATMAP_ROOT / relative


@_requires("tile")
def test_tile_upload_is_localized_with_confidence_and_a_heatmap(client, qe_headers, category_products, tmp_path):
    image = _copy_train_good("tile", tmp_path)
    product = category_products.create(category="tile")

    response = category_products.upload(product, image.read_bytes(), filename="tile.png")

    assert response.status_code == 201
    body = response.json()
    assert "heatmap_path" not in body and "storage" not in response.text
    config = SERVING_CONFIGS["tile"]
    assert body["ai_threshold"] == config.threshold
    assert 0.5 <= body["ai_confidence"] < 1.0
    assert body["ai_reliability"] in {"high", "medium", "low"}
    assert body["model_gate"] == "EXCELLENT"
    assert body["review_required"] is (body["ai_confidence"] < 0.70)
    loc = body["localization"]
    assert loc["method"] == "anomaly_map_threshold_v1"
    assert loc["analysed_region"] == [0.0, 0.0, 1.0, 1.0]
    if body["ai_prediction"] == "good":
        assert loc["boxes"] == [] and loc["area_pct"] == 0.0
    expected_decision = ("MANUAL_REVIEW" if body["review_required"]
                         else "FAIL" if body["ai_prediction"] == "defective" else "PASS")
    assert body["quality_decision"] == expected_decision
    assert body["has_heatmap"] is True

    heatmap = client.get(f"/inspections/{body['id']}/heatmap", headers=qe_headers)
    assert heatmap.status_code == 200
    assert heatmap.headers["content-type"] == "image/png"
    with Image.open(image) as original:
        width, height = original.size
    png = tmp_path / "heatmap.png"
    png.write_bytes(heatmap.content)
    with Image.open(png) as rendered:
        assert rendered.mode == "RGBA"
        scale = min(1.0, 320 / max(width, height))
        assert rendered.size == (round(width * scale), round(height * scale))

    report = client.get(f"/inspections/{body['id']}/report", headers=qe_headers).json()
    for field in NEW_FIELDS + ("has_heatmap",):
        assert report["defect"][field] == body[field], field


@_requires("wood")
def test_wood_upload_always_goes_to_manual_review(client, qe_headers, category_products, tmp_path):
    image = _copy_train_good("wood", tmp_path)
    product = category_products.create(category="wood")

    body = category_products.upload(product, image.read_bytes(), filename="wood.png").json()

    assert body["ai_prediction"] in {"good", "defective"}
    assert body["model_gate"] == "NOT_PRODUCTION_READY"
    assert body["review_required"] is True
    assert "category model not production ready" in body["review_reason"]
    assert body["quality_decision"] == "MANUAL_REVIEW"
    assert body["quality_assessment"] == (
        "AI result is low-reliability for this category/confidence; manual inspection required."
    )
    assert body["has_heatmap"] is True


@_requires("tile")
def test_deleting_the_inspection_removes_its_heatmap(client, qe_headers, category_products, tmp_path):
    product = category_products.create(category="tile")
    body = category_products.upload(product, _copy_train_good("tile", tmp_path).read_bytes()).json()
    path = _heatmap_file(body["id"])
    assert path.is_file() and path.parent == HEATMAP_ROOT and path.name == f"{body['id']}.png"

    assert client.delete(f"/inspections/{body['id']}", headers=qe_headers).status_code == 204

    assert not path.exists()


# ---------------------------------------------------------------------------
# Heatmap endpoint: auth, 404, containment
# ---------------------------------------------------------------------------

def test_heatmap_requires_authentication(client):
    assert client.get("/inspections/1/heatmap").status_code == 401


def test_heatmap_404_for_unknown_inspection_and_for_one_without_heatmap(client, qe_headers, supervisor_headers, category_products):
    assert client.get("/inspections/999999999/heatmap", headers=qe_headers).status_code == 404
    product = category_products.create(category=None)  # no category -> no AI result -> no heatmap
    body = category_products.upload(product, make_image_bytes("PNG")).json()
    assert body["has_heatmap"] is False
    for field in NEW_FIELDS:
        assert body[field] is None, field
    assert client.get(f"/inspections/{body['id']}/heatmap", headers=supervisor_headers).status_code == 404


@pytest.mark.parametrize("stored", ["../uploads/x.png", "../../.env", "..\\..\\.env", "C:/Windows/win.ini", "/etc/passwd"])
def test_heatmap_path_outside_the_heatmap_root_is_never_served(client, qe_headers, category_products, stored):
    env_existed = (BACKEND_DIR / ".env").is_file()
    product = category_products.create(category=None)
    body = category_products.upload(product, make_image_bytes("PNG")).json()
    with SessionLocal() as session:
        session.get(Inspection, body["id"]).heatmap_path = stored
        session.commit()

    response = client.get(f"/inspections/{body['id']}/heatmap", headers=qe_headers)

    assert response.status_code == 404
    assert "env" not in response.text.lower() and "windows" not in response.text.lower()
    # Deleting it must not touch the file outside storage/heatmaps either (containment raises, delete still 204).
    assert client.delete(f"/inspections/{body['id']}", headers=qe_headers).status_code == 204
    assert (BACKEND_DIR / ".env").is_file() == env_existed


# ---------------------------------------------------------------------------
# Best-effort failures: the upload is still 201 and the affected new fields stay NULL
# ---------------------------------------------------------------------------

def _fake_with_grid(prediction="defective", score=3.0, threshold=1.0):
    grid = torch.zeros(32, 32)
    grid[10:16, 12:20] = score
    return PredictionResult(category="tile", prediction=prediction, reconstruction_error=score, threshold=threshold,
                            model_name=SERVING_CONFIGS["tile"].model_name, input_size=(256, 256),
                            processing_time_ms=1.0, patch_scores=grid, input_mode="resize")


def test_localization_failure_keeps_the_upload_and_ai_result(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake_with_grid())

    def _boom(*args, **kwargs):
        raise RuntimeError(r"render failed at C:\secret\storage\heatmaps")

    monkeypatch.setattr("app.inspections.service.localization.render_heatmap", _boom)
    product = category_products.create(category="tile")

    response = category_products.upload(product, make_image_bytes("PNG", size=(64, 48)))

    assert response.status_code == 201
    body = response.json()
    assert body["ai_prediction"] == "defective"
    assert body["localization"] is None and body["has_heatmap"] is False
    assert body["ai_confidence"] is not None  # confidence/review fail independently of localization
    assert "secret" not in response.text and "render failed" not in response.text
    assert client.get(f"/inspections/{body['id']}/heatmap", headers=qe_headers).status_code == 404


def test_confidence_failure_keeps_the_upload_and_leaves_confidence_null(category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake_with_grid())

    def _boom(*args, **kwargs):
        raise RuntimeError("confidence failed")

    monkeypatch.setattr("app.inspections.service.localization.margin_confidence", _boom)
    product = category_products.create(category="tile")

    response = category_products.upload(product, make_image_bytes("PNG", size=(64, 48)))

    assert response.status_code == 201
    body = response.json()
    assert body["ai_confidence"] is None and body["review_required"] is None and body["review_reason"] is None
    assert body["ai_prediction"] == "defective"
    assert body["quality_decision"] == "FAIL"  # review not computed -> treated as not required
    assert body["localization"]["boxes"]  # localization still ran


def test_inference_failure_leaves_every_new_field_null(category_products, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("app.inspections.service.predict_image", _boom)
    product = category_products.create(category="tile")

    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    body = response.json()
    for field in NEW_FIELDS:
        assert body[field] is None, field
    assert body["has_heatmap"] is False
    assert body["quality_decision"] == "NOT_ASSESSED"


def test_fake_defective_grid_upload_is_localized_in_original_coordinates(client, qe_headers, category_products, monkeypatch, tmp_path):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake_with_grid())
    product = category_products.create(category="tile")

    body = category_products.upload(product, make_image_bytes("PNG", size=(64, 48))).json()

    assert body["quality_decision"] == "FAIL"
    assert body["review_required"] is False and body["ai_reliability"] == "high"
    (box,) = body["localization"]["boxes"]
    # Hot patches: columns 12..19, rows 10..15 of a 32x32 grid; the blurred map thresholded at 1.0 is centred on them.
    assert box["x"] < 12 / 32 < 20 / 32 < box["x"] + box["w"]
    assert box["y"] < 10 / 32 < 16 / 32 < box["y"] + box["h"]
    assert body["localization"]["centroid"] == pytest.approx([16 / 32, 13 / 32], abs=0.02)
    png = tmp_path / "h.png"
    png.write_bytes(client.get(f"/inspections/{body['id']}/heatmap", headers=qe_headers).content)
    with Image.open(png) as rendered:
        assert rendered.mode == "RGBA" and rendered.size == (64, 48)


def test_result_without_patch_grid_gets_confidence_but_no_localization(category_products, monkeypatch):
    fake = PredictionResult(category="tile", prediction="good", reconstruction_error=1.0, threshold=2.0,
                            model_name="fake-model", input_size=(256, 256), processing_time_ms=1.0)
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake)
    product = category_products.create(category="tile")

    body = category_products.upload(product, make_image_bytes("PNG")).json()

    assert body["ai_confidence"] == pytest.approx(1 / (1 + 2 ** -10))  # sigmoid(10 * ln 2)
    assert body["localization"] is None and body["has_heatmap"] is False
    assert body["model_gate"] is None  # "fake-model" is not tile's registered model
    assert body["quality_decision"] == "PASS"


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def test_manual_review_flows_through_the_quality_distribution_and_summary(client, qe_headers, category_products, monkeypatch):
    def _summary():
        response = client.get("/inspections/analytics/summary", headers=qe_headers)
        assert response.status_code == 200
        return response.json()

    def _row(body):
        return next((r["count"] for r in body["quality_decisions"] if r["decision"] == "MANUAL_REVIEW"), 0)

    before = _summary()
    wood = SERVING_CONFIGS["wood"]
    fake = PredictionResult(category="wood", prediction="good", reconstruction_error=1.0, threshold=wood.threshold,
                            model_name=wood.model_name, input_size=wood.input_size, processing_time_ms=1.0)
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake)
    body = category_products.upload(category_products.create(category="wood"), make_image_bytes("PNG")).json()
    assert body["quality_decision"] == "MANUAL_REVIEW"

    after = _summary()
    assert after["manual_review_count"] == before["manual_review_count"] + 1 == _row(after)
    assert sum(r["count"] for r in after["quality_decisions"]) == after["total_inspections"]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_localization_columns_migration():
    script = ScriptDirectory.from_config(Config(str(BACKEND_DIR / "alembic.ini")))
    head = script.get_current_head()
    assert script.get_revision(head).down_revision == "39bd1b8da129"
    expected = {
        "ai_confidence": ("double precision", None),
        "review_required": ("boolean", None),
        "review_reason": ("character varying", 255),
        "localization": ("jsonb", None),
        "heatmap_path": ("character varying", 500),
    }
    with SessionLocal() as session:
        current = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        rows = session.execute(text(
            "SELECT column_name, data_type, is_nullable, character_maximum_length, column_default "
            "FROM information_schema.columns WHERE table_name = 'inspections' AND column_name = ANY(:names)"
        ), {"names": list(NEW_COLUMNS)}).all()
        # No backfill: a row without an AI result never has any of the new values.
        filled_without_ai = session.execute(text(
            "SELECT count(*) FROM inspections WHERE ai_prediction IS NULL AND (ai_confidence IS NOT NULL OR "
            "review_required IS NOT NULL OR review_reason IS NOT NULL OR localization IS NOT NULL OR heatmap_path IS NOT NULL)"
        )).scalar_one()

    assert current == head
    assert {r.column_name: (r.data_type, r.character_maximum_length) for r in rows} == expected
    assert all(r.is_nullable == "YES" and r.column_default is None for r in rows)
    assert filled_without_ai == 0
