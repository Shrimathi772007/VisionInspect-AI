"""Uploads with defect form + severity_v1, POST /inspections/batch, the enhancement preview endpoints, the
automation rate, and score identity of the upload path. Only synthetic images and copies of
dataset/<category>/train/good/ images are used (never the test split)."""

import io
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.ai.analytics import enhancement
from app.ai.inference import PredictionResult, predict_image
from app.ai.inference.serving import SERVING_CONFIGS
from app.ai.models.resnet18 import pretrained_weights_path
from app.ai.training.artifacts import get_model_path
from app.database import SessionLocal
from app.inspections import router as inspections_router
from app.inspections.storage import DATASET_ROOT
from app.models.inspection import Inspection
from tests.conftest import make_image_bytes

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _fake(category, prediction="defective", score=3.0, threshold=1.0, hot=(10, 16, 12, 20)):
    """A PredictionResult with a 32x32 patch grid that has one hot block (rows r0:r1, cols c0:c1)."""
    grid = torch.zeros(32, 32)
    if prediction == "defective":
        r0, r1, c0, c1 = hot
        grid[r0:r1, c0:c1] = score
    config = SERVING_CONFIGS[category]
    return PredictionResult(category=category, prediction=prediction, reconstruction_error=score, threshold=threshold,
                            model_name=config.model_name, input_size=(256, 256), processing_time_ms=1.0,
                            patch_scores=grid, input_mode="resize")


def _png(size=(96, 64), seed=0):
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Uploads: defect form + severity + precedence, end to end through the API
# ---------------------------------------------------------------------------

def test_defective_tile_upload_gets_defect_form_severity_and_fail(category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile"))
    body = category_products.upload(category_products.create(category="tile"), _png()).json()

    loc = body["localization"]
    assert loc["defect_form"] in {"small_spot", "localized_patch", "linear", "large_area", "multiple_regions"}
    assert loc["defect_form_version"] == "defect_form_v1"
    assert loc["image_size"] == [96, 64]
    assert body["defect_form"] == loc["defect_form"] and body["defect_form_label"]
    assert body["defect_category"] is None  # never written into the ground-truth column
    assert body["severity_score"] is not None and body["severity_level"] in {"Critical", "High", "Medium", "Low"}
    assert body["severity_action"]
    assert body["quality_decision"] == "FAIL"
    assert f"Severity: {body['severity_level']}" in body["quality_assessment"]
    assert body["review_required"] is False


def test_weak_category_defective_upload_goes_to_manual_review_with_severity_stored(category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("wood", score=4.0))
    body = category_products.upload(category_products.create(category="wood"), _png()).json()

    assert body["model_gate"] == "NOT_PRODUCTION_READY"
    assert body["review_required"] is True
    assert body["severity_score"] is not None and body["severity_level"] is not None
    assert body["quality_decision"] == "MANUAL_REVIEW"
    assert f"Severity: {body['severity_level']}" in body["quality_assessment"]


def test_good_upload_gets_no_severity_and_no_defect_form(category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile", "good", 0.5))
    body = category_products.upload(category_products.create(category="tile"), _png()).json()

    assert body["quality_decision"] == "PASS"
    assert body["severity_score"] is None and body["severity_level"] is None
    assert body["defect_form"] is None and "defect_form" not in body["localization"]


def test_report_carries_defect_form_and_recommended_action(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile"))
    body = category_products.upload(category_products.create(category="tile"), _png()).json()
    report = client.get(f"/inspections/{body['id']}/report", headers=qe_headers).json()
    assert report["defect"]["defect_form"] == body["defect_form"]
    assert report["severity"]["recommended_action"] == body["severity_action"]


def test_defect_form_failure_keeps_the_upload_and_leaves_severity_null(category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile"))

    def _boom(*args, **kwargs):
        raise RuntimeError(r"form failed at C:\secret")

    monkeypatch.setattr("app.inspections.service.add_defect_form", _boom)
    response = category_products.upload(category_products.create(category="tile"), _png())

    assert response.status_code == 201
    body = response.json()
    assert body["localization"]["boxes"] and body["defect_form"] is None
    assert body["severity_score"] is None
    assert body["quality_decision"] == "FAIL"
    assert "secret" not in response.text


# ---------------------------------------------------------------------------
# Score identity: the upload path still produces exactly predict_image's score
# ---------------------------------------------------------------------------

TILE_TRAIN_GOOD = Path(DATASET_ROOT) / "tile" / "train" / "good" / "000.png"
_tile_config = SERVING_CONFIGS["tile"]
_tile_model = get_model_path("tile", _tile_config.artifact_name)


@pytest.mark.skipif(not (TILE_TRAIN_GOOD.is_file() and _tile_model.is_file() and pretrained_weights_path().is_file()),
                    reason="tile model, ResNet-18 backbone or tile train/good image not present")
def test_upload_path_score_and_prediction_are_identical_to_predict_image(client, qe_headers, category_products, tmp_path):
    copy = tmp_path / "tile_train_good_000.png"
    shutil.copyfile(TILE_TRAIN_GOOD, copy)
    expected = predict_image(copy, "tile")

    body = category_products.upload(category_products.create(category="tile"), copy.read_bytes(), filename="t.png").json()

    assert body["ai_reconstruction_error"] == expected.reconstruction_error
    assert body["ai_prediction"] == expected.prediction
    assert body["ai_threshold"] == expected.threshold
    # The preview endpoints exist and work on the same inspection without touching its stored result.
    assert client.get(f"/inspections/{body['id']}/enhanced", headers=qe_headers).status_code == 200
    assert client.get(f"/inspections/{body['id']}/image-quality", headers=qe_headers).status_code == 200
    again = client.get(f"/inspections/{body['id']}", headers=qe_headers).json()
    assert again["ai_reconstruction_error"] == expected.reconstruction_error


# ---------------------------------------------------------------------------
# POST /inspections/batch
# ---------------------------------------------------------------------------

def _batch(client, headers, product_id, files):
    return client.post("/inspections/batch", headers=headers, data={"product_id": str(product_id)},
                       files=[("files", f) for f in files])


def test_batch_of_three_good_files(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile", "good", 0.5))
    product = category_products.create(category="tile")

    response = _batch(client, qe_headers, product["id"], [(f"img{i}.png", _png(seed=i), "image/png") for i in range(3)])

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["succeeded"], body["failed"]) == (3, 3, 0)
    assert [item["filename"] for item in body["items"]] == ["img0.png", "img1.png", "img2.png"]
    for item in body["items"]:
        assert item["error"] is None
        inspection = item["inspection"]
        assert inspection["product_id"] == product["id"] and inspection["source"] == "upload"
        assert inspection["quality_decision"] == "PASS" and inspection["has_heatmap"] is True
    assert len({item["inspection"]["id"] for item in body["items"]}) == 3


def test_one_invalid_file_does_not_abort_the_batch(client, qe_headers, category_products, monkeypatch):
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile"))
    product = category_products.create(category="tile")

    response = _batch(client, qe_headers, product["id"], [
        ("a.png", _png(seed=1), "image/png"),
        ("../../evil.txt", b"not an image", "text/plain"),
        ("broken.png", b"\x89PNG\r\n\x1a\nnot really", "image/png"),
        ("b.png", _png(seed=2), "image/png"),
    ])

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["succeeded"], body["failed"]) == (4, 2, 2)
    bad = [item for item in body["items"] if item["error"]]
    assert [item["filename"] for item in bad] == ["evil.txt", "broken.png"]
    assert all(item["inspection"] is None for item in bad)
    assert "Unsupported file type" in bad[0]["error"] and "not a valid" in bad[1]["error"]
    assert "storage" not in response.text and "\\" not in response.text
    good = [item for item in body["items"] if item["inspection"]]
    assert all(item["inspection"]["quality_decision"] == "FAIL" and item["inspection"]["severity_level"] for item in good)


def test_batch_without_category_succeeds_with_null_ai(client, qe_headers, category_products):
    product = category_products.create(category=None)
    body = _batch(client, qe_headers, product["id"], [("a.png", make_image_bytes("PNG"), "image/png"),
                                                       ("b.png", make_image_bytes("PNG"), "image/png")]).json()
    assert body["succeeded"] == 2
    for item in body["items"]:
        assert item["inspection"]["ai_prediction"] is None
        assert item["inspection"]["quality_decision"] == "NOT_ASSESSED"


def test_batch_limits(client, qe_headers, category_products, monkeypatch):
    assert inspections_router.MAX_BATCH_FILES == 20
    assert inspections_router.MAX_BATCH_BYTES == 50 * 1024 * 1024
    product = category_products.create(category=None)

    too_many = _batch(client, qe_headers, product["id"], [(f"{i}.png", b"x", "image/png") for i in range(21)])
    assert too_many.status_code == 400 and "at most 20" in too_many.json()["detail"]

    monkeypatch.setattr(inspections_router, "MAX_BATCH_BYTES", 1000)
    too_big = _batch(client, qe_headers, product["id"], [("a.png", b"x" * 600, "image/png"), ("b.png", b"x" * 600, "image/png")])
    assert too_big.status_code == 413 and "maximum total size" in too_big.json()["detail"]
    with SessionLocal() as session:
        assert session.query(Inspection).filter(Inspection.product_id == product["id"]).count() == 0


def test_batch_auth_and_unknown_product(client, supervisor_headers, qe_headers, category_products):
    product = category_products.create(category=None)
    files = [("a.png", make_image_bytes("PNG"), "image/png")]
    assert _batch(client, {}, product["id"], files).status_code == 401
    assert _batch(client, supervisor_headers, product["id"], files).status_code == 403
    assert _batch(client, qe_headers, 999999999, files).status_code == 404


# ---------------------------------------------------------------------------
# Enhancement preview: pure function
# ---------------------------------------------------------------------------

def test_enhanced_preview_size_and_determinism():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (500, 1000, 3), dtype=np.uint8)
    first = enhancement.enhance(image)
    assert first.shape == (320, 640, 3)
    assert np.array_equal(first, enhancement.enhance(image))
    small = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    assert enhancement.enhance(small).shape == (100, 120, 3)  # never upscaled


def test_contrast_rises_on_a_low_contrast_image():
    gradient = np.tile(np.linspace(110, 140, 400, dtype=np.float32), (300, 1))
    image = cv2.cvtColor(gradient.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    report = enhancement.quality_report(image)
    assert report["after"]["contrast"] >= report["before"]["contrast"]
    assert report["after"]["contrast"] > report["before"]["contrast"]  # CLAHE stretches it


def test_noise_estimate_drops_on_a_noisy_image():
    rng = np.random.default_rng(5)
    noisy = np.clip(128 + rng.normal(0, 15, (300, 400, 3)), 0, 255).astype(np.uint8)
    before = enhancement.noise_estimate(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY))
    assert before == pytest.approx(15 / (3 ** 0.5) * 1.0, rel=0.35)  # gray = mean of 3 channels: sigma / sqrt(3)
    report = enhancement.quality_report(noisy)
    assert report["after"]["noise"] < report["before"]["noise"]


def test_quality_report_shape():
    report = enhancement.quality_report(np.full((50, 80, 3), 100, dtype=np.uint8))
    assert report["preview_only"] is True
    assert report["note"] == "Preview only. The AI models do not use the enhanced image."
    assert report["resolution"] == {"width": 80, "height": 50}
    assert set(report["before"]) == set(report["after"]) == {"sharpness", "contrast", "noise", "brightness"}


# ---------------------------------------------------------------------------
# Enhancement preview: endpoints
# ---------------------------------------------------------------------------

def test_enhanced_and_image_quality_endpoints(client, qe_headers, supervisor_headers, category_products):
    body = category_products.upload(category_products.create(category=None), _png(size=(900, 300))).json()

    enhanced = client.get(f"/inspections/{body['id']}/enhanced", headers=supervisor_headers)
    assert enhanced.status_code == 200 and enhanced.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(enhanced.content)) as preview:
        assert preview.size == (640, 213)

    quality = client.get(f"/inspections/{body['id']}/image-quality", headers=supervisor_headers)
    assert quality.status_code == 200
    data = quality.json()
    assert data["resolution"] == {"width": 900, "height": 300}
    assert data["analysed_resolution"] == {"width": 640, "height": 213}
    assert data["preview_only"] is True


def test_enhancement_endpoints_401_404_and_containment(client, qe_headers, category_products):
    body = category_products.upload(category_products.create(category=None), _png()).json()
    for suffix in ("enhanced", "image-quality"):
        assert client.get(f"/inspections/{body['id']}/{suffix}").status_code == 401
        assert client.get(f"/inspections/999999999/{suffix}", headers=qe_headers).status_code == 404

    with SessionLocal() as session:
        inspection = session.get(Inspection, body["id"])
        original = inspection.image_path
        inspection.image_path = "../../.env"
        session.commit()
    try:
        for suffix in ("enhanced", "image-quality"):
            response = client.get(f"/inspections/{body['id']}/{suffix}", headers=qe_headers)
            assert response.status_code == 404
            assert response.json() == {"detail": "Image not found"}
    finally:
        with SessionLocal() as session:
            session.get(Inspection, body["id"]).image_path = original
            session.commit()


# ---------------------------------------------------------------------------
# Automation rate
# ---------------------------------------------------------------------------

def test_automation_rate_and_counts(client, qe_headers, category_products, monkeypatch):
    def summary():
        response = client.get("/inspections/analytics/summary", headers=qe_headers)
        assert response.status_code == 200
        return response.json()

    before = summary()
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: _fake("tile", "good", 0.5))
    category_products.upload(category_products.create(category="tile"), _png())
    after = summary()

    counts = after["automation_counts"]
    assert set(counts) == {"automatic", "manual_review", "not_assessed"}
    assert counts["automatic"] == before["automation_counts"]["automatic"] + 1
    assert counts["manual_review"] == after["manual_review_count"]
    decided = counts["automatic"] + counts["manual_review"] + counts["not_assessed"]
    with SessionLocal() as session:
        assert decided == session.query(Inspection).filter(Inspection.quality_decision.is_not(None)).count()
    assert after["automation_rate"] == pytest.approx(counts["automatic"] / decided)
    assert 0 <= after["automation_rate"] <= 1
