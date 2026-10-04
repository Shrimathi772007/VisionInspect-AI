"""Tile serving: production inference uses the locked model-family winner (knn_l23_256) and its locked threshold.

Covers the registration of the Tile alternative-model-family model in app.ai.inference.serving: Tile resolves
to the frozen-ResNet-18 patch nearest-neighbour detector from
backend/ai_models/tile/model_family_study/selected_candidate/ (never to the failed Phase 1 ConvAE), with its
hash-verified artifact/configuration, 256x256 preprocessing and the exact locked threshold.

No test here reads the Tile final-test images (dataset/tile/test/): an autouse guard refuses to decode them.
Images are synthetic, or come from Tile train/good (the model's own training pool). Tests that need the real
(Git-ignored) artifact, backbone or dataset skip when they are absent, like the other AI serving tests.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.ai.inference.predict as predict_module
from app.ai.inference import ModelArtifactNotFoundError, ModelIntegrityError, PredictionResult, predict_image
from app.ai.inference.serving import (
    MODEL_FAMILY_CONVAE,
    MODEL_FAMILY_PATCH_ANOMALY,
    MODEL_FAMILY_PATCHCORE_WRN50,
    SERVING_CONFIGS,
    CategoryServingConfig,
    clear_model_cache,
    get_serving_config,
    load_serving_model,
)
from app.ai.models.patch_anomaly import (
    AGG_TOP1PCT,
    KNNPatchScorer,
    PatchAnomalyDetector,
    extract_patch_features,
)
from app.ai.models.resnet18 import load_pretrained_resnet18, pretrained_weights_path
from app.ai.preprocessing import pipeline
from app.ai.preprocessing.errors import ImageDecodeError, ImageNotFoundError
from app.ai.training import build_model, save_model
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.storage import DATASET_ROOT
from tests.conftest import make_image_bytes

TILE_THRESHOLD = 1.7393077017650718
TILE_MODEL_SHA256 = "de00e2774daf02d97e1414fe75b10305e2ad392696449b1627a250b1ec0dd4d7"
TILE_MODEL_MD5 = "cb9530d76709319ec0eed58f55717356"
TILE_CONFIG_SHA256 = "6ab1c5e603cf2b63cb9c9f0e17320d83c2a5242c6df7c3cf386dedb63ed3812d"
TILE_EXPERIMENT_ID = "tile-model-family-tile-0ed823168064"
CONVAE_PHASE1_THRESHOLD = 0.014651721413975766

STUDY_DIR = ARTIFACTS_ROOT / "tile" / "model_family_study"
SELECTED_DIR = STUDY_DIR / "selected_candidate"
TILE_MODEL_PATH = SELECTED_DIR / "model_state.pt"
LOCK_PATH = STUDY_DIR / "selection_lock.json"
PHASE1_DIR = ARTIFACTS_ROOT / "tile" / "phase1_baseline"
BACKBONE_PATH = pretrained_weights_path()
TRAIN_GOOD_IMAGE = DATASET_ROOT / "tile" / "train" / "good" / "000.png"
TILE_TEST_ROOT = (DATASET_ROOT / "tile" / "test").resolve()

requires_tile_model = pytest.mark.skipif(
    not (TILE_MODEL_PATH.is_file() and BACKBONE_PATH.is_file()),
    reason="Tile model-family artifact or ResNet-18 backbone not present in this environment",
)
requires_backbone = pytest.mark.skipif(not BACKBONE_PATH.is_file(), reason="ResNet-18 backbone not present")
requires_train_image = pytest.mark.skipif(not TRAIN_GOOD_IMAGE.is_file(), reason="MVTec tile train/good not present")


@pytest.fixture(autouse=True)
def _fresh_cache_and_final_test_guard(monkeypatch):
    real_load = pipeline._load_image

    def guarded(path):
        if TILE_TEST_ROOT in Path(path).resolve().parents:
            raise AssertionError(f"Tile final-test image must not be read: {path}")
        return real_load(path)

    monkeypatch.setattr(pipeline, "_load_image", guarded)
    clear_model_cache()
    yield
    clear_model_cache()


def _digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def _png(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Registration: tile -> knn_l23_256, never ConvAE
# ---------------------------------------------------------------------------

def test_tile_resolves_to_the_knn_l23_256_patch_model():
    config = get_serving_config("tile")
    assert config.model_name == "knn_l23_256"
    assert config.model_family == MODEL_FAMILY_PATCH_ANOMALY
    assert config.artifact_name == "model_family_study/selected_candidate/model_state"
    assert config.experiment_id == TILE_EXPERIMENT_ID
    assert "selection_lock.json" in config.provenance


def test_tile_does_not_resolve_to_the_phase1_convae():
    config = get_serving_config("tile")
    assert config.model_family != MODEL_FAMILY_CONVAE
    assert config.model_name != "autoencoder"
    assert "phase1_baseline" not in config.artifact_name
    assert config.threshold != CONVAE_PHASE1_THRESHOLD
    assert get_model_path("tile", config.artifact_name).parent != PHASE1_DIR


def test_tile_uses_the_selected_candidate_artifact_path():
    config = get_serving_config("tile")
    assert get_model_path("tile", config.artifact_name) == TILE_MODEL_PATH
    assert config.expected_md5 == TILE_MODEL_MD5
    assert config.expected_sha256 == TILE_MODEL_SHA256
    assert config.expected_config_sha256 == TILE_CONFIG_SHA256


def test_tile_uses_exactly_the_locked_threshold():
    config = get_serving_config("tile")
    assert config.threshold == TILE_THRESHOLD
    assert repr(config.threshold) == "1.7393077017650718"
    assert (config.threshold_method, config.threshold_parameter) == ("mean_std", 3.0)
    assert config.input_size == (256, 256)


@pytest.mark.skipif(not LOCK_PATH.is_file(), reason="Tile selection lock not present in this environment")
def test_tile_configuration_matches_the_selection_lock():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = get_serving_config("tile")
    assert lock["selected_candidate"] == config.model_name
    assert lock["selected_policy"] == "mean_std_3"
    assert lock["selected_threshold"] == config.threshold
    assert lock["selected_artifact"]["sha256"] == config.expected_sha256
    assert lock["selected_artifact"]["md5"] == config.expected_md5
    assert lock["experiment_id"] == config.experiment_id
    assert _digest(SELECTED_DIR / "model_config.json", "sha256") == config.expected_config_sha256


def test_other_categories_keep_their_existing_configuration():
    # Updated deliberately (all-categories registration): Bottle's Phase 3 ConvAE was replaced by its WRN-50
    # PatchCore model, and every MVTec category is registered (test_ai_all_categories_serving).
    bottle = get_serving_config("bottle")
    assert bottle.model_family == MODEL_FAMILY_PATCHCORE_WRN50 and bottle.model_name == "wrn50_patchcore_crop224"
    assert bottle.artifact_name == "patchcore_wrn50/final_model/model_state" and bottle.input_size == (224, 224)
    assert bottle.threshold == 1.6858729828595898
    assert bottle.expected_sha256 == "0c7bd7f45c4769980b7c4e6cb5d6d8762cef35a9088eed045d03bebda8ecb7cb"
    assert set(SERVING_CONFIGS) == set(MVTEC_CATEGORIES)


# ---------------------------------------------------------------------------
# Real locked Tile model: loading, integrity, configuration, scoring
# ---------------------------------------------------------------------------

@requires_tile_model
def test_real_tile_model_loads_with_the_locked_configuration():
    detector = load_serving_model(get_serving_config("tile"))
    assert isinstance(detector, PatchAnomalyDetector)
    assert detector.layers == (2, 3)
    assert detector.image_size == 256
    assert detector.aggregation == AGG_TOP1PCT
    assert isinstance(detector.scorer, KNNPatchScorer)
    assert detector.scorer.bank.shape[1] == 384
    assert not detector.extractor.training
    assert not any(p.requires_grad for p in detector.extractor.parameters())
    assert json.loads((SELECTED_DIR / "model_config.json").read_text(encoding="utf-8")) == {
        "detector": "patch_anomaly", "backbone": "resnet18_imagenet_frozen", "layers": [2, 3], "image_size": 256,
        "scorer": "knn", "aggregation": "top1pct_mean", "feature_dim": 384}


@requires_tile_model
def test_real_tile_model_is_cached():
    config = get_serving_config("tile")
    assert load_serving_model(config) is load_serving_model(config)


@requires_tile_model
def test_tile_preprocessing_is_256_and_score_is_knn_l23_top1pct(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    image = _png(tmp_path / "synthetic.png", rng.integers(0, 256, (300, 280, 3)))
    sizes = []
    real_process_image = predict_module.process_image

    def spy(path, *args, **kwargs):
        sizes.append(kwargs.get("target_size", args[0] if args else None))
        return real_process_image(path, *args, **kwargs)

    monkeypatch.setattr(predict_module, "process_image", spy)
    result = predict_image(image, "tile")
    assert sizes == [(256, 256)]
    assert result.input_size == (256, 256)

    # Independent recomputation: project preprocessing at 256 -> ResNet-18 l2+l3 (3x3 avg pool) -> k=1 Euclidean
    # distance to the bank -> mean of the top 1% of patch scores.
    detector = load_serving_model(get_serving_config("tile"))
    processed = real_process_image(image, target_size=(256, 256)).preprocessing.normalized_image
    tensor = torch.from_numpy(processed).permute(2, 0, 1)[None]
    patches = extract_patch_features(detector.extractor, tensor, (2, 3))
    assert patches.shape == (1, 32, 32, 384)
    flat = patches.reshape(-1, 384)
    nearest = torch.cat([torch.cdist(flat[i : i + 1024], detector.scorer.bank).min(dim=1).values
                         for i in range(0, flat.shape[0], 1024)])
    k = math.ceil(0.01 * nearest.numel())
    expected = float(nearest.topk(k).values.mean())
    assert result.reconstruction_error == pytest.approx(expected, rel=1e-6)
    assert result.prediction == ("good" if result.reconstruction_error <= TILE_THRESHOLD else "defective")


@requires_tile_model
def test_tile_refuses_any_other_input_size_before_loading(tmp_path, monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("no model may be loaded")

    monkeypatch.setattr(predict_module, "load_serving_model", _must_not_load)
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    for size in [(128, 128), (224, 224), (32, 32)]:
        with pytest.raises(ValueError, match="validated input size"):
            predict_image(image, "tile", image_size=size)


@requires_tile_model
def test_tile_default_call_returns_the_registered_model_and_threshold(tmp_path):
    image = tmp_path / "flat.png"
    image.write_bytes(make_image_bytes("PNG", size=(64, 64), color=(120, 110, 100)))
    result = predict_image(image, "tile")  # exactly how app.inspections.service calls it
    assert isinstance(result, PredictionResult)
    assert result.category == "tile" and result.model_name == "knn_l23_256"
    assert result.threshold == TILE_THRESHOLD
    assert result.prediction in ("good", "defective")
    assert predict_image(image, "tile", model_name="knn_l23_256").reconstruction_error == result.reconstruction_error
    with pytest.raises(ModelArtifactNotFoundError, match="configured model: 'knn_l23_256'"):
        predict_image(image, "tile", model_name="autoencoder")


@requires_tile_model
@requires_train_image
def test_real_tile_normal_and_painted_defect_give_valid_threshold_decisions(tmp_path):
    """A Tile train/good image (training pool, not the final test) and a synthetic defect painted on it."""
    normal = predict_image(TRAIN_GOOD_IMAGE, "tile")
    array = np.asarray(Image.open(TRAIN_GOOD_IMAGE).convert("RGB")).copy()
    h, w = array.shape[:2]
    array[h // 3 : h // 3 + h // 6, w // 3 : w // 3 + w // 6] = 0
    painted = predict_image(_png(tmp_path / "painted.png", array), "tile")
    for result in (normal, painted):
        assert result.prediction == ("good" if result.reconstruction_error <= TILE_THRESHOLD else "defective")
        assert result.threshold == TILE_THRESHOLD and result.model_name == "knn_l23_256"
    assert painted.reconstruction_error > normal.reconstruction_error
    assert painted.prediction == "defective"


@requires_tile_model
def test_tile_inference_is_deterministic(tmp_path):
    image = _png(tmp_path / "noise.png", np.random.default_rng(3).integers(0, 256, (256, 256, 3)))
    assert predict_image(image, "tile").reconstruction_error == predict_image(image, "tile").reconstruction_error


@requires_tile_model
def test_tile_inference_never_modifies_or_adds_artifacts_and_never_trains(tmp_path, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("inference must not train")

    monkeypatch.setattr(torch.optim, "Adam", _forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", _forbidden)

    def snapshot():
        return {p: (p.stat().st_mtime_ns, _digest(p, "sha256")) for p in sorted(STUDY_DIR.rglob("*")) if p.is_file()}

    before = snapshot()
    image = _png(tmp_path / "noise.png", np.random.default_rng(5).integers(0, 256, (200, 200, 3)))
    predict_image(image, "tile")
    clear_model_cache()
    predict_image(image, "tile")
    assert snapshot() == before
    assert _digest(TILE_MODEL_PATH, "sha256") == TILE_MODEL_SHA256


@requires_tile_model
def test_tile_invalid_or_missing_image_raises_the_existing_preprocessing_errors(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image at all")
    with pytest.raises(ImageDecodeError):
        predict_image(corrupt, "tile")
    with pytest.raises(ImageNotFoundError):
        predict_image(tmp_path / "missing.png", "tile")


# ---------------------------------------------------------------------------
# Missing / tampered artifacts (isolated artifact root)
# ---------------------------------------------------------------------------

def test_missing_tile_artifact_raises_and_never_falls_back_to_the_convae(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "tile" / "phase1_baseline" / "autoencoder.pt")  # only the failed ConvAE exists
    save_model(build_model(), root / "tile" / "autoencoder.pt")
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ModelArtifactNotFoundError, match="tile"):
        predict_image(image, "tile")


def test_missing_tile_model_config_raises_not_found(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    target = root / "tile" / "model_family_study" / "selected_candidate" / "model_state.pt"
    target.parent.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 384)}, target)  # state without its model_config.json
    with pytest.raises(ModelArtifactNotFoundError, match="tile"):
        load_serving_model(get_serving_config("tile"))


def test_tampered_tile_artifact_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    directory = root / "tile" / "model_family_study" / "selected_candidate"
    directory.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 384)}, directory / "model_state.pt")  # right place, wrong content
    (directory / "model_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="expected MD5 " + TILE_MODEL_MD5):
        load_serving_model(get_serving_config("tile"))


# ---------------------------------------------------------------------------
# Patch-anomaly serving path on a small synthetic model (real frozen backbone, synthetic bank)
# ---------------------------------------------------------------------------

@pytest.fixture
def widget(tmp_path, monkeypatch):
    """A patch-anomaly 'widget' model whose memory bank holds the patches of a flat grey 64x64 image."""
    backbone = load_pretrained_resnet18(BACKBONE_PATH)
    monkeypatch.setattr("app.ai.inference.serving.load_pretrained_resnet18", lambda: backbone)
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    grey = _png(tmp_path / "grey.png", np.full((64, 64, 3), 128))
    tensor = torch.from_numpy(pipeline.process_image(grey, target_size=(64, 64)).preprocessing.normalized_image)
    patches = extract_patch_features(backbone, tensor.permute(2, 0, 1)[None], (2, 3))
    detector = PatchAnomalyDetector(backbone, (2, 3), 64, KNNPatchScorer(patches.reshape(-1, 384).clone()), AGG_TOP1PCT)
    state = detector.save(root / "widget" / "patch")
    config = CategoryServingConfig(
        category="widget", artifact_name="patch/model_state", model_name="widget_knn", threshold=0.5,
        threshold_method="mean_std", threshold_parameter=3.0, expected_md5=_digest(state, "md5"), provenance="test fixture",
        model_family=MODEL_FAMILY_PATCH_ANOMALY, input_size=(64, 64), expected_sha256=_digest(state, "sha256"),
        expected_config_sha256=_digest(state.parent / "model_config.json", "sha256"), experiment_id="widget-test")
    monkeypatch.setitem(SERVING_CONFIGS, "widget", config)
    return {"config": config, "grey": grey, "state": state, "tmp": tmp_path}


@requires_backbone
def test_patch_model_scores_normal_as_good_and_anomaly_as_defective(widget):
    good = predict_image(widget["grey"], "widget")
    noise = _png(widget["tmp"] / "noise.png", np.random.default_rng(1).integers(0, 256, (64, 64, 3)))
    bad = predict_image(noise, "widget")
    assert good.prediction == "good" and good.reconstruction_error <= 0.5
    assert bad.prediction == "defective" and bad.reconstruction_error > 0.5
    assert good.model_name == bad.model_name == "widget_knn"
    assert good.threshold == bad.threshold == 0.5 and good.input_size == (64, 64)


@requires_backbone
def test_patch_model_sha256_mismatch_fails_loading(widget, monkeypatch):
    config = CategoryServingConfig(**{**widget["config"].__dict__, "expected_sha256": "0" * 64})
    monkeypatch.setitem(SERVING_CONFIGS, "widget", config)
    with pytest.raises(ModelIntegrityError, match="expected SHA256 " + "0" * 64):
        predict_image(widget["grey"], "widget")


@requires_backbone
def test_patch_model_configuration_change_fails_loading(widget):
    config_file = widget["state"].parent / "model_config.json"
    config_file.write_text(config_file.read_text().replace('"image_size": 64', '"image_size": 128'), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="Model configuration"):
        load_serving_model(widget["config"])


@requires_backbone
def test_patch_model_state_change_fails_loading(widget):
    state = widget["state"]
    state.write_bytes(state.read_bytes() + b"\0")
    with pytest.raises(ModelIntegrityError, match="expected MD5"):
        load_serving_model(widget["config"])
