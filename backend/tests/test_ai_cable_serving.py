"""Cable serving: production inference uses the locked model-family winner (knn_l23_128) and its locked threshold.

Covers the registration of the Cable alternative-model-family model in app.ai.inference.serving: Cable resolves
to the frozen-ResNet-18 patch nearest-neighbour detector from
backend/ai_models/cable/model_family_study/selected_candidate/ (never to the failed Phase 1 ConvAE, never to another
study candidate), with its hash-verified artifact/configuration, 128x128 preprocessing and the exact locked threshold.

No test here reads the Cable final-test images (dataset/cable/test/ - consumed by the one-time final test): an
autouse guard refuses to open or decode them. Images are synthetic, or come from Cable train/good (the model's own
training pool). Tests that need the real (Git-ignored) artifact, backbone or dataset skip when they are absent,
like the other AI serving tests.
"""

import builtins
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.ai.inference.predict as predict_module
import app.ai.inference.serving as serving_module
from app.ai.inference import ModelArtifactNotFoundError, ModelIntegrityError, PredictionResult, predict_image
from app.ai.inference.serving import (
    MODEL_FAMILY_CONVAE,
    MODEL_FAMILY_PATCH_ANOMALY,
    MODEL_FAMILY_PATCHCORE_WRN50,
    SERVING_CONFIGS,
    clear_model_cache,
    get_serving_config,
    load_serving_model,
)
from app.ai.models.patch_anomaly import AGG_TOP1PCT, KNNPatchScorer, PatchAnomalyDetector, extract_patch_features
from app.ai.models.resnet18 import pretrained_weights_path
from app.ai.models.wide_resnet50 import wide_resnet50_weights_path
from app.ai.preprocessing import pipeline
from app.ai.training import build_model, save_model
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.storage import DATASET_ROOT
from tests.conftest import make_image_bytes

CABLE_THRESHOLD = 2.244786389430004
CABLE_MODEL_SHA256 = "432859c5e39aaa85f8adbbd952f82ff0be1e6a905e360cfcca978a5638de7f5a"
CABLE_MODEL_MD5 = "c878207964fc22865743f77c43ed06ea"
CABLE_CONFIG_SHA256 = "f6696e7fbb3af5f275d8678c66339210ec7230195c212cad70dd22777bcf3230"
CABLE_MANIFEST_SHA256 = "6bdd6543d1fd3a247673ab701fa85725a46e62685e52a54724dd375ab59093a3"
CABLE_LOCK_SHA256 = "72c2bf182b71e8fefb9e5907cec9380989f0322caaf2bcb711be64230bef534e"
CABLE_LOCK_DIGEST = "31d1aed63843aa0e316c3343b5a82309b2b349fde803d93219e22d012bcbc675"
CABLE_EXPERIMENT_ID = "cable-model-family-cable-ce2a8f015143"
CONVAE_PHASE1_THRESHOLD = 0.010599855769247735
TILE_THRESHOLD = 1.7393077017650718
BOTTLE_THRESHOLD = 1.6858729828595898  # WRN-50 PatchCore (replaced the Phase 3 ConvAE's 0.0028031117030001018)

STUDY_DIR = ARTIFACTS_ROOT / "cable" / "model_family_study"
SELECTED_DIR = STUDY_DIR / "selected_candidate"
CABLE_MODEL_PATH = SELECTED_DIR / "model_state.pt"
LOCK_PATH = STUDY_DIR / "selection_lock.json"
PHASE1_DIR = ARTIFACTS_ROOT / "cable" / "phase1_baseline"
TILE_MODEL_PATH = ARTIFACTS_ROOT / "tile" / "model_family_study" / "selected_candidate" / "model_state.pt"
BOTTLE_MODEL_PATH = ARTIFACTS_ROOT / "bottle" / "patchcore_wrn50" / "final_model" / "model_state.pt"
BACKBONE_PATH = pretrained_weights_path()
TRAIN_GOOD_IMAGE = DATASET_ROOT / "cable" / "train" / "good" / "000.png"
CABLE_TEST_ROOT = (DATASET_ROOT / "cable" / "test").resolve()
DATASET_DIR = Path(DATASET_ROOT).resolve()

requires_cable_model = pytest.mark.skipif(
    not (CABLE_MODEL_PATH.is_file() and BACKBONE_PATH.is_file()),
    reason="Cable model-family artifact or ResNet-18 backbone not present in this environment",
)
requires_train_image = pytest.mark.skipif(not TRAIN_GOOD_IMAGE.is_file(), reason="MVTec cable train/good not present")


def _under(path, root: Path) -> bool:
    try:
        resolved = Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError):
        return False
    return resolved == root or root in resolved.parents


@pytest.fixture(autouse=True)
def _fresh_cache_and_final_test_guard(monkeypatch):
    """Refuse any open or decode of a Cable final-test file for the duration of every test."""
    real_load, real_open = pipeline._load_image, builtins.open

    def guarded_load(path):
        if _under(path, CABLE_TEST_ROOT):
            raise AssertionError(f"Cable final-test image must not be read: {path}")
        return real_load(path)

    def guarded_open(file, *args, **kwargs):
        if not isinstance(file, int) and _under(file, CABLE_TEST_ROOT):
            raise AssertionError(f"Cable final-test file must not be opened: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_load_image", guarded_load)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    clear_model_cache()
    yield
    clear_model_cache()


def _digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def _png(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Registration: cable -> knn_l23_128, never ConvAE, never another candidate
# ---------------------------------------------------------------------------

def test_cable_registry_entry_exists_and_resolves_to_the_knn_l23_128_patch_model():
    assert "cable" in SERVING_CONFIGS
    config = get_serving_config("cable")
    assert config.category == "cable"
    assert config.model_name == "knn_l23_128"
    assert config.model_family == MODEL_FAMILY_PATCH_ANOMALY
    assert config.artifact_name == "model_family_study/selected_candidate/model_state"
    assert config.experiment_id == CABLE_EXPERIMENT_ID
    assert "cable/model_family_study/selection_lock.json" in config.provenance
    assert CABLE_LOCK_DIGEST in config.provenance


def test_cable_uses_the_selected_candidate_artifact_and_its_locked_hashes():
    config = get_serving_config("cable")
    assert get_model_path("cable", config.artifact_name) == CABLE_MODEL_PATH
    assert config.expected_sha256 == CABLE_MODEL_SHA256
    assert config.expected_md5 == CABLE_MODEL_MD5
    assert config.expected_config_sha256 == CABLE_CONFIG_SHA256


def test_cable_uses_exactly_the_locked_threshold_and_input_size():
    config = get_serving_config("cable")
    assert config.threshold == CABLE_THRESHOLD
    assert repr(config.threshold) == "2.244786389430004"
    assert (config.threshold_method, config.threshold_parameter) == ("mean_std", 3.0)
    assert config.input_size == (128, 128)


def test_cable_does_not_resolve_to_the_phase1_convae_or_another_candidate():
    config = get_serving_config("cable")
    assert config.model_family != MODEL_FAMILY_CONVAE
    assert config.model_name != "autoencoder"
    assert "phase1_baseline" not in config.artifact_name and "candidates" not in config.artifact_name
    assert config.threshold != CONVAE_PHASE1_THRESHOLD
    assert get_model_path("cable", config.artifact_name).parent == SELECTED_DIR
    assert get_model_path("cable", config.artifact_name).parent != PHASE1_DIR


@pytest.mark.skipif(not LOCK_PATH.is_file(), reason="Cable selection lock not present in this environment")
def test_cable_configuration_matches_the_selection_lock():
    assert _digest(LOCK_PATH, "sha256") == CABLE_LOCK_SHA256
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = get_serving_config("cable")
    assert lock["lock_digest"] == CABLE_LOCK_DIGEST
    assert lock["selected_candidate"] == config.model_name
    assert lock["selected_policy"] == "mean_std_3"
    assert lock["selected_threshold"] == config.threshold
    assert lock["selected_artifact"]["sha256"] == config.expected_sha256
    assert lock["selected_artifact"]["md5"] == config.expected_md5
    assert lock["experiment_id"] == config.experiment_id
    assert lock["selected_configuration"]["image_size"] == config.input_size[0] == config.input_size[1]
    assert _digest(SELECTED_DIR / "model_config.json", "sha256") == config.expected_config_sha256
    assert _digest(SELECTED_DIR / "candidate_manifest.json", "sha256") == CABLE_MANIFEST_SHA256


def test_other_categories_keep_their_existing_configuration():
    # Updated deliberately (all-categories registration): Bottle's Phase 3 ConvAE was replaced by its WRN-50
    # PatchCore model, and every MVTec category is registered (test_ai_all_categories_serving).
    bottle = get_serving_config("bottle")
    assert bottle.model_family == MODEL_FAMILY_PATCHCORE_WRN50 and bottle.model_name == "wrn50_patchcore_crop224"
    assert bottle.artifact_name == "patchcore_wrn50/final_model/model_state" and bottle.input_size == (224, 224)
    assert bottle.threshold == BOTTLE_THRESHOLD
    assert bottle.expected_sha256 == "0c7bd7f45c4769980b7c4e6cb5d6d8762cef35a9088eed045d03bebda8ecb7cb"
    tile = get_serving_config("tile")
    assert tile.model_name == "knn_l23_256" and tile.threshold == TILE_THRESHOLD and tile.input_size == (256, 256)
    assert tile.expected_sha256 == "de00e2774daf02d97e1414fe75b10305e2ad392696449b1627a250b1ec0dd4d7"
    assert set(SERVING_CONFIGS) == set(MVTEC_CATEGORIES)


def test_serving_has_no_threshold_override_or_threshold_computation():
    """The threshold is a constant in the registry: no environment override, no data-derived recomputation."""
    for module in (serving_module, predict_module):
        source = inspect.getsource(module)
        assert "environ" not in source and "getenv" not in source
        assert "compute_threshold" not in source and "policy_threshold" not in source


def test_threshold_k_is_refused_and_the_config_is_immutable(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ValueError, match="threshold_k"):
        predict_image(image, "cable", threshold_k=3.0)
    with pytest.raises(Exception):
        get_serving_config("cable").threshold = 1.0  # type: ignore[misc]
    assert get_serving_config("cable").threshold == CABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Real locked Cable model: loading, integrity, configuration, scoring
# ---------------------------------------------------------------------------

@requires_cable_model
def test_real_cable_model_loads_with_the_locked_configuration():
    detector = load_serving_model(get_serving_config("cable"))
    assert isinstance(detector, PatchAnomalyDetector)
    assert detector.layers == (2, 3)
    assert detector.image_size == 128
    assert detector.aggregation == AGG_TOP1PCT
    assert isinstance(detector.scorer, KNNPatchScorer)
    # 10% per-image coreset of every one of the 224 train/good images: round(0.1 * 16 * 16) = 26 patches each.
    assert tuple(detector.scorer.bank.shape) == (224 * 26, 384)
    assert not detector.extractor.training
    assert not any(p.requires_grad for p in detector.extractor.parameters())
    assert json.loads((SELECTED_DIR / "model_config.json").read_text(encoding="utf-8")) == {
        "detector": "patch_anomaly", "backbone": "resnet18_imagenet_frozen", "layers": [2, 3], "image_size": 128,
        "scorer": "knn", "aggregation": "top1pct_mean", "feature_dim": 384}


@requires_cable_model
def test_real_cable_model_is_cached():
    config = get_serving_config("cable")
    assert load_serving_model(config) is load_serving_model(config)


@requires_cable_model
def test_cable_preprocessing_is_128_and_score_is_knn_l23_top1pct(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    image = _png(tmp_path / "synthetic.png", rng.integers(0, 256, (300, 280, 3)))
    sizes = []
    real_process_image = predict_module.process_image

    def spy(path, *args, **kwargs):
        sizes.append(kwargs.get("target_size", args[0] if args else None))
        return real_process_image(path, *args, **kwargs)

    monkeypatch.setattr(predict_module, "process_image", spy)
    result = predict_image(image, "cable")
    assert sizes == [(128, 128)]
    assert result.input_size == (128, 128)

    # Independent recomputation: project preprocessing at 128 -> ResNet-18 l2+l3 (3x3 avg pool) -> k=1 Euclidean
    # distance to the bank -> mean of the top 1% of patch scores.
    detector = load_serving_model(get_serving_config("cable"))
    processed = real_process_image(image, target_size=(128, 128)).preprocessing.normalized_image
    tensor = torch.from_numpy(processed).permute(2, 0, 1)[None]
    patches = extract_patch_features(detector.extractor, tensor, (2, 3))
    assert patches.shape == (1, 16, 16, 384)
    nearest = torch.cdist(patches.reshape(-1, 384), detector.scorer.bank).min(dim=1).values
    k = math.ceil(0.01 * nearest.numel())
    expected = float(nearest.topk(k).values.mean())
    assert math.isfinite(result.reconstruction_error) and result.reconstruction_error >= 0.0
    assert result.reconstruction_error == pytest.approx(expected, rel=1e-6)
    assert result.prediction == ("good" if result.reconstruction_error <= CABLE_THRESHOLD else "defective")


@requires_cable_model
def test_cable_refuses_any_other_input_size_before_loading(tmp_path, monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("no model may be loaded")

    monkeypatch.setattr(predict_module, "load_serving_model", _must_not_load)
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    for size in [(256, 256), (224, 224), (64, 64), (128, 127)]:
        with pytest.raises(ValueError, match="validated input size"):
            predict_image(image, "cable", image_size=size)


@requires_cable_model
def test_cable_default_call_returns_the_registered_model_and_threshold(tmp_path):
    image = tmp_path / "flat.png"
    image.write_bytes(make_image_bytes("PNG", size=(64, 64), color=(120, 110, 100)))
    result = predict_image(image, "cable")  # exactly how app.inspections.service calls it
    assert isinstance(result, PredictionResult)
    assert result.category == "cable" and result.model_name == "knn_l23_128"
    assert result.threshold == CABLE_THRESHOLD
    assert result.prediction == ("good" if result.reconstruction_error <= CABLE_THRESHOLD else "defective")
    assert predict_image(image, "cable", model_name="knn_l23_128").reconstruction_error == result.reconstruction_error
    for wrong in ("autoencoder", "knn_l23_256", "gauss_l23_256", "knn_l3_128"):
        with pytest.raises(ModelArtifactNotFoundError, match="configured model: 'knn_l23_128'"):
            predict_image(image, "cable", model_name=wrong)


@requires_cable_model
@requires_train_image
def test_real_cable_train_good_and_painted_defect_give_valid_threshold_decisions(tmp_path):
    """A Cable train/good image (training pool, not the final test) and a synthetic defect painted on it."""
    normal = predict_image(TRAIN_GOOD_IMAGE, "cable")
    array = np.asarray(Image.open(TRAIN_GOOD_IMAGE).convert("RGB")).copy()
    h, w = array.shape[:2]
    array[h // 3 : h // 3 + h // 6, w // 3 : w // 3 + w // 6] = 0
    painted = predict_image(_png(tmp_path / "painted.png", array), "cable")
    for result in (normal, painted):
        assert result.prediction == ("good" if result.reconstruction_error <= CABLE_THRESHOLD else "defective")
        assert result.threshold == CABLE_THRESHOLD and result.model_name == "knn_l23_128"
    assert painted.reconstruction_error > normal.reconstruction_error


@requires_cable_model
def test_cable_inference_is_deterministic(tmp_path):
    image = _png(tmp_path / "noise.png", np.random.default_rng(3).integers(0, 256, (128, 128, 3)))
    assert predict_image(image, "cable").reconstruction_error == predict_image(image, "cable").reconstruction_error


@requires_cable_model
def test_cable_startup_and_inference_never_touch_the_dataset(tmp_path, monkeypatch):
    """Loading and scoring read only the artifact + backbone: no train/good scan, no final-test access."""
    touched = []
    real_scandir, real_listdir, real_iterdir, real_glob = os.scandir, os.listdir, Path.iterdir, Path.glob
    real_load, real_open = pipeline._load_image, builtins.open

    def record(path):
        if _under(path, DATASET_DIR):
            touched.append(str(path))

    def scandir(path="."):
        record(path)
        return real_scandir(path)

    def listdir(path="."):
        record(path)
        return real_listdir(path)

    def iterdir(self):
        record(self)
        return real_iterdir(self)

    def glob(self, *args, **kwargs):
        record(self)
        return real_glob(self, *args, **kwargs)

    def load(path):
        record(path)
        return real_load(path)

    def opener(file, *args, **kwargs):
        if not isinstance(file, int):
            record(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(pipeline, "_load_image", load)
    monkeypatch.setattr(builtins, "open", opener)
    monkeypatch.setattr(io, "open", opener)

    image = _png(tmp_path / "noise.png", np.random.default_rng(7).integers(0, 256, (160, 160, 3)))
    clear_model_cache()
    load_serving_model(get_serving_config("cable"))  # startup / model loading
    result = predict_image(image, "cable")  # inference
    assert result.model_name == "knn_l23_128"
    assert touched == []


@requires_cable_model
def test_cable_inference_never_modifies_or_adds_artifacts_and_never_trains(tmp_path, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("inference must not train")

    monkeypatch.setattr(torch.optim, "Adam", _forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", _forbidden)
    monkeypatch.setattr(PatchAnomalyDetector, "save", _forbidden)

    def snapshot():
        return {p: (p.stat().st_mtime_ns, _digest(p, "sha256")) for p in sorted(STUDY_DIR.rglob("*")) if p.is_file()}

    before = snapshot()
    image = _png(tmp_path / "noise.png", np.random.default_rng(5).integers(0, 256, (200, 200, 3)))
    predict_image(image, "cable")
    clear_model_cache()
    predict_image(image, "cable")
    assert snapshot() == before
    assert _digest(CABLE_MODEL_PATH, "sha256") == CABLE_MODEL_SHA256
    assert _digest(CABLE_MODEL_PATH, "md5") == CABLE_MODEL_MD5


# ---------------------------------------------------------------------------
# Missing / tampered artifacts (isolated artifact root) - no fallback of any kind
# ---------------------------------------------------------------------------

def test_missing_cable_artifact_raises_and_never_falls_back(tmp_path, monkeypatch):
    """Only the failed ConvAE, a legacy autoencoder.pt and ANOTHER study candidate exist - none may be used."""
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "cable" / "phase1_baseline" / "autoencoder.pt")
    save_model(build_model(), root / "cable" / "autoencoder.pt")
    other = root / "cable" / "model_family_study" / "candidates" / "knn_l23_256"
    other.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 384)}, other / "model_state.pt")
    (other / "model_config.json").write_text("{}", encoding="utf-8")
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ModelArtifactNotFoundError, match="cable"):
        predict_image(image, "cable")
    with pytest.raises(ModelArtifactNotFoundError, match="cable"):
        load_serving_model(get_serving_config("cable"))


def test_missing_cable_model_config_raises_not_found(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    target = root / "cable" / "model_family_study" / "selected_candidate" / "model_state.pt"
    target.parent.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 384)}, target)  # state without its model_config.json
    with pytest.raises(ModelArtifactNotFoundError, match="cable"):
        load_serving_model(get_serving_config("cable"))


def test_tampered_cable_artifact_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    directory = root / "cable" / "model_family_study" / "selected_candidate"
    directory.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 384)}, directory / "model_state.pt")  # right place, wrong content
    (directory / "model_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="expected MD5 " + CABLE_MODEL_MD5):
        load_serving_model(get_serving_config("cable"))


def test_cable_sha256_and_config_hashes_are_enforced_independently_of_md5(tmp_path, monkeypatch):
    """With the MD5 check satisfied by construction, a SHA-256 or config-hash mismatch still refuses to load."""
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    directory = root / "cable" / "model_family_study" / "selected_candidate"
    directory.mkdir(parents=True)
    state = directory / "model_state.pt"
    torch.save({"bank": torch.zeros(4, 384)}, state)
    (directory / "model_config.json").write_text("{}", encoding="utf-8")
    base = get_serving_config("cable")

    sha_only = serving_module.CategoryServingConfig(**{**base.__dict__, "expected_md5": _digest(state, "md5")})
    with pytest.raises(ModelIntegrityError, match="expected SHA256 " + CABLE_MODEL_SHA256):
        load_serving_model(sha_only)

    config_only = serving_module.CategoryServingConfig(
        **{**base.__dict__, "expected_md5": _digest(state, "md5"), "expected_sha256": _digest(state, "sha256")})
    with pytest.raises(ModelIntegrityError, match="Model configuration"):
        load_serving_model(config_only)


# ---------------------------------------------------------------------------
# Regression: the existing Tile and Bottle serving paths still work alongside Cable
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (TILE_MODEL_PATH.is_file() and BACKBONE_PATH.is_file()), reason="Tile artifact not present")
def test_tile_serving_still_works_next_to_cable(tmp_path):
    image = _png(tmp_path / "noise.png", np.random.default_rng(11).integers(0, 256, (200, 200, 3)))
    tile = predict_image(image, "tile")
    cable = predict_image(image, "cable") if CABLE_MODEL_PATH.is_file() else None
    assert tile.model_name == "knn_l23_256" and tile.threshold == TILE_THRESHOLD and tile.input_size == (256, 256)
    assert tile.prediction == ("good" if tile.reconstruction_error <= TILE_THRESHOLD else "defective")
    if cable is not None:
        assert cable.model_name == "knn_l23_128" and cable.input_size == (128, 128)
        assert load_serving_model(get_serving_config("tile")) is not load_serving_model(get_serving_config("cable"))


@pytest.mark.skipif(not (BOTTLE_MODEL_PATH.is_file() and wide_resnet50_weights_path().is_file()),
                    reason="Bottle WRN-50 artifact or WRN-50-2 backbone not present")
def test_bottle_serving_still_works_next_to_cable(tmp_path):
    # Updated deliberately: Bottle is served by its WRN-50 PatchCore model (crop224) since the all-categories
    # registration.
    image = _png(tmp_path / "noise.png", np.random.default_rng(13).integers(0, 256, (200, 200, 3)))
    bottle = predict_image(image, "bottle")
    assert bottle.model_name == "wrn50_patchcore_crop224" and bottle.threshold == BOTTLE_THRESHOLD
    assert bottle.input_size == (224, 224)
    assert bottle.prediction == ("good" if bottle.reconstruction_error <= BOTTLE_THRESHOLD else "defective")
