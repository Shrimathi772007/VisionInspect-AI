"""Category-specific AI serving: which model and which threshold production inference uses.

Originally covered the serving fix that made the validated Milestone 4 Phase 3 Bottle ConvAE (and its
stored, validated threshold) the served model in place of the Phase 1 model with a K=3 threshold
recomputed from all 209 train/good images on every call. Since the all-categories registration, Bottle
is served by its locked WRN-50 PatchCore model (crop224), which replaced the Phase 3 ConvAE by protocol;
the Bottle tests below were updated deliberately to that model, and the retired ConvAE (still on disk,
untouched) is kept as a test-local configuration so the ConvAE serving path stays covered.

Real-image tests use COPIES of Bottle train/good images only - never dataset/bottle/test/. Tests that need
the real (Git-ignored) model artifacts / MVTec dataset skip when they are absent, the same convention as
the other AI validation tests.
"""

import ast
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from app.ai.inference import ModelArtifactNotFoundError, ModelIntegrityError, PredictionResult, predict_image
from app.ai.inference.serving import (
    MODEL_FAMILY_CONVAE,
    MODEL_FAMILY_PATCHCORE_WRN50,
    SERVING_CONFIGS,
    CategoryServingConfig,
    clear_model_cache,
    get_serving_config,
    get_supported_categories,
    load_serving_model,
)
from app.ai.models.patchcore import PatchCoreDetector
from app.ai.models.wide_resnet50 import load_pretrained_wide_resnet50_2, wide_resnet50_weights_path
from app.ai.training import build_model, save_model
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.storage import DATASET_ROOT
from tests.conftest import make_image_bytes

PHASE3_MD5 = "76478dd6996feb50fadaf5e5e5be1ae4"
PHASE1_MD5 = "2f470c30834baf80252f1d352c7fb37f"
PHASE3_THRESHOLD = 0.0028031117030001018

PHASE3_MODEL_PATH = ARTIFACTS_ROOT / "bottle" / "phase3_validation" / "autoencoder.pt"
PHASE1_MODEL_PATH = ARTIFACTS_ROOT / "bottle" / "autoencoder.pt"
PHASE3_REPORT_PATH = ARTIFACTS_ROOT / "bottle" / "evaluation_reports" / "phase3_validated_model_report.json"

# The served Bottle model: WRN-50 PatchCore, ai_models/bottle/patchcore_wrn50/lock.json.
BOTTLE_MODEL_NAME = "wrn50_patchcore_crop224"
BOTTLE_THRESHOLD = 1.6858729828595898
BOTTLE_STATE_SHA256 = "0c7bd7f45c4769980b7c4e6cb5d6d8762cef35a9088eed045d03bebda8ecb7cb"
BOTTLE_WRN_DIR = ARTIFACTS_ROOT / "bottle" / "patchcore_wrn50"
BOTTLE_MODEL_PATH = BOTTLE_WRN_DIR / "final_model" / "model_state.pt"
BOTTLE_LOCK_PATH = BOTTLE_WRN_DIR / "lock.json"
WRN_BACKBONE_PATH = wide_resnet50_weights_path()
TRAIN_GOOD_IMAGES = [DATASET_ROOT / "bottle" / "train" / "good" / name for name in ("000.png", "001.png")]

# The retired Bottle ConvAE (Milestone 4 Phase 3), exactly as it was registered before the WRN-50 replacement.
# Not served; kept only so the ConvAE serving path and its provenance stay under test.
RETIRED_BOTTLE_CONVAE = CategoryServingConfig(
    category="bottle",
    artifact_name="phase3_validation/autoencoder",
    model_name="autoencoder",
    threshold=PHASE3_THRESHOLD,
    threshold_method="mean_std",
    threshold_parameter=1.5,
    expected_md5=PHASE3_MD5,
    provenance="backend/ai_models/bottle/evaluation_reports/phase3_validated_model_report.json",
)

requires_phase3_model = pytest.mark.skipif(
    not PHASE3_MODEL_PATH.is_file(), reason="Phase 3 Bottle model artifact not present in this environment"
)
requires_bottle_model = pytest.mark.skipif(
    not (BOTTLE_MODEL_PATH.is_file() and WRN_BACKBONE_PATH.is_file()),
    reason="Bottle WRN-50 model or WRN-50-2 backbone not present in this environment",
)
requires_bottle_images = pytest.mark.skipif(
    not all(p.is_file() for p in TRAIN_GOOD_IMAGES), reason="MVTec bottle train/good images not present"
)


@pytest.fixture
def train_good_copies(tmp_path):
    """Copies of two Bottle train/good images (the dataset files are only read)."""
    copies = []
    for source in TRAIN_GOOD_IMAGES:
        target = tmp_path / f"bottle_train_good_{source.name}"
        shutil.copyfile(source, target)
        copies.append(target)
    return copies


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def _md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Bottle production configuration
# ---------------------------------------------------------------------------

# Updated deliberately (all-categories registration): Bottle now resolves to its WRN-50 PatchCore model, which
# replaced the Phase 3 ConvAE by protocol. Each test keeps its original intent (exact artifact, never Phase 1,
# exact locked threshold, traceable to the authoritative record) against the new model.

def test_bottle_resolves_to_the_wrn50_patchcore_model_path():
    config = get_serving_config("bottle")
    assert get_model_path("bottle", config.artifact_name) == BOTTLE_MODEL_PATH
    assert config.artifact_name == "patchcore_wrn50/final_model/model_state"
    assert config.model_family == MODEL_FAMILY_PATCHCORE_WRN50


def test_bottle_does_not_resolve_to_the_phase1_or_the_retired_phase3_convae():
    config = get_serving_config("bottle")
    resolved = get_model_path("bottle", config.artifact_name)
    assert resolved != PHASE1_MODEL_PATH
    assert resolved != PHASE3_MODEL_PATH
    assert resolved.parent.parent.name == "patchcore_wrn50"
    assert config.model_family != MODEL_FAMILY_CONVAE


def test_bottle_uses_the_wrn50_locked_threshold():
    config = get_serving_config("bottle")
    assert config.threshold == BOTTLE_THRESHOLD
    assert config.threshold != PHASE3_THRESHOLD
    assert (config.threshold_method, config.threshold_parameter) == ("mean_std", 2.5)
    assert config.expected_sha256 == BOTTLE_STATE_SHA256
    assert config.model_name == BOTTLE_MODEL_NAME  # new rows; old rows keep their stored "autoencoder"
    assert (config.input_mode, config.input_size, config.aggregation) == ("crop224", (224, 224), "max")
    assert "patchcore_wrn50/lock.json" in config.provenance


@pytest.mark.skipif(not BOTTLE_LOCK_PATH.is_file(), reason="Bottle WRN-50 lock not present in this environment")
def test_configured_threshold_and_hash_match_the_bottle_wrn50_lock():
    """The stored constants must stay traceable to the authoritative lock."""
    lock = json.loads(BOTTLE_LOCK_PATH.read_text(encoding="utf-8"))
    config = get_serving_config("bottle")

    assert config.threshold == lock["threshold"]
    assert lock["policy"] == "mean_std_2.5"
    assert config.input_mode == lock["mode"] and config.aggregation == lock["aggregation"]
    assert config.expected_sha256 == lock["model_state_sha256"]
    assert config.expected_config_sha256 == lock["model_config_sha256"]
    assert config.expected_backbone_sha256 == lock["backbone_weights_sha256"]
    assert config.lock_digest == lock["lock_digest"]


@pytest.mark.skipif(not PHASE3_REPORT_PATH.is_file(), reason="Phase 3 report not present in this environment")
def test_retired_convae_constants_match_the_phase3_report():
    """The retired ConvAE's record stays traceable (old inspection rows were scored with it)."""
    report = json.loads(PHASE3_REPORT_PATH.read_text())
    selected = report["selection"]["selected_final_test_result"]

    assert RETIRED_BOTTLE_CONVAE.threshold == selected["threshold"]
    assert RETIRED_BOTTLE_CONVAE.threshold_method == selected["method"]
    assert RETIRED_BOTTLE_CONVAE.threshold_parameter == selected["parameter"]
    assert RETIRED_BOTTLE_CONVAE.expected_md5 == report["phase3_model"]["md5"]


@requires_phase3_model
def test_phase3_model_artifact_hash_is_unchanged():
    assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5


def test_serving_configuration_is_deterministic_and_immutable():
    assert get_serving_config("bottle") == get_serving_config("bottle")
    with pytest.raises(FrozenInstanceError):
        get_serving_config("bottle").threshold = 1.0  # type: ignore[misc]


def test_every_mvtec_category_and_nothing_else_has_a_serving_configuration():
    """Updated deliberately (all-categories registration): every MVTec category has exactly one registered final
    model (see test_ai_all_categories_serving), and no other key exists."""
    assert set(SERVING_CONFIGS) == set(MVTEC_CATEGORIES)
    assert len(SERVING_CONFIGS) == 15
    assert set(get_supported_categories()) == set(MVTEC_CATEGORIES)


# ---------------------------------------------------------------------------
# Unconfigured categories / no fallback
# ---------------------------------------------------------------------------

# Updated deliberately: the 12 MVTec names this used to list are all served now; only unknown names remain
# unconfigured, and they must still never reach Bottle's (or any) artifact.
@pytest.mark.parametrize("category", ["not_a_category", "Bottle", "bottles", "mvtec_bottle", "screws"])
def test_unconfigured_category_never_uses_the_bottle_model(category, tmp_path, monkeypatch):
    # Even with Bottle's Phase 3 artifact sitting in the artifact root, other categories must not reach it.
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "bottle" / "phase3_validation" / "autoencoder.pt")
    save_model(build_model(), root / "bottle" / "autoencoder.pt")

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match=category):
        predict_image(image_path, category)


def test_missing_bottle_model_raises_and_never_falls_back_to_a_convae(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    # Only the Phase 1 and the retired Phase 3 ConvAE artifacts exist; the configured WRN-50 model is missing.
    save_model(build_model(), root / "bottle" / "autoencoder.pt")
    save_model(build_model(), root / "bottle" / "phase3_validation" / "autoencoder.pt")

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match="bottle"):
        predict_image(image_path, "bottle")


def test_artifact_that_is_not_the_validated_model_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    directory = root / "bottle" / "patchcore_wrn50" / "final_model"
    directory.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 1536)}, directory / "model_state.pt")  # right place, wrong bank
    (directory / "model_config.json").write_text("{}", encoding="utf-8")

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelIntegrityError, match="expected SHA256 " + BOTTLE_STATE_SHA256):
        predict_image(image_path, "bottle")


@requires_phase3_model
def test_retired_convae_still_loads_through_the_convae_serving_path():
    """The ConvAE family stays supported (and MD5-verified) although no category is served by it any more."""
    model = load_serving_model(RETIRED_BOTTLE_CONVAE)
    assert model is load_serving_model(RETIRED_BOTTLE_CONVAE)
    assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

def _widget_config():
    return CategoryServingConfig(
        category="widget",
        artifact_name="autoencoder",
        model_name="autoencoder",
        threshold=0.01,
        threshold_method="mean_std",
        threshold_parameter=1.5,
        expected_md5=None,
        provenance="test fixture",
    )


def test_model_is_loaded_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    save_model(build_model(), get_model_path("widget"))

    config = _widget_config()
    assert load_serving_model(config) is load_serving_model(config)


def test_cache_reloads_when_the_artifact_on_disk_changes(tmp_path, monkeypatch):
    import os

    import torch

    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    path = get_model_path("widget")
    torch.manual_seed(1)
    save_model(build_model(), path)

    config = _widget_config()
    first = load_serving_model(config)
    first_weights = next(first.parameters()).detach().clone()

    torch.manual_seed(2)
    save_model(build_model(), path)  # same architecture, different weights
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    second = load_serving_model(config)
    assert second is not first
    assert not torch.equal(first_weights, next(second.parameters()).detach())


def test_cache_is_scoped_per_category(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    save_model(build_model(), get_model_path("widget"))
    save_model(build_model(), get_model_path("gadget"))

    widget = _widget_config()
    gadget = CategoryServingConfig(**{**widget.__dict__, "category": "gadget"})
    assert load_serving_model(widget) is not load_serving_model(gadget)


# ---------------------------------------------------------------------------
# No threshold recomputation from the training set (regression guards)
# ---------------------------------------------------------------------------

def test_predict_module_no_longer_references_training_set_threshold_recomputation():
    """Fails if someone reintroduces compute_threshold(discover_train_samples(...)) into inference."""
    import app.ai.inference.predict as predict_module
    import app.ai.inference.serving as serving_module

    forbidden = {"compute_threshold", "discover_train_samples", "discover_test_samples", "DEFAULT_THRESHOLD_K"}
    for module in (predict_module, serving_module):
        tree = ast.parse(Path(module.__file__).read_text())
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        referenced |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert not (forbidden & referenced), f"{module.__name__} references {forbidden & referenced}"


# ---------------------------------------------------------------------------
# Backward-compatible parameters (model_name / threshold_k) cannot bypass the registry
# ---------------------------------------------------------------------------

def test_legacy_model_name_autoencoder_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    monkeypatch.setitem(SERVING_CONFIGS, "widget", _widget_config())
    save_model(build_model(), get_model_path("widget"))
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    implicit = predict_image(image_path, "widget", image_size=(32, 32))
    positional = predict_image(image_path, "widget", "autoencoder", (32, 32))  # original positional order
    keyword = predict_image(image_path, "widget", model_name="autoencoder", image_size=(32, 32))

    assert implicit.reconstruction_error == positional.reconstruction_error == keyword.reconstruction_error
    assert implicit.threshold == positional.threshold == keyword.threshold == 0.01
    assert keyword.model_name == "autoencoder"


@pytest.mark.parametrize(
    "bad_name", ["autoencoder", "phase1", "other_model", "phase3_validation/autoencoder", "../autoencoder", ""]
)
def test_conflicting_model_name_is_refused_and_loads_nothing(bad_name, tmp_path, monkeypatch):
    """A caller cannot pick another artifact by name - not even one that exists on disk."""
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "bottle" / "autoencoder.pt")  # a Phase 1-style file exists
    save_model(build_model(), root / "bottle" / "phase1.pt")
    save_model(build_model(), root / "bottle" / "other_model.pt")

    def _must_not_load(*args, **kwargs):
        raise AssertionError("no model may be loaded for a conflicting model_name")

    monkeypatch.setattr("app.ai.inference.predict.load_serving_model", _must_not_load)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match=f"configured model: '{BOTTLE_MODEL_NAME}'"):
        predict_image(image_path, "bottle", model_name=bad_name)


@pytest.mark.parametrize("threshold_k", [3.0, 1.5, 0.0, 1])
def test_threshold_k_is_rejected_and_never_recomputes_a_threshold(threshold_k, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("must not reach model loading, dataset scanning or threshold computation")

    monkeypatch.setattr("app.ai.inference.predict.load_serving_model", _forbidden)
    monkeypatch.setattr("app.ai.training.dataset.discover_train_samples", _forbidden)
    monkeypatch.setattr("app.ai.evaluation.threshold.compute_threshold", _forbidden)

    with pytest.raises(ValueError, match="threshold_k"):
        predict_image(Path("x.png"), "bottle", threshold_k=threshold_k)


def test_threshold_k_none_is_the_default_and_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    monkeypatch.setitem(SERVING_CONFIGS, "widget", _widget_config())
    save_model(build_model(), get_model_path("widget"))
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    result = predict_image(image_path, "widget", image_size=(32, 32), threshold_k=None)
    assert result.threshold == 0.01


# ---------------------------------------------------------------------------
# Real Bottle inference (real WRN-50 artifact + copies of Bottle train/good images)
# Updated deliberately from the Phase 3 ConvAE versions: same checks, new model, train/good images only.
# ---------------------------------------------------------------------------

def _forbid_dataset_scanning(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("inference must not scan the training set")

    monkeypatch.setattr("app.ai.training.dataset.discover_train_samples", _forbidden)
    monkeypatch.setattr("app.ai.evaluation.threshold.compute_threshold", _forbidden)


@requires_bottle_model
@requires_bottle_images
def test_real_bottle_inference_serves_the_wrn50_model_with_its_locked_threshold(monkeypatch, train_good_copies):
    _forbid_dataset_scanning(monkeypatch)

    # Count image decodes: exactly one (the inspected image), not 1 + 209 training images.
    import app.ai.inference.predict as predict_module

    decoded = []
    real_decode_image = predict_module.decode_image

    def _counting_decode_image(path, *args, **kwargs):
        decoded.append(Path(path))
        return real_decode_image(path, *args, **kwargs)

    monkeypatch.setattr(predict_module, "decode_image", _counting_decode_image)

    image = train_good_copies[0]
    result = predict_image(image, "bottle")

    assert decoded == [image]
    assert isinstance(result, PredictionResult)
    assert result.category == "bottle"
    assert result.prediction in ("good", "defective")
    assert result.reconstruction_error >= 0.0
    assert result.threshold == BOTTLE_THRESHOLD
    assert result.model_name == BOTTLE_MODEL_NAME
    assert result.input_size == (224, 224)
    assert result.processing_time_ms >= 0.0
    assert isinstance(load_serving_model(get_serving_config("bottle")), PatchCoreDetector)
    if PHASE3_MODEL_PATH.is_file():
        assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5  # the retired ConvAE file stays untouched


@requires_bottle_model
@requires_bottle_images
def test_real_bottle_prediction_is_decided_by_the_stored_threshold(train_good_copies):
    for image in train_good_copies:
        result = predict_image(image, "bottle")
        expected = "good" if result.reconstruction_error <= BOTTLE_THRESHOLD else "defective"
        assert result.prediction == expected


@requires_bottle_model
@requires_bottle_images
def test_real_bottle_inference_is_deterministic(train_good_copies):
    first = predict_image(train_good_copies[0], "bottle")
    second = predict_image(train_good_copies[0], "bottle")
    assert first.reconstruction_error == second.reconstruction_error
    assert first.prediction == second.prediction
    assert first.threshold == second.threshold


@requires_bottle_model
def test_serving_loads_exactly_the_wrn50_artifact_not_a_convae(tmp_path, monkeypatch):
    """Point the artifact root at a copy holding the WRN-50 model AND both ConvAEs; the WRN-50 one must be what is
    served, and with it removed serving must refuse rather than fall back."""
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    monkeypatch.setattr(
        "app.ai.inference.serving.load_pretrained_wide_resnet50_2",
        lambda: load_pretrained_wide_resnet50_2(WRN_BACKBONE_PATH),
    )
    target_dir = root / "bottle" / "patchcore_wrn50" / "final_model"
    shutil.copytree(BOTTLE_MODEL_PATH.parent, target_dir)
    if PHASE3_MODEL_PATH.is_file():
        (root / "bottle" / "phase3_validation").mkdir(parents=True)
        shutil.copyfile(PHASE3_MODEL_PATH, root / "bottle" / "phase3_validation" / "autoencoder.pt")
    if PHASE1_MODEL_PATH.is_file():
        shutil.copyfile(PHASE1_MODEL_PATH, root / "bottle" / "autoencoder.pt")
        assert _md5(root / "bottle" / "autoencoder.pt") == PHASE1_MD5

    model = load_serving_model(get_serving_config("bottle"))  # passes the SHA-256 checks only for the WRN-50 bank
    assert isinstance(model, PatchCoreDetector)

    # Remove the WRN-50 copy: with only the ConvAEs left, serving must refuse rather than fall back.
    clear_model_cache()
    (target_dir / "model_state.pt").unlink()
    with pytest.raises(ModelArtifactNotFoundError):
        load_serving_model(get_serving_config("bottle"))


@requires_bottle_model
@requires_bottle_images
def test_real_bottle_legacy_style_call_returns_exactly_the_configured_threshold(monkeypatch, train_good_copies):
    """Old-style call (explicit model_name, original positional order) on real data: the registered model,
    exactly the configured threshold, and no dataset scan."""
    _forbid_dataset_scanning(monkeypatch)

    modern = predict_image(train_good_copies[0], "bottle")
    legacy = predict_image(train_good_copies[0], "bottle", BOTTLE_MODEL_NAME, (224, 224))

    assert legacy.threshold == BOTTLE_THRESHOLD
    assert legacy.model_name == BOTTLE_MODEL_NAME
    assert legacy.reconstruction_error == modern.reconstruction_error
    assert legacy.prediction == modern.prediction
    with pytest.raises(ValueError, match="validated input size"):
        predict_image(train_good_copies[0], "bottle", BOTTLE_MODEL_NAME, (128, 128))
