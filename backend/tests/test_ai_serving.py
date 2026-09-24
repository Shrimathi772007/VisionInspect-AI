"""Category-specific AI serving: which model and which threshold production inference uses.

Covers the serving fix that made the validated Milestone 4 Phase 3 Bottle model (and its
stored, validated threshold) the model production inference actually serves, in place of the
original Phase 1 model with a K=3 threshold recomputed from all 209 train/good images on every
call. Tests that need the real (Git-ignored) model artifact / MVTec dataset skip when they are
absent, the same convention as the other AI validation tests.
"""

import ast
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.ai.inference import ModelArtifactNotFoundError, ModelIntegrityError, PredictionResult, predict_image
from app.ai.inference.serving import (
    SERVING_CONFIGS,
    CategoryServingConfig,
    clear_model_cache,
    get_serving_config,
    get_supported_categories,
    load_serving_model,
)
from app.ai.training import build_model, save_model
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.inspections.storage import DATASET_ROOT
from tests.conftest import make_image_bytes

PHASE3_MD5 = "76478dd6996feb50fadaf5e5e5be1ae4"
PHASE1_MD5 = "2f470c30834baf80252f1d352c7fb37f"
PHASE3_THRESHOLD = 0.0028031117030001018

PHASE3_MODEL_PATH = ARTIFACTS_ROOT / "bottle" / "phase3_validation" / "autoencoder.pt"
PHASE1_MODEL_PATH = ARTIFACTS_ROOT / "bottle" / "autoencoder.pt"
PHASE3_REPORT_PATH = ARTIFACTS_ROOT / "bottle" / "evaluation_reports" / "phase3_validated_model_report.json"
GOOD_IMAGE = DATASET_ROOT / "bottle" / "test" / "good" / "000.png"
DEFECTIVE_IMAGE = DATASET_ROOT / "bottle" / "test" / "contamination" / "000.png"

requires_phase3_model = pytest.mark.skipif(
    not PHASE3_MODEL_PATH.is_file(), reason="Phase 3 Bottle model artifact not present in this environment"
)
requires_bottle_images = pytest.mark.skipif(
    not (GOOD_IMAGE.is_file() and DEFECTIVE_IMAGE.is_file()), reason="MVTec bottle test images not present"
)


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

def test_bottle_resolves_to_the_phase3_model_path():
    config = get_serving_config("bottle")
    assert get_model_path("bottle", config.artifact_name) == PHASE3_MODEL_PATH
    assert config.artifact_name == "phase3_validation/autoencoder"


def test_bottle_does_not_resolve_to_the_phase1_model_path():
    config = get_serving_config("bottle")
    resolved = get_model_path("bottle", config.artifact_name)
    assert resolved != PHASE1_MODEL_PATH
    assert resolved.parent.name == "phase3_validation"


def test_bottle_uses_the_phase3_validated_threshold():
    config = get_serving_config("bottle")
    assert config.threshold == PHASE3_THRESHOLD
    assert (config.threshold_method, config.threshold_parameter) == ("mean_std", 1.5)
    assert config.expected_md5 == PHASE3_MD5
    assert config.model_name == "autoencoder"  # unchanged database contract (Inspection.ai_model_name)
    assert "phase3_validated_model_report.json" in config.provenance


@pytest.mark.skipif(not PHASE3_REPORT_PATH.is_file(), reason="Phase 3 report not present in this environment")
def test_configured_threshold_and_hash_match_the_phase3_report():
    """The stored constants must stay traceable to the authoritative Phase 3 artifact."""
    report = json.loads(PHASE3_REPORT_PATH.read_text())
    selected = report["selection"]["selected_final_test_result"]
    config = get_serving_config("bottle")

    assert config.threshold == selected["threshold"]
    assert config.threshold_method == selected["method"]
    assert config.threshold_parameter == selected["parameter"]
    assert config.expected_md5 == report["phase3_model"]["md5"]


@requires_phase3_model
def test_phase3_model_artifact_hash_is_unchanged():
    assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5


def test_serving_configuration_is_deterministic_and_immutable():
    assert get_serving_config("bottle") == get_serving_config("bottle")
    with pytest.raises(FrozenInstanceError):
        get_serving_config("bottle").threshold = 1.0  # type: ignore[misc]


def test_only_bottle_has_an_active_serving_configuration():
    """No fake configuration for categories without a validated model."""
    assert set(SERVING_CONFIGS) == {"bottle"}
    assert get_supported_categories() == ("bottle",)


# ---------------------------------------------------------------------------
# Unconfigured categories / no fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "category",
    ["cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
     "toothbrush", "transistor", "wood", "zipper", "not_a_category"],
)
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


def test_missing_bottle_phase3_model_raises_and_never_falls_back_to_phase1(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    # Only the Phase 1 artifact exists; the configured Phase 3 artifact is missing.
    save_model(build_model(), root / "bottle" / "autoencoder.pt")

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match="bottle"):
        predict_image(image_path, "bottle")


def test_artifact_that_is_not_the_validated_model_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "bottle" / "phase3_validation" / "autoencoder.pt")  # right place, wrong weights

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelIntegrityError, match="expected MD5 " + PHASE3_MD5):
        predict_image(image_path, "bottle")


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


@pytest.mark.parametrize("bad_name", ["phase1", "other_model", "phase3_validation/autoencoder", "../autoencoder", ""])
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

    with pytest.raises(ModelArtifactNotFoundError, match="configured model: 'autoencoder'"):
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
# Real Bottle inference (real Phase 3 artifact + real MVTec images)
# ---------------------------------------------------------------------------

def _forbid_dataset_scanning(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("inference must not scan the training set")

    monkeypatch.setattr("app.ai.training.dataset.discover_train_samples", _forbidden)
    monkeypatch.setattr("app.ai.evaluation.threshold.compute_threshold", _forbidden)


@requires_phase3_model
@requires_bottle_images
def test_real_bottle_inference_serves_phase3_model_with_stored_threshold(monkeypatch):
    _forbid_dataset_scanning(monkeypatch)

    # Count image decodes: exactly one (the inspected image), not 1 + 209 training images.
    import app.ai.inference.predict as predict_module

    decoded = []
    real_process_image = predict_module.process_image

    def _counting_process_image(path, *args, **kwargs):
        decoded.append(Path(path))
        return real_process_image(path, *args, **kwargs)

    monkeypatch.setattr(predict_module, "process_image", _counting_process_image)

    result = predict_image(GOOD_IMAGE, "bottle")

    assert decoded == [GOOD_IMAGE]
    assert isinstance(result, PredictionResult)
    assert result.category == "bottle"
    assert result.prediction in ("good", "defective")
    assert result.reconstruction_error >= 0.0
    assert result.threshold == PHASE3_THRESHOLD
    assert result.model_name == "autoencoder"
    assert result.input_size == (128, 128)
    assert result.processing_time_ms >= 0.0
    assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5  # inference never touches the artifact


@requires_phase3_model
@requires_bottle_images
def test_real_bottle_prediction_is_decided_by_the_stored_threshold():
    good = predict_image(GOOD_IMAGE, "bottle")
    defective = predict_image(DEFECTIVE_IMAGE, "bottle")

    for result in (good, defective):
        expected = "good" if result.reconstruction_error <= PHASE3_THRESHOLD else "defective"
        assert result.prediction == expected


@requires_phase3_model
@requires_bottle_images
def test_real_bottle_inference_is_deterministic():
    first = predict_image(GOOD_IMAGE, "bottle")
    second = predict_image(GOOD_IMAGE, "bottle")
    assert first.reconstruction_error == second.reconstruction_error
    assert first.prediction == second.prediction
    assert first.threshold == second.threshold


@requires_phase3_model
def test_serving_loads_exactly_the_phase3_artifact_not_phase1(tmp_path, monkeypatch):
    """Point the artifact root at a copy holding BOTH files; the Phase 3 one must be what is served."""
    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    target = root / "bottle" / "phase3_validation" / "autoencoder.pt"
    target.parent.mkdir(parents=True)
    shutil.copyfile(PHASE3_MODEL_PATH, target)
    if PHASE1_MODEL_PATH.is_file():
        shutil.copyfile(PHASE1_MODEL_PATH, root / "bottle" / "autoencoder.pt")
        assert _md5(root / "bottle" / "autoencoder.pt") == PHASE1_MD5

    model = load_serving_model(get_serving_config("bottle"))  # passes the MD5 check only for Phase 3 weights
    assert model is not None

    # Remove the Phase 3 copy: with only Phase 1 left, serving must refuse rather than fall back.
    clear_model_cache()
    target.unlink()
    with pytest.raises(ModelArtifactNotFoundError):
        load_serving_model(get_serving_config("bottle"))


@requires_phase3_model
@requires_bottle_images
def test_real_bottle_legacy_style_call_returns_exactly_the_configured_threshold(monkeypatch):
    """Old-style call (explicit model_name, original positional order) on real data: Phase 3
    model, exactly the configured threshold, and no dataset scan."""
    _forbid_dataset_scanning(monkeypatch)

    modern = predict_image(GOOD_IMAGE, "bottle")
    legacy = predict_image(GOOD_IMAGE, "bottle", "autoencoder", (128, 128))

    assert legacy.threshold == PHASE3_THRESHOLD
    assert legacy.model_name == "autoencoder"
    assert legacy.reconstruction_error == modern.reconstruction_error
    assert legacy.prediction == modern.prediction
    assert _md5(PHASE3_MODEL_PATH) == PHASE3_MD5
