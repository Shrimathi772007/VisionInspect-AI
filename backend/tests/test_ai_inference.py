import pytest

from app.ai.inference import (
    DEFECTIVE_PREDICTION,
    GOOD_PREDICTION,
    ModelArtifactNotFoundError,
    PredictionResult,
    predict_image,
)
from app.ai.inference.serving import SERVING_CONFIGS, CategoryServingConfig, clear_model_cache
from app.ai.preprocessing.errors import ImageDecodeError, ImageNotFoundError
from app.ai.training import build_model, save_model
from app.ai.training.artifacts import get_model_path
from tests.conftest import make_image_bytes

CATEGORY = "widget"
IMAGE_SIZE = (32, 32)
CONFIGURED_THRESHOLD = 0.0123


def _register_widget(monkeypatch, threshold=CONFIGURED_THRESHOLD):
    """Give the throwaway category a serving configuration, exactly as a real validated
    category would have one. Without an entry a category has no AI support at all."""
    config = CategoryServingConfig(
        category=CATEGORY,
        artifact_name="autoencoder",
        model_name="autoencoder",
        threshold=threshold,
        threshold_method="mean_std",
        threshold_parameter=1.5,
        expected_md5=None,
        provenance="test fixture",
    )
    monkeypatch.setitem(SERVING_CONFIGS, CATEGORY, config)
    return config


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def fake_category(tmp_path, monkeypatch):
    """A saved model + serving configuration for a throwaway category. Deliberately NO
    dataset at all (no train/good, no test/): inference must not need one, so any attempt
    to read training or test data would fail these tests. The artifact root is monkeypatched
    rather than touching the real bottle model."""
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    _register_widget(monkeypatch)
    model = build_model()
    save_model(model, get_model_path(CATEGORY))

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(64, 48), color=(10, 200, 10)))

    return image_path


# ---------------------------------------------------------------------------
# Prediction schema / output fields
# ---------------------------------------------------------------------------

def test_prediction_result_has_expected_fields(fake_category):
    image_path = fake_category
    result = predict_image(image_path, CATEGORY, image_size=IMAGE_SIZE)

    assert isinstance(result, PredictionResult)
    assert result.category == CATEGORY
    assert result.prediction in (GOOD_PREDICTION, DEFECTIVE_PREDICTION)
    assert isinstance(result.reconstruction_error, float)
    assert isinstance(result.threshold, float)
    assert result.model_name == "autoencoder"
    assert result.input_size == IMAGE_SIZE
    assert result.processing_time_ms >= 0.0


def test_prediction_never_carries_a_ground_truth_or_confidence_field(fake_category):
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    field_names = set(result.__dataclass_fields__)
    assert "label" not in field_names
    assert "ground_truth" not in field_names
    assert "confidence" not in field_names


# ---------------------------------------------------------------------------
# Model loading (reuses the existing Phase 4 loader)
# ---------------------------------------------------------------------------

def test_predict_image_uses_existing_model_loader(fake_category):
    """No exception, and the result is derived from a real forward pass - if loading were
    broken this would raise, not silently succeed."""
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.reconstruction_error >= 0.0


def test_missing_model_artifact_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")  # empty - no saved model
    _register_widget(monkeypatch)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match=CATEGORY):
        predict_image(image_path, CATEGORY, image_size=IMAGE_SIZE)


def test_unconfigured_category_raises_model_artifact_not_found(tmp_path, monkeypatch):
    """A model file on disk is not enough - without a serving configuration the category has
    no AI support (an untrained/unvalidated category must never appear to have it)."""
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    save_model(build_model(), get_model_path("gadget"))  # artifact exists, configuration does not

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match="gadget"):
        predict_image(image_path, "gadget", image_size=IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------

def test_reconstruction_error_is_non_negative(fake_category):
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.reconstruction_error >= 0.0


def test_reconstruction_error_is_deterministic(fake_category):
    result_a = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    result_b = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result_a.reconstruction_error == result_b.reconstruction_error


def test_repeated_inference_on_same_image_gives_same_result(fake_category):
    """Same everything except processing_time_ms, which is a real wall-clock measurement
    and is expected to vary slightly between runs."""
    result_a = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    result_b = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result_a.category == result_b.category
    assert result_a.prediction == result_b.prediction
    assert result_a.reconstruction_error == result_b.reconstruction_error
    assert result_a.threshold == result_b.threshold


# ---------------------------------------------------------------------------
# Threshold (configured, never recomputed)
# ---------------------------------------------------------------------------

def test_threshold_is_the_configured_threshold(fake_category):
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.threshold == CONFIGURED_THRESHOLD


def test_inference_needs_no_dataset(fake_category, monkeypatch):
    """The fake category has no dataset directory at all, and dataset discovery is booby-trapped -
    if inference ever touched training/test data or labels, this would raise."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("inference must not read dataset images")

    monkeypatch.setattr("app.ai.training.dataset.discover_train_samples", _forbidden)
    monkeypatch.setattr("app.ai.training.dataset.discover_test_samples", _forbidden)
    monkeypatch.setattr("app.ai.evaluation.threshold.compute_threshold", _forbidden)

    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction in (GOOD_PREDICTION, DEFECTIVE_PREDICTION)


# ---------------------------------------------------------------------------
# Prediction rule / boundary behavior (must match the validated evaluation exactly)
# ---------------------------------------------------------------------------

def _pin_error(monkeypatch, error):
    _register_widget(monkeypatch, threshold=0.05)
    monkeypatch.setattr("app.ai.inference.predict.compute_reconstruction_error", lambda *a, **k: error)


def test_boundary_error_equal_to_threshold_is_good(fake_category, monkeypatch):
    _pin_error(monkeypatch, 0.05)
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction == GOOD_PREDICTION


def test_boundary_error_just_above_threshold_is_defective(fake_category, monkeypatch):
    _pin_error(monkeypatch, 0.0501)
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction == DEFECTIVE_PREDICTION


def test_boundary_error_just_below_threshold_is_good(fake_category, monkeypatch):
    _pin_error(monkeypatch, 0.0499)
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction == GOOD_PREDICTION


# ---------------------------------------------------------------------------
# Invalid image handling
# ---------------------------------------------------------------------------

def test_missing_image_raises_existing_preprocessing_error(fake_category, tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ImageNotFoundError):
        predict_image(missing_path, CATEGORY, image_size=IMAGE_SIZE)


def test_corrupt_image_raises_existing_preprocessing_error(fake_category, tmp_path):
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not a real image")
    with pytest.raises(ImageDecodeError):
        predict_image(corrupt_path, CATEGORY, image_size=IMAGE_SIZE)
