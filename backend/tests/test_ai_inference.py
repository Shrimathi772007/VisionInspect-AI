import pytest

from app.ai.evaluation import compute_reconstruction_error, compute_threshold
from app.ai.inference import (
    DEFECTIVE_PREDICTION,
    GOOD_PREDICTION,
    ModelArtifactNotFoundError,
    PredictionResult,
    predict_image,
)
from app.ai.preprocessing.errors import ImageDecodeError, ImageNotFoundError
from app.ai.training import build_model, discover_train_samples, save_model
from app.ai.training.artifacts import get_model_path
from tests.conftest import make_image_bytes

CATEGORY = "widget"
IMAGE_SIZE = (32, 32)


def _make_fake_train_only(dataset_root, category, good_count):
    """A category with ONLY train/good images - no test/ directory at all, so it is
    impossible for inference to accidentally use test images or labels."""
    good_dir = dataset_root / category / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(good_count):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=IMAGE_SIZE, color=(i * 5, 40, 90)))


@pytest.fixture
def fake_category(tmp_path, monkeypatch):
    """A saved model + a train/good-only dataset for a throwaway category, wired up the
    same way test_ai_evaluation.py does it: monkeypatch the artifact/dataset roots rather
    than touching the real bottle model or the real MVTec dataset."""
    dataset_root = tmp_path / "dataset"
    _make_fake_train_only(dataset_root, CATEGORY, good_count=6)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", dataset_root)

    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
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
    dataset_root = tmp_path / "dataset"
    _make_fake_train_only(dataset_root, CATEGORY, good_count=3)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", dataset_root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")  # empty - no saved model

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    with pytest.raises(ModelArtifactNotFoundError, match=CATEGORY):
        predict_image(image_path, CATEGORY, image_size=IMAGE_SIZE)


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
# Threshold integration (reuses the existing Phase 5 threshold logic)
# ---------------------------------------------------------------------------

def test_threshold_matches_existing_phase5_threshold_logic(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _make_fake_train_only(dataset_root, CATEGORY, good_count=6)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", dataset_root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")

    model = build_model()
    save_model(model, get_model_path(CATEGORY))

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(make_image_bytes("PNG", size=(32, 32)))

    result = predict_image(image_path, CATEGORY, image_size=IMAGE_SIZE)

    from app.ai.inference.predict import _image_to_tensor

    train_samples = discover_train_samples(CATEGORY)
    expected_errors = [compute_reconstruction_error(model, _image_to_tensor(s.path, IMAGE_SIZE)) for s in train_samples]
    expected_threshold = compute_threshold(expected_errors)

    assert result.threshold == pytest.approx(expected_threshold)


def test_threshold_is_independent_of_test_set(fake_category):
    """The fake category has no test/ directory at all - if inference ever touched test
    data or labels, this would raise before reaching a prediction."""
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction in (GOOD_PREDICTION, DEFECTIVE_PREDICTION)


# ---------------------------------------------------------------------------
# Prediction rule / boundary behavior (must match Phase 5 exactly)
# ---------------------------------------------------------------------------

def test_boundary_error_equal_to_threshold_is_good(fake_category, monkeypatch):
    monkeypatch.setattr("app.ai.inference.predict._get_threshold", lambda *a, **k: 0.05)
    monkeypatch.setattr("app.ai.inference.predict.compute_reconstruction_error", lambda *a, **k: 0.05)
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction == GOOD_PREDICTION


def test_boundary_error_just_above_threshold_is_defective(fake_category, monkeypatch):
    monkeypatch.setattr("app.ai.inference.predict._get_threshold", lambda *a, **k: 0.05)
    monkeypatch.setattr("app.ai.inference.predict.compute_reconstruction_error", lambda *a, **k: 0.0501)
    result = predict_image(fake_category, CATEGORY, image_size=IMAGE_SIZE)
    assert result.prediction == DEFECTIVE_PREDICTION


def test_boundary_error_just_below_threshold_is_good(fake_category, monkeypatch):
    monkeypatch.setattr("app.ai.inference.predict._get_threshold", lambda *a, **k: 0.05)
    monkeypatch.setattr("app.ai.inference.predict.compute_reconstruction_error", lambda *a, **k: 0.0499)
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
