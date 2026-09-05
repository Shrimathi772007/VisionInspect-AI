import hashlib

import numpy as np
import pytest

from app.ai.preprocessing import (
    ImageDecodeError,
    ImageNotFoundError,
    UnsupportedImageFormatError,
    preprocess_image,
    process_image,
)
from app.ai.analytics.quality import BLUR_VARIANCE_THRESHOLD
from app.inspections.storage import DATASET_ROOT
from tests.conftest import make_image_bytes

# Real MVTec AD image, resolved via the project's actual dataset configuration
# (VisionInspect-AI/dataset/bottle/... - no mvtec_anomaly_detection wrapper).
MVTEC_GOOD_IMAGE = DATASET_ROOT / "bottle" / "test" / "good" / "000.png"


def _write_image(tmp_path, name="sample.png", fmt="PNG", size=(40, 30), color=(200, 30, 10)):
    path = tmp_path / name
    path.write_bytes(make_image_bytes(fmt=fmt, size=size, color=color))
    return path


# ---------------------------------------------------------------------------
# Valid images
# ---------------------------------------------------------------------------

def test_process_real_mvtec_image():
    assert MVTEC_GOOD_IMAGE.is_file(), "expected MVTec dataset image not found; is DATASET_ROOT correct?"

    result = process_image(MVTEC_GOOD_IMAGE)

    assert result.source_path == MVTEC_GOOD_IMAGE
    assert result.preprocessing.original_width > 0
    assert result.preprocessing.original_height > 0
    assert result.quality.width == result.preprocessing.original_width
    assert result.quality.height == result.preprocessing.original_height
    assert result.quality.file_size_bytes == MVTEC_GOOD_IMAGE.stat().st_size


def test_process_uploaded_style_image(tmp_path):
    path = _write_image(tmp_path, size=(64, 48))

    result = process_image(path)

    assert result.preprocessing.original_width == 64
    assert result.preprocessing.original_height == 48
    assert result.quality.width == 64
    assert result.quality.height == 48


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageNotFoundError):
        preprocess_image(tmp_path / "does_not_exist.png")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image")
    with pytest.raises(UnsupportedImageFormatError):
        preprocess_image(path)


def test_corrupted_image_raises(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not actually a png file")
    with pytest.raises(ImageDecodeError):
        preprocess_image(path)


# ---------------------------------------------------------------------------
# Preprocessing output
# ---------------------------------------------------------------------------

def test_preprocessing_resizes_to_target_dimensions(tmp_path):
    path = _write_image(tmp_path, size=(500, 300))

    result = preprocess_image(path, target_size=(224, 224))

    assert result.resized_image.shape == (224, 224, 3)
    assert result.normalized_image.shape == (224, 224, 3)
    assert result.target_width == 224
    assert result.target_height == 224
    # original dimensions are recorded but not what was resized
    assert result.original_width == 500
    assert result.original_height == 300


def test_preprocessing_custom_target_size(tmp_path):
    path = _write_image(tmp_path)

    result = preprocess_image(path, target_size=(64, 64))

    assert result.resized_image.shape == (64, 64, 3)


def test_color_conversion_to_rgb(tmp_path):
    # PIL saves this as an exact RGB (200, 30, 10) flat image.
    path = _write_image(tmp_path, size=(20, 20), color=(200, 30, 10))

    result = preprocess_image(path, target_size=(20, 20))

    assert result.color_space == "RGB"
    # A flat-color image survives resize unchanged, so every pixel should
    # still be exactly the RGB color PIL wrote - proving the BGR->RGB
    # conversion (not a channel-swapped BGR result) was applied.
    pixel = result.resized_image[10, 10]
    assert tuple(int(c) for c in pixel) == (200, 30, 10)


def test_normalized_output_is_scaled_uint8_to_float(tmp_path):
    path = _write_image(tmp_path, size=(32, 32))

    result = preprocess_image(path, target_size=(32, 32))

    assert result.normalized_image.dtype == np.float32
    assert result.normalized_image.min() >= 0.0
    assert result.normalized_image.max() <= 1.0
    np.testing.assert_allclose(
        result.normalized_image, result.resized_image.astype(np.float32) / 255.0
    )


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------

def test_quality_metrics_on_flat_color_image(tmp_path):
    # A perfectly flat image has zero contrast and effectively no edges
    # (near-zero Laplacian variance), so it should be flagged as blurry.
    path = _write_image(tmp_path, size=(50, 50), color=(120, 120, 120))

    result = process_image(path)
    quality = result.quality

    assert quality.width == 50
    assert quality.height == 50
    assert quality.channels == 3
    assert quality.file_size_bytes == path.stat().st_size
    assert quality.contrast == pytest.approx(0.0, abs=1e-6)
    assert quality.sharpness < BLUR_VARIANCE_THRESHOLD
    assert quality.is_blurry is True
    assert quality.processing_time_ms >= 0.0


def test_quality_metrics_on_real_mvtec_image_are_sane():
    result = process_image(MVTEC_GOOD_IMAGE)
    quality = result.quality

    assert 0.0 <= quality.brightness <= 255.0
    assert quality.contrast >= 0.0
    assert quality.sharpness >= 0.0
    assert isinstance(quality.is_blurry, bool)
    assert quality.processing_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Original image safety
# ---------------------------------------------------------------------------

def test_original_file_is_not_modified(tmp_path):
    path = _write_image(tmp_path, size=(40, 40))
    original_bytes = path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    process_image(path, target_size=(224, 224))

    assert path.read_bytes() == original_bytes
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_real_mvtec_image_is_not_modified():
    original_hash = hashlib.sha256(MVTEC_GOOD_IMAGE.read_bytes()).hexdigest()

    process_image(MVTEC_GOOD_IMAGE)

    assert hashlib.sha256(MVTEC_GOOD_IMAGE.read_bytes()).hexdigest() == original_hash
