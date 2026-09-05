"""Reusable image preprocessing pipeline: IMAGE -> PREPROCESSING -> QUALITY ANALYSIS -> RESULT.

Works on any resolved filesystem path, so it is equally usable for MVTec AD
dataset images and user-uploaded inspection images - both are already
resolved to safe, on-disk paths by the existing project code
(app.inspections.storage.resolve_image_path / app.dataset.service) before
reaching this pipeline. This module does not re-implement that path/traversal
validation; it only re-checks the basics (existence, extension, decodability)
that are meaningful for any path it is handed.

Framework-independent: no FastAPI imports, no HTTP concerns. Errors are
plain exceptions from app.ai.preprocessing.errors.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from app.ai.analytics.quality import analyze_quality
from app.ai.preprocessing.errors import ImageDecodeError, ImageNotFoundError, UnsupportedImageFormatError
from app.ai.preprocessing.schemas import ImageProcessingResult, PreprocessingResult
from app.inspections.storage import ALLOWED_EXTENSIONS

# Standard square input size used by common CNN backbones; a sensible,
# consistent default for future model use.
DEFAULT_TARGET_SIZE = (224, 224)  # (width, height)


def _load_image(path: Path) -> np.ndarray:
    """Validate and decode an image file, returning it untouched as BGR uint8.

    Raises ImageNotFoundError, UnsupportedImageFormatError, or ImageDecodeError.
    Never writes to `path` - this only reads it.
    """
    path = Path(path)

    if not path.is_file():
        raise ImageNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"Unsupported image format '{path.suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ImageDecodeError(f"File could not be decoded as an image: {path}")

    return image_bgr


def preprocess_image(path: Path, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE) -> PreprocessingResult:
    """Load, validate, color-convert, resize and normalize one image.

    Returns a PreprocessingResult describing both the original image
    (dimensions/channels, untouched on disk) and the model-ready copy.
    """
    path = Path(path)
    image_bgr = _load_image(path)
    return _build_preprocessing_result(path, image_bgr, target_size)


def _build_preprocessing_result(path: Path, image_bgr: np.ndarray, target_size: tuple[int, int]) -> PreprocessingResult:
    original_height, original_width = image_bgr.shape[:2]
    original_channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    target_width, target_height = target_size
    resized = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    return PreprocessingResult(
        source_path=path,
        original_width=original_width,
        original_height=original_height,
        original_channels=original_channels,
        color_space="RGB",
        target_width=target_width,
        target_height=target_height,
        resized_image=resized,
        normalized_image=normalized,
    )


def process_image(path: Path, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE) -> ImageProcessingResult:
    """Run the full pipeline: load -> preprocess -> analyze quality -> combined result."""
    path = Path(path)

    start = time.perf_counter()
    image_bgr = _load_image(path)
    preprocessing = _build_preprocessing_result(path, image_bgr, target_size)
    elapsed_ms = (time.perf_counter() - start) * 1000

    quality = analyze_quality(path, image_bgr, processing_time_ms=elapsed_ms)

    return ImageProcessingResult(source_path=path, preprocessing=preprocessing, quality=quality)
