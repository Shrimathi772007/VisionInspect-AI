"""Image preprocessing pipeline (resize, normalize) and combined result.

Public API:
    preprocess_image(path, target_size=DEFAULT_TARGET_SIZE) -> PreprocessingResult
    process_image(path, target_size=DEFAULT_TARGET_SIZE) -> ImageProcessingResult
"""

from app.ai.preprocessing.errors import (
    ImageDecodeError,
    ImageNotFoundError,
    ImageProcessingError,
    UnsupportedImageFormatError,
)
from app.ai.preprocessing.pipeline import DEFAULT_TARGET_SIZE, preprocess_image, process_image
from app.ai.preprocessing.schemas import ImageProcessingResult, PreprocessingResult

__all__ = [
    "DEFAULT_TARGET_SIZE",
    "preprocess_image",
    "process_image",
    "PreprocessingResult",
    "ImageProcessingResult",
    "ImageProcessingError",
    "ImageNotFoundError",
    "UnsupportedImageFormatError",
    "ImageDecodeError",
]
