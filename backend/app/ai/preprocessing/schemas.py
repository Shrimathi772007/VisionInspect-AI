"""Data structures produced by the preprocessing pipeline.

Plain dataclasses (not pydantic) since these carry numpy arrays and are meant
for internal/programmatic use, not JSON serialization over the API yet.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ai.analytics.schemas import ImageQualityResult


@dataclass
class PreprocessingResult:
    """Output of running the resize/normalize pipeline on one image.

    `resized_image` and `normalized_image` are copies; the source file on
    disk is never read into a mutable buffer that gets written back, so the
    original image is left untouched.
    """

    source_path: Path
    original_width: int
    original_height: int
    original_channels: int
    color_space: str
    target_width: int
    target_height: int
    resized_image: np.ndarray  # uint8, shape (target_height, target_width, 3), RGB
    normalized_image: np.ndarray  # float32, same shape, values scaled to [0, 1]


@dataclass
class ImageProcessingResult:
    """Combined result of the full IMAGE -> PREPROCESSING -> QUALITY ANALYSIS pipeline."""

    source_path: Path
    preprocessing: PreprocessingResult
    quality: ImageQualityResult
