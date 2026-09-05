"""Data structures produced by image-quality analysis."""

from dataclasses import dataclass


@dataclass
class ImageQualityResult:
    """Real, measured properties of a source image.

    All values are computed directly from the image's pixel data (or the
    file itself) - nothing here is an arbitrary or fabricated score.
    """

    width: int
    height: int
    channels: int
    file_size_bytes: int
    brightness: float  # mean grayscale pixel intensity, 0-255
    contrast: float  # standard deviation of grayscale pixel intensity, 0-255
    sharpness: float  # variance of the Laplacian (higher = sharper)
    is_blurry: bool  # sharpness below BLUR_VARIANCE_THRESHOLD
    processing_time_ms: float  # wall-clock time to load + preprocess the image
