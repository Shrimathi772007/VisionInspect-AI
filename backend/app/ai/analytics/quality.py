"""Image-quality analysis: brightness, contrast, sharpness and basic stats.

Every metric is a standard, explainable computer-vision measurement:
- brightness: mean grayscale pixel intensity
- contrast:   standard deviation of grayscale pixel intensity
- sharpness:  variance of the Laplacian (Pech-Pacheco et al.'s well-known
              blur-detection heuristic - higher variance means more
              high-frequency edge content, i.e. a sharper image)

No score here is invented; each is a direct, reproducible calculation over
the image's own pixels.
"""

from pathlib import Path

import cv2
import numpy as np

from app.ai.analytics.schemas import ImageQualityResult

# Below this Laplacian-variance, an image is flagged as blurry. This is the
# commonly cited default threshold for this heuristic; it is a documented
# cut-off on a real measurement, not a fabricated quality score.
BLUR_VARIANCE_THRESHOLD = 100.0


def analyze_quality(path: Path, image_bgr: np.ndarray, processing_time_ms: float) -> ImageQualityResult:
    """Compute real quality metrics from an already-loaded BGR image.

    `image_bgr` should be the untouched, original-resolution image as
    decoded by OpenCV (see preprocessing.pipeline._load_image) - metrics are
    measured on the real image, not a resized/normalized copy.
    """
    height, width = image_bgr.shape[:2]
    channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if channels > 1 else image_bgr

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return ImageQualityResult(
        width=width,
        height=height,
        channels=channels,
        file_size_bytes=path.stat().st_size,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        is_blurry=sharpness < BLUR_VARIANCE_THRESHOLD,
        processing_time_ms=processing_time_ms,
    )
