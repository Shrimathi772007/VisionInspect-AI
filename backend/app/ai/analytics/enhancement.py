"""Image enhancement PREVIEW (denoise + contrast) and before/after image-quality metrics.

PREVIEW ONLY. THE AI MODELS DO NOT USE THE ENHANCED IMAGE: every prediction, score, localization and
heatmap is computed from the original upload/dataset file exactly as before. Nothing here is stored; the
preview is recomputed per request from the original file.

    original (BGR) -> downscale so the long side is at most 640 px (cv2.INTER_AREA; never upscaled)
                   -> denoise: cv2.fastNlMeansDenoisingColored(h=5, hColor=5, templateWindowSize=7,
                      searchWindowSize=21)
                   -> contrast: CLAHE (clipLimit 2.0, 8x8 tiles) on the L channel of LAB
                   -> enhanced preview (BGR)

Metrics (grayscale, measured on the downscaled original "before" and on the enhanced preview "after", so
both are at the same resolution and comparable):
    brightness  mean intensity                      (app.ai.analytics.quality.basic_metrics)
    contrast    intensity standard deviation        (same)
    sharpness   variance of the Laplacian           (same)
    noise       robust noise sigma: 1.4826 * MAD(Laplacian) / sqrt(20), the median absolute deviation of
                the 3x3 Laplacian response (kernel [[0,1,0],[1,-4,1],[0,1,0]], whose squared weights sum to
                20) scaled to a Gaussian-noise standard deviation. Edges inflate it a little; it is an
                estimate, not a measurement of the sensor noise.
All are deterministic for the same input.
"""

import math

import cv2
import numpy as np

from app.ai.analytics.quality import basic_metrics

PREVIEW_MAX_SIDE = 640
DENOISE_H = 5
DENOISE_H_COLOR = 5
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
_MAD_TO_SIGMA = 1.4826
_LAPLACIAN_NORM = math.sqrt(20.0)


def downscale(image_bgr: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """The image with its long side at most `max_side` px (aspect kept; never upscaled)."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image_bgr.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Denoise + CLAHE contrast enhancement of a BGR uint8 image (downscaled first). Preview only."""
    small = downscale(np.ascontiguousarray(image_bgr, dtype=np.uint8))
    denoised = cv2.fastNlMeansDenoisingColored(
        small, None, DENOISE_H, DENOISE_H_COLOR, DENOISE_TEMPLATE_WINDOW, DENOISE_SEARCH_WINDOW
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def noise_estimate(gray: np.ndarray) -> float:
    """Robust noise sigma from the median absolute deviation of the 3x3 Laplacian (see module docstring)."""
    response = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=1)
    mad = float(np.median(np.abs(response - np.median(response))))
    return _MAD_TO_SIGMA * mad / _LAPLACIAN_NORM


def image_metrics(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    brightness, contrast, sharpness = basic_metrics(gray)
    return {
        "sharpness": round(sharpness, 4),
        "contrast": round(contrast, 4),
        "noise": round(noise_estimate(gray), 4),
        "brightness": round(brightness, 4),
    }


def quality_report(image_bgr: np.ndarray) -> dict:
    """Before/after metrics of the enhancement preview, plus resolutions. Preview only."""
    before = downscale(image_bgr)
    after = enhance(image_bgr)
    height, width = image_bgr.shape[:2]
    small_height, small_width = before.shape[:2]
    return {
        "preview_only": True,
        "note": "Preview only. The AI models do not use the enhanced image.",
        "resolution": {"width": width, "height": height},
        "analysed_resolution": {"width": small_width, "height": small_height},
        "before": image_metrics(before),
        "after": image_metrics(after),
    }


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buffer.tobytes()
