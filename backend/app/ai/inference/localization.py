"""Anomaly-based defect localization, margin-based confidence and the manual-review rule.

    per-patch anomaly scores (the grid the image score was aggregated from)
    -> anomaly map: bilinear upsample to the model's input size + Gaussian blur (sigma 4, 33 taps) - the
       PatchCoreDetector.anomaly_map helper, reused for both patch families
    -> region rule "anomaly_map_threshold_v1" (defective predictions only): mask = map >= the model's own
       decision threshold (fallback: map >= 0.9 * map.max() when smoothing pulled the whole map under it),
       8-connected components, components under 0.25% of the image area discarded, at most 5 kept by peak
    -> bounding boxes, area_pct and centroid in ORIGINAL-IMAGE normalised coordinates (0..1, origin top-left)
    -> RGBA heatmap PNG in original-image geometry (alpha 0 below half the threshold and outside the area
       the model saw, rising to 180 at/above the threshold)

This is anomaly-map localization: the boxes are derived from where the anomaly score is high. It is NOT an
object detector (no YOLO or any trained box regressor) and it does not classify defect types.

All rules here were declared before any use and are never tuned on test-set results. Pure computation on
numpy/torch/OpenCV/Pillow values - no database, no model loading, no dataset access - so every function is
unit-testable on synthetic maps. Does not change any image score, threshold or prediction.
"""

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.ai.models.patchcore import PatchCoreDetector

LOCALIZATION_METHOD = "anomaly_map_threshold_v1"
FALLBACK_PEAK_FRACTION = 0.9
MIN_COMPONENT_AREA_FRACTION = 0.0025  # 0.25% of the original image area
MAX_BOXES = 5

# Geometry of the model input relative to the original image.
MODE_CROP224 = "crop224"
MODE_FULL256 = "full256"
MODE_FULL320 = "full320"
MODE_RESIZE = "resize"  # ResNet-18 patch models: whole image resized to the input size
FULL_IMAGE_MODES = (MODE_FULL256, MODE_FULL320, MODE_RESIZE)
CROP224_SIZE = 224
CROP224_RESIZE_SHORTER_SIDE = 256

# Heatmap PNG.
HEATMAP_MAX_SIDE = 320
HEATMAP_MAX_ALPHA = 180
HEATMAP_ALPHA_START = 0.5  # alpha is 0 at or below 0.5 x threshold, rising linearly to the maximum at the threshold
HEATMAP_COLOUR_TOP = 1.25  # JET colour scale spans 0.5 x .. 1.25 x threshold (red at/above 1.25 x threshold)

# Confidence / review.
CONFIDENCE_SLOPE = 10.0
HIGH_RELIABILITY = "high"
MEDIUM_RELIABILITY = "medium"
LOW_RELIABILITY = "low"
HIGH_CONFIDENCE_MIN = 0.95
REVIEW_CONFIDENCE_BELOW = 0.70
REVIEW_REASON_LOW_CONFIDENCE = "low confidence"
REVIEW_REASON_NOT_PRODUCTION_READY = "category model not production ready"
GATE_NOT_PRODUCTION_READY = "NOT_PRODUCTION_READY"


# ---------------------------------------------------------------------------
# Confidence and the manual-review rule
# ---------------------------------------------------------------------------

def margin_confidence(score: float | None, threshold: float | None) -> float | None:
    """sigmoid(10 * |ln(score / threshold)|), in [0.5, 1).

    A margin-based heuristic: how far (on a log scale) the image score sits from the model's own decision
    threshold, in either direction. It is NOT a calibrated probability of the prediction being right. 0.5 at
    score == threshold, ~0.90 at score = 1.25 x threshold (or threshold / 1.25), symmetric in the log ratio and
    monotonic in its magnitude. None when the score or threshold is missing, not finite, or <= 0 (the log ratio
    is undefined).
    """
    if score is None or threshold is None:
        return None
    score, threshold = float(score), float(threshold)
    if not (math.isfinite(score) and math.isfinite(threshold)) or score <= 0 or threshold <= 0:
        return None
    margin = abs(math.log(score / threshold))
    return 1.0 / (1.0 + math.exp(-CONFIDENCE_SLOPE * margin))


def reliability_level(confidence: float | None) -> str | None:
    """Display band: >= 0.95 "high", 0.70 to < 0.95 "medium", < 0.70 "low" (manual review); None stays None."""
    if confidence is None:
        return None
    if confidence >= HIGH_CONFIDENCE_MIN:
        return HIGH_RELIABILITY
    if confidence >= REVIEW_CONFIDENCE_BELOW:
        return MEDIUM_RELIABILITY
    return LOW_RELIABILITY


@dataclass(frozen=True)
class ReviewDecision:
    required: bool
    reason: str | None


def review_rule(confidence: float | None, gate: str | None) -> ReviewDecision:
    """Manual review when confidence < 0.70 OR the category's model gate is NOT_PRODUCTION_READY.

    A missing confidence (score or threshold <= 0, see margin_confidence) counts as low: without a margin the
    result cannot be called reliable. Reasons are joined by "; " in a fixed order.
    """
    reasons = []
    if confidence is None or confidence < REVIEW_CONFIDENCE_BELOW:
        reasons.append(REVIEW_REASON_LOW_CONFIDENCE)
    if gate == GATE_NOT_PRODUCTION_READY:
        reasons.append(REVIEW_REASON_NOT_PRODUCTION_READY)
    return ReviewDecision(required=bool(reasons), reason="; ".join(reasons) if reasons else None)


# ---------------------------------------------------------------------------
# Anomaly map and geometry
# ---------------------------------------------------------------------------

def anomaly_map(patch_scores, out_side: int) -> np.ndarray:
    """(h, w) square patch-score grid -> (out_side, out_side) float32 map: bilinear upsample + Gaussian blur
    (sigma 4, 33 taps, reflect padding).

    Reuses PatchCoreDetector.anomaly_map itself: with an explicit out_size it never reads the detector
    instance (only the default size comes from self.config), so the WRN-50 and ResNet-18 maps go through the
    very same code."""
    grid = patch_scores if torch.is_tensor(patch_scores) else torch.from_numpy(np.asarray(patch_scores, dtype=np.float32))
    grid = grid.detach().float().reshape(1, -1)
    return PatchCoreDetector.anomaly_map(None, grid, int(out_side))[0].cpu().numpy()


def map_region(mode: str, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1): the part of the original image the model input (and so the anomaly map) covers, in
    normalised original-image coordinates.

    crop224 mirrors app.ai.preprocessing.patchcore_preprocess.resize_and_crop: shorter side resized to 256,
    then the centre 224x224 is cropped - for a square image that is the central 87.5% on each axis. Every other
    mode resizes the whole image, so the map covers all of it."""
    if mode in FULL_IMAGE_MODES:
        return 0.0, 0.0, 1.0, 1.0
    if mode != MODE_CROP224:
        raise ValueError(f"Unknown localization input mode '{mode}'.")
    scale = CROP224_RESIZE_SHORTER_SIDE / min(image_height, image_width)
    new_width = max(CROP224_SIZE, round(image_width * scale))
    new_height = max(CROP224_SIZE, round(image_height * scale))
    top, left = (new_height - CROP224_SIZE) // 2, (new_width - CROP224_SIZE) // 2
    return left / new_width, top / new_height, (left + CROP224_SIZE) / new_width, (top + CROP224_SIZE) / new_height


# ---------------------------------------------------------------------------
# Region rule
# ---------------------------------------------------------------------------

def _r(value: float) -> float:
    return round(float(value), 6)


def empty_localization() -> dict:
    """The localization of a "good" prediction: no boxes, nothing covered."""
    return {"method": LOCALIZATION_METHOD, "boxes": [], "area_pct": 0.0, "centroid": None, "mask_rule": None}


def localize(amap: np.ndarray, threshold: float, region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> dict:
    """Region rule anomaly_map_threshold_v1 on one (S, S) anomaly map whose area is `region` of the original
    image. Call it for "defective" predictions only (a "good" prediction gets empty_localization()).

    mask = map >= threshold; if empty, map >= 0.9 * map.max() ("mask_rule": "threshold" / "relative_peak").
    8-connected components; components covering < 0.25% of the ORIGINAL image area are discarded; at most 5
    are kept, by descending peak. Each box: x, y, w, h (normalised, original image), peak (map value) and
    area_fraction (of the original image). area_pct = 100 x the summed area fractions of the kept components;
    centroid = (cx, cy) of the largest kept component (None when nothing is kept)."""
    amap = np.asarray(amap, dtype=np.float32)
    height, width = amap.shape
    x0, y0, x1, y1 = region
    region_w, region_h = x1 - x0, y1 - y0
    pixel_area = (region_w / width) * (region_h / height)  # original-image area fraction of one map pixel

    mask = amap >= threshold
    mask_rule = "threshold"
    if not mask.any():
        peak = float(amap.max())
        if not math.isfinite(peak) or peak <= 0:
            return {**empty_localization(), "mask_rule": "relative_peak"}
        mask = amap >= FALLBACK_PEAK_FRACTION * peak
        mask_rule = "relative_peak"

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, count):
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = pixels * pixel_area
        if area_fraction < MIN_COMPONENT_AREA_FRACTION:
            continue
        components.append((float(amap[labels == label].max()), pixels, label))
    components.sort(key=lambda c: (-c[0], -c[1], c[2]))
    kept = components[:MAX_BOXES]

    boxes = []
    for peak, pixels, label in kept:
        left, top = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        box_w, box_h = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append({
            "x": _r(x0 + left / width * region_w),
            "y": _r(y0 + top / height * region_h),
            "w": _r(box_w / width * region_w),
            "h": _r(box_h / height * region_h),
            "peak": _r(peak),
            "area_fraction": _r(pixels * pixel_area),
        })

    centroid = None
    if kept:
        _peak, _pixels, largest = max(kept, key=lambda c: (c[1], c[0]))
        cx, cy = centroids[largest]
        centroid = [_r(x0 + (cx + 0.5) / width * region_w), _r(y0 + (cy + 0.5) / height * region_h)]

    return {
        "method": LOCALIZATION_METHOD,
        "boxes": boxes,
        "area_pct": _r(100.0 * sum(pixels * pixel_area for _peak, pixels, _label in kept)),
        "centroid": centroid,
        "mask_rule": mask_rule,
    }


# ---------------------------------------------------------------------------
# Heatmap PNG
# ---------------------------------------------------------------------------

def heatmap_size(image_width: int, image_height: int, max_side: int = HEATMAP_MAX_SIDE) -> tuple[int, int]:
    """(width, height) of the heatmap: the original aspect ratio, at most `max_side` on the long side."""
    scale = min(1.0, max_side / max(image_width, image_height))
    return max(1, round(image_width * scale)), max(1, round(image_height * scale))


def render_heatmap(amap: np.ndarray, threshold: float, image_width: int, image_height: int,
                   region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """RGBA uint8 (H, W, 4) heatmap in ORIGINAL-IMAGE geometry (scaled to <= 320 px on the long side), ready to
    be stretched over the original image. JET colours over 0.5x..1.25x the threshold; alpha 0 at or below 0.5x
    the threshold and everywhere outside `region` (the area the model did not see), rising linearly to 180 at
    and above the threshold."""
    out_w, out_h = heatmap_size(image_width, image_height)
    ratio = np.full((out_h, out_w), np.nan, dtype=np.float32)
    x0, y0, x1, y1 = region
    left, top = round(x0 * out_w), round(y0 * out_h)
    right, bottom = max(left + 1, round(x1 * out_w)), max(top + 1, round(y1 * out_h))
    resized = cv2.resize(np.asarray(amap, dtype=np.float32), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
    ratio[top:bottom, left:right] = resized / float(threshold)

    inside = np.isfinite(ratio)
    r = np.where(inside, ratio, 0.0)
    colour_t = np.clip((r - HEATMAP_ALPHA_START) / (HEATMAP_COLOUR_TOP - HEATMAP_ALPHA_START), 0.0, 1.0)
    bgr = cv2.applyColorMap((colour_t * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip((r - HEATMAP_ALPHA_START) / (1.0 - HEATMAP_ALPHA_START), 0.0, 1.0) * HEATMAP_MAX_ALPHA
    alpha = np.where(inside, np.round(alpha), 0).astype(np.uint8)
    return np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha])


def write_png_atomic(rgba: np.ndarray, path: Path) -> None:
    """Write an RGBA PNG to `path` via a temporary file in the same directory + os.replace, so a reader never
    sees a half-written file. The temporary file is removed on failure."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=".heatmap-", suffix=".png", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8)).save(handle, format="PNG")  # (H, W, 4) -> RGBA
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
