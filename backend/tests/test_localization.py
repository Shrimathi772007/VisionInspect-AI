"""Anomaly-based defect localization, margin-based confidence, the manual-review rule and the quality
matrix for AI-only inspections - pure unit tests on synthetic arrays (no model, no database, no dataset).

The geometry tests push a synthetic hot spot at known original-image coordinates through the project's
REAL preprocessing for each input mode (crop224 / full256 / full320 / ResNet-18 resize), use the
preprocessed brightness as the anomaly map, and check the box maps back onto the hot spot.
"""

import math
from itertools import product

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.ai.inference import localization as L
from app.ai.inference.serving import GATE_NOT_PRODUCTION_READY, SERVING_CONFIGS, full320_module
from app.ai.models.patchcore import PatchCoreConfig, PatchCoreDetector
from app.ai.preprocessing import process_image
from app.ai.preprocessing.patchcore_preprocess import resize_and_crop
from app.inspections.quality import FAIL, MANUAL_REVIEW, NOT_ASSESSED, PASS, assess_quality

NPR_CATEGORIES = {"wood", "carpet", "screw", "pill"}


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_confidence_is_one_half_at_the_threshold():
    assert L.margin_confidence(1.7, 1.7) == 0.5


def test_confidence_at_one_and_a_quarter_times_threshold_is_about_0_90():
    assert L.margin_confidence(1.25 * 2.0, 2.0) == pytest.approx(1 / (1 + math.exp(-10 * math.log(1.25))))
    assert L.margin_confidence(1.25 * 2.0, 2.0) == pytest.approx(0.903, abs=0.001)


@pytest.mark.parametrize("ratio", [1.01, 1.1, 1.25, 2.0, 10.0])
def test_confidence_is_symmetric_in_the_log_ratio(ratio):
    threshold = 1.8
    assert L.margin_confidence(threshold * ratio, threshold) == pytest.approx(L.margin_confidence(threshold / ratio, threshold))


def test_confidence_is_monotonic_in_the_margin_and_stays_in_half_open_unit_interval():
    threshold = 2.0
    above = [L.margin_confidence(threshold * r, threshold) for r in np.linspace(1.0, 3.0, 41)]
    below = [L.margin_confidence(threshold / r, threshold) for r in np.linspace(1.0, 3.0, 41)]
    for series in (above, below):
        assert all(b > a for a, b in zip(series, series[1:]))
        assert all(0.5 <= c < 1.0 for c in series)


@pytest.mark.parametrize("score, threshold", [
    (None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0), (float("nan"), 1.0), (1.0, float("inf")),
])
def test_confidence_is_none_when_undefined(score, threshold):
    assert L.margin_confidence(score, threshold) is None


@pytest.mark.parametrize("confidence, level", [
    (None, None), (0.5, "low"), (0.6999, "low"), (0.70, "medium"), (0.9499, "medium"), (0.95, "high"), (0.999, "high"),
])
def test_reliability_bands(confidence, level):
    assert L.reliability_level(confidence) == level


# ---------------------------------------------------------------------------
# Manual-review rule
# ---------------------------------------------------------------------------

def test_exactly_the_four_declared_categories_are_not_production_ready():
    assert {c for c, cfg in SERVING_CONFIGS.items() if cfg.gate == GATE_NOT_PRODUCTION_READY} == NPR_CATEGORIES


@pytest.mark.parametrize("category", sorted(SERVING_CONFIGS))
def test_review_truth_table_per_category(category):
    gate = SERVING_CONFIGS[category].gate
    npr = category in NPR_CATEGORIES
    for confidence in (0.5, 0.6999):
        decision = L.review_rule(confidence, gate)
        assert decision.required is True
        expected = "low confidence; category model not production ready" if npr else "low confidence"
        assert decision.reason == expected
    for confidence in (0.70, 0.9, 0.95, 0.999):
        decision = L.review_rule(confidence, gate)
        assert decision.required is npr
        assert decision.reason == ("category model not production ready" if npr else None)


def test_missing_confidence_counts_as_low():
    assert L.review_rule(None, "EXCELLENT") == L.ReviewDecision(True, "low confidence")


# ---------------------------------------------------------------------------
# Region rule (anomaly_map_threshold_v1), full-image region, exact boxes
# ---------------------------------------------------------------------------

def _blank(side=100, value=0.0):
    return np.full((side, side), value, dtype=np.float32)


def test_single_blob_exact_box_area_and_centroid():
    amap = _blank()
    amap[20:30, 40:60] = 2.0
    loc = L.localize(amap, threshold=1.0)
    assert loc["method"] == "anomaly_map_threshold_v1"
    assert loc["mask_rule"] == "threshold"
    assert loc["boxes"] == [{"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.1, "peak": 2.0, "area_fraction": 0.02}]
    assert loc["area_pct"] == pytest.approx(2.0)
    assert loc["centroid"] == pytest.approx([0.5, 0.25])


def test_two_blobs_ordered_by_peak_and_centroid_of_the_largest():
    amap = _blank()
    amap[20:30, 40:60] = 2.0  # 200 px, peak 2
    amap[60:70, 10:15] = 3.0  # 50 px, peak 3
    loc = L.localize(amap, threshold=1.0)
    assert [b["peak"] for b in loc["boxes"]] == [3.0, 2.0]
    assert loc["boxes"][0] == {"x": 0.1, "y": 0.6, "w": 0.05, "h": 0.1, "peak": 3.0, "area_fraction": 0.005}
    assert loc["area_pct"] == pytest.approx(2.5)
    assert loc["centroid"] == pytest.approx([0.5, 0.25])  # the 200 px blob


def test_component_below_a_quarter_percent_of_the_image_is_discarded():
    amap = _blank()
    amap[10:14, 10:15] = 5.0  # 20 px = 0.20% -> discarded
    amap[50:56, 50:55] = 2.0  # 30 px = 0.30% -> kept
    loc = L.localize(amap, threshold=1.0)
    assert len(loc["boxes"]) == 1
    assert loc["boxes"][0]["x"] == 0.5 and loc["boxes"][0]["peak"] == 2.0
    assert loc["area_pct"] == pytest.approx(0.3)


def test_only_tiny_components_gives_no_boxes():
    amap = _blank()
    amap[10:14, 10:15] = 5.0
    loc = L.localize(amap, threshold=1.0)
    assert loc["boxes"] == [] and loc["area_pct"] == 0.0 and loc["centroid"] is None


def test_empty_mask_falls_back_to_ninety_percent_of_the_peak():
    amap = _blank(value=0.1)
    amap[30:40, 30:50] = 0.8   # peak 0.8 < threshold 1.0; 0.9 * 0.8 = 0.72
    amap[70:80, 70:80] = 0.7   # below 0.72 -> not part of the fallback mask
    loc = L.localize(amap, threshold=1.0)
    assert loc["mask_rule"] == "relative_peak"
    assert len(loc["boxes"]) == 1
    assert loc["boxes"][0] == pytest.approx({"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.1, "peak": 0.8, "area_fraction": 0.02})


def test_six_blobs_keep_only_the_five_highest_peaks():
    amap = _blank()
    for i in range(6):
        amap[5 + 15 * i : 12 + 15 * i, 5:12] = 1.5 + i  # 49 px each, peaks 1.5 .. 6.5
    loc = L.localize(amap, threshold=1.0)
    assert [b["peak"] for b in loc["boxes"]] == [6.5, 5.5, 4.5, 3.5, 2.5]
    assert loc["area_pct"] == pytest.approx(5 * 0.49)


def test_good_prediction_localization_is_empty():
    assert L.empty_localization() == {
        "method": "anomaly_map_threshold_v1", "boxes": [], "area_pct": 0.0, "centroid": None, "mask_rule": None,
    }


def test_all_zero_map_gives_no_boxes():
    loc = L.localize(_blank(), threshold=1.0)
    assert loc["boxes"] == [] and loc["area_pct"] == 0.0


# ---------------------------------------------------------------------------
# Anomaly map: the PatchCoreDetector.anomaly_map helper, reused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid_side, out_side", [(28, 224), (32, 256), (40, 320), (16, 128)])
def test_anomaly_map_is_patchcore_anomaly_map(grid_side, out_side):
    generator = torch.Generator().manual_seed(grid_side)
    scores = torch.rand(1, grid_side * grid_side, generator=generator) * 3
    detector = PatchCoreDetector(PatchCoreConfig(), extractor=torch.nn.Identity())
    expected = detector.anomaly_map(scores, out_side)[0].numpy()
    actual = L.anomaly_map(scores.reshape(grid_side, grid_side), out_side)
    assert actual.shape == (out_side, out_side)
    assert np.array_equal(actual, expected)
    assert actual.max() <= scores.max().item() + 1e-6  # convex combinations never exceed the largest patch score


# ---------------------------------------------------------------------------
# Geometry through the real preprocessing, square and non-square
# ---------------------------------------------------------------------------

def _hot_spot_image(width, height, box):
    """BGR uint8 image, black except a white rectangle at normalised `box` (x0, y0, x1, y1)."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0, x1, y1 = box
    image[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = 255
    return image


def _model_view(image_bgr, mode, tmp_path):
    """(S, S) float map in [0, 1] of what the model sees, through the project's own preprocessing."""
    if mode in ("crop224", "full256"):
        rgb = resize_and_crop(image_bgr, mode)
    elif mode == "full320":
        rgb = full320_module().resize_full320(image_bgr)  # the locked Grid Addendum 1 resize itself
    else:  # ResNet-18 "resize": app.ai.preprocessing.process_image at 256x256
        path = tmp_path / "hot.png"
        cv2.imwrite(str(path), image_bgr)
        rgb = process_image(path, target_size=(256, 256)).preprocessing.resized_image
    return np.asarray(rgb, dtype=np.float32).mean(axis=2) / 255.0


GEOMETRY_CASES = [
    # mode, width, height, hot spot (x0, y0, x1, y1) in the original image - inside the analysed area
    ("crop224", 400, 400, (0.30, 0.20, 0.45, 0.35)),
    ("crop224", 400, 300, (0.60, 0.55, 0.75, 0.80)),   # landscape: crop removes ~17% left/right, ~6% top/bottom
    ("crop224", 300, 480, (0.20, 0.40, 0.50, 0.55)),   # portrait
    ("full256", 400, 400, (0.05, 0.70, 0.25, 0.95)),
    ("full256", 640, 360, (0.80, 0.10, 0.95, 0.40)),
    ("full320", 512, 512, (0.40, 0.40, 0.60, 0.50)),
    ("full320", 300, 500, (0.10, 0.05, 0.40, 0.20)),
    ("resize", 512, 512, (0.65, 0.15, 0.85, 0.30)),
    ("resize", 600, 400, (0.02, 0.60, 0.20, 0.90)),
]


@pytest.mark.parametrize("mode, width, height, spot", GEOMETRY_CASES)
def test_box_maps_back_to_the_hot_spot_in_original_coordinates(mode, width, height, spot, tmp_path):
    amap = _model_view(_hot_spot_image(width, height, spot), mode, tmp_path)
    region = L.map_region(mode, width, height)
    loc = L.localize(amap, threshold=0.5, region=region)
    assert len(loc["boxes"]) == 1
    box = loc["boxes"][0]
    side = amap.shape[0]
    tol_x = 2.0 / side * (region[2] - region[0]) + 1.0 / width
    tol_y = 2.0 / side * (region[3] - region[1]) + 1.0 / height
    assert box["x"] == pytest.approx(spot[0], abs=tol_x)
    assert box["y"] == pytest.approx(spot[1], abs=tol_y)
    assert box["x"] + box["w"] == pytest.approx(spot[2], abs=tol_x)
    assert box["y"] + box["h"] == pytest.approx(spot[3], abs=tol_y)
    assert loc["centroid"] == pytest.approx([(spot[0] + spot[2]) / 2, (spot[1] + spot[3]) / 2], abs=max(tol_x, tol_y))


def test_crop224_square_image_covers_the_central_87_5_percent():
    assert L.map_region("crop224", 900, 900) == (0.0625, 0.0625, 0.9375, 0.9375)


def test_crop224_non_square_region_matches_resize_and_crop():
    # 400x300: shorter side 300 -> 256, width round(341.33) = 341; left = (341 - 224) // 2 = 58, top = 16.
    assert L.map_region("crop224", 400, 300) == pytest.approx((58 / 341, 16 / 256, 282 / 341, 240 / 256))


@pytest.mark.parametrize("mode", ["full256", "full320", "resize"])
def test_whole_image_modes_cover_everything(mode):
    assert L.map_region(mode, 640, 360) == (0.0, 0.0, 1.0, 1.0)


def test_hot_spot_outside_the_crop_is_not_seen_by_crop224(tmp_path):
    amap = _model_view(_hot_spot_image(400, 300, (0.0, 0.3, 0.1, 0.5)), "crop224", tmp_path)  # left 10%: cropped away
    assert L.localize(amap, threshold=0.5, region=L.map_region("crop224", 400, 300))["boxes"] == []


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width, height, expected", [(1024, 1024, (320, 320)), (640, 360, (320, 180)), (200, 100, (200, 100)), (300, 900, (107, 320))])
def test_heatmap_size_keeps_the_aspect_ratio_and_caps_the_long_side(width, height, expected):
    assert L.heatmap_size(width, height) == expected


def test_heatmap_alpha_is_zero_outside_the_crop_and_rises_to_180_at_the_threshold():
    side = 224
    amap = np.zeros((side, side), dtype=np.float32)
    amap[100:124, 100:124] = 2.0   # 2x threshold
    amap[0:20, 0:20] = 0.4         # below half the threshold
    region = L.map_region("crop224", 640, 480)
    rgba = L.render_heatmap(amap, 1.0, 640, 480, region)
    assert rgba.shape == (240, 320, 4) and rgba.dtype == np.uint8
    alpha = rgba[..., 3]
    x0, y0, x1, y1 = region
    left, top, right, bottom = round(x0 * 320), round(y0 * 240), round(x1 * 320), round(y1 * 240)
    assert alpha[:, :left].max() == 0 and alpha[:, right:].max() == 0
    assert alpha[:top, :].max() == 0 and alpha[bottom:, :].max() == 0
    assert alpha.max() == 180
    centre_x, centre_y = left + round(112 / 224 * (right - left)), top + round(112 / 224 * (bottom - top))
    assert alpha[centre_y, centre_x] == 180
    assert alpha[top + 2, left + 2] == 0   # 0.4 x threshold -> transparent


def test_heatmap_png_round_trips_as_rgba(tmp_path):
    rgba = L.render_heatmap(np.full((32, 32), 1.5, dtype=np.float32), 1.0, 100, 50)
    path = tmp_path / "h.png"
    L.write_png_atomic(rgba, path)
    with Image.open(path) as image:
        assert image.mode == "RGBA" and image.size == (100, 50)
    assert [p.name for p in tmp_path.iterdir()] == ["h.png"]  # no temporary file left behind


# ---------------------------------------------------------------------------
# Quality decision matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ai_prediction, review_required, expected", [
    (None, None, NOT_ASSESSED),
    (None, True, NOT_ASSESSED),
    ("good", True, MANUAL_REVIEW),
    ("defective", True, MANUAL_REVIEW),
    ("defective", False, FAIL),
    ("good", False, PASS),
    ("good", None, PASS),
])
def test_ai_only_quality_matrix(ai_prediction, review_required, expected):
    result = assess_quality(status="pending", ai_prediction=ai_prediction, defect_category=None,
                            severity_level=None, review_required=review_required)
    assert result.decision == expected


def test_manual_review_wording():
    result = assess_quality("pending", "good", None, None, review_required=True)
    assert result.assessment == "AI result is low-reliability for this category/confidence; manual inspection required."
    assert result.recommendation


def test_high_severity_still_fails_an_ai_only_inspection_first():
    assert assess_quality("pending", "good", None, "High", review_required=True).decision == FAIL


def _pre_task_decision(status, ai_prediction, severity_level):
    """The decision rules for inspections WITH ground truth, frozen as they were before this task."""
    if severity_level in ("Critical", "High"):
        return FAIL
    if ai_prediction in ("good", "defective") and status != ai_prediction:
        return NOT_ASSESSED
    if status == "defective" or ai_prediction == "defective":
        return FAIL
    return PASS  # status == "good"


@pytest.mark.parametrize("status, ai_prediction, severity_level, defect_category", list(product(
    ("good", "defective"), (None, "good", "defective"), (None, "Medium", "High"), (None, "good", "broken_large"),
)))
def test_imports_with_ground_truth_keep_their_decisions_whatever_the_review_flag(status, ai_prediction, severity_level, defect_category):
    baseline = assess_quality(status, ai_prediction, defect_category, severity_level)
    assert baseline.decision == _pre_task_decision(status, ai_prediction, severity_level)
    for review_required in (None, False, True):
        result = assess_quality(status, ai_prediction, defect_category, severity_level, review_required=review_required)
        assert result == baseline
