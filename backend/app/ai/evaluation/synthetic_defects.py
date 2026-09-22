"""SYNTHETIC defects for model DEVELOPMENT only - never a substitute for real MVTec test performance.

MVTec AD ships real defective images only in its test split, and the protocol forbids using those for
model selection. To compare models/resolutions/scoring rules without touching them, this module paints
controlled artificial defects onto GOOD images (validation good images, at full resolution, BEFORE any
downsampling, so resolution effects are real).

What this is and is not:
  * Generic, parameterised generators (soft-edged colour blob, dark blob, thin line, wavy coloured strand,
    small grey object patch) at three severities. They are informed by the *names* of MVTec's Carpet defect
    types (color / hole / cut / thread / metal_contamination) but were never fitted to any real defective
    image, and use no test image, label or statistic.
  * Deterministic: every image is a pure function of (source image, defect type, severity, seed).
  * Numbers derived from them are labelled "synthetic" everywhere. They can rank candidates plausibly but
    real defects differ, so they can neither prove nor promise real-world recall.
"""

import math

import cv2
import numpy as np

DEFECT_TYPES = ("color_blob", "dark_blob", "line", "wavy_line", "object_patch")
SEVERITIES = (0, 1, 2)
SEVERITY_NAMES = {0: "small", 1: "medium", 2: "large"}

# Blob / patch areas as a fraction of the image; line lengths and widths relative to the image side.
_BLOB_AREA = {"color_blob": (0.003, 0.012, 0.04), "dark_blob": (0.002, 0.008, 0.03)}
_OBJECT_AREA = (0.0006, 0.0025, 0.008)
_LINE_LENGTH = (0.12, 0.25, 0.45)
_LINE_WIDTH = (0.003, 0.006, 0.010)


def _rng(seed: int, defect_type: str, severity: int) -> np.random.Generator:
    return np.random.default_rng([seed, DEFECT_TYPES.index(defect_type), severity])


def _odd(value: float, minimum: int = 3) -> int:
    k = max(minimum, int(value))
    return k if k % 2 == 1 else k + 1


def _blob_alpha(h: int, w: int, area_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Soft irregular blob (union of 3 jittered ellipses), float32 alpha in [0, 1]."""
    radius = math.sqrt(area_fraction * h * w / math.pi)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    mask = np.zeros((h, w), np.uint8)
    for _ in range(3):
        center = (int(cx + rng.uniform(-0.5, 0.5) * radius), int(cy + rng.uniform(-0.5, 0.5) * radius))
        axes = (max(1, int(radius * rng.uniform(0.6, 1.2))), max(1, int(radius * rng.uniform(0.6, 1.2))))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    k = _odd(radius * 0.35)
    return cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0


def _stroke_alpha(h: int, w: int, points: np.ndarray, width: int) -> np.ndarray:
    """Soft-edged stroke, at least 2 px wide, with its alpha peak normalised to 1 so a thin line stays visible
    at any image size (a raw blur of a 1-px line would peak below 0.5)."""
    width = max(2, width)
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [points.astype(np.int32).reshape(-1, 1, 2)], False, 255, thickness=width)
    k = _odd(width * 0.6)
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    return alpha / max(float(alpha.max()), 1e-6)


def _blend(image: np.ndarray, colored: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(image.astype(np.float32) * (1 - a) + colored.astype(np.float32) * a, 0, 255).astype(np.uint8)


def make_synthetic_defect(
    image_bgr: np.ndarray, defect_type: str, severity: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(defective BGR uint8 image, boolean mask of the painted region). Deterministic; never modifies the input."""
    if defect_type not in DEFECT_TYPES or severity not in SEVERITIES:
        raise ValueError(f"Unknown synthetic defect ({defect_type!r}, severity {severity!r}).")
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    rng = _rng(seed, defect_type, severity)

    if defect_type == "color_blob":
        alpha = _blob_alpha(h, w, _BLOB_AREA[defect_type][severity], rng)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.choice([-1, 1]) * rng.uniform(25, 80)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1.2, 2.0) + 30, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.75, 1.25), 0, 255)
        colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif defect_type == "dark_blob":  # torn / hole-like dark region
        alpha = _blob_alpha(h, w, _BLOB_AREA[defect_type][severity], rng)
        colored = image_bgr.astype(np.float32) * rng.uniform(0.12, 0.35) + rng.normal(0, 4, image_bgr.shape)

    elif defect_type == "line":  # cut-like thin straight line, dark or light
        length = _LINE_LENGTH[severity] * side
        angle = rng.uniform(0, math.pi)
        p0 = np.array([rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h])
        d = np.array([math.cos(angle), math.sin(angle)]) * length / 2
        alpha = _stroke_alpha(h, w, np.stack([p0 - d, p0 + d]), int(_LINE_WIDTH[severity] * side))
        tone = rng.choice([25, 215])
        colored = np.full(image_bgr.shape, tone, np.float32)

    elif defect_type == "wavy_line":  # thread-like wavy coloured strand
        length = _LINE_LENGTH[severity] * side
        angle = rng.uniform(0, math.pi)
        p0 = np.array([rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h])
        direction = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        t = np.linspace(-0.5, 0.5, 40)
        amplitude, cycles = 0.02 * side * rng.uniform(0.7, 1.3), rng.uniform(1.5, 3.5)
        points = p0 + np.outer(t * length, direction) + np.outer(amplitude * np.sin(2 * math.pi * cycles * t), normal)
        alpha = _stroke_alpha(h, w, points, max(2, int(_LINE_WIDTH[severity] * side * 0.8)))
        hsv = np.zeros(image_bgr.shape, np.uint8)
        hsv[..., 0], hsv[..., 1], hsv[..., 2] = int(rng.integers(0, 180)), 200, int(rng.integers(150, 230))
        colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    else:  # object_patch: small grey metallic-looking fragments
        alpha = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            alpha = np.maximum(alpha, _blob_alpha(h, w, _OBJECT_AREA[severity] / 2, rng))
        grey = rng.uniform(100, 190)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gradient = 1 + 0.25 * np.sin((xx + yy) / (side * 0.02) + rng.uniform(0, 6))
        colored = np.repeat((grey * gradient + rng.normal(0, 8, (h, w)))[..., None], 3, axis=2)

    out = _blend(image_bgr, colored, alpha)
    mask = alpha > 0.5
    if not mask.any() or np.array_equal(out, image_bgr):
        raise RuntimeError(f"Synthetic defect ({defect_type}, {severity}) produced no visible change.")
    return out, mask
