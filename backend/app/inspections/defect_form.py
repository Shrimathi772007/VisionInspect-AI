"""Rule-based defect form ("defect_form_v1") from the anomaly-map localization of one inspection.

A SHAPE-BASED FORM DERIVED FROM THE ANOMALY REGION; NOT A TRAINED DEFECT-TYPE CLASSIFIER. It only says what
the flagged region looks like (several regions, a large area, a line, a small spot, a patch) - never what
kind of defect it is (scratch, crack, contamination...). It is never written into Inspection.defect_category,
which stays MVTec ground truth.

Inputs are the stored localization (app.ai.inference.localization, rule "anomaly_map_threshold_v1") and the
original image's pixel size. Declared rules, applied in this order (first match wins):

    >= 3 boxes                                   -> "multiple_regions"  (type score 75)
    area_pct >= 15                               -> "large_area"        (85)
    largest box pixel aspect ratio >= 3          -> "linear"            (60)
    area_pct < 2                                 -> "small_spot"        (30)
    otherwise                                    -> "localized_patch"   (50)

The aspect ratio is measured in PIXELS (box w * image width vs box h * image height), not in normalised
units, so a non-square image does not distort it. "Largest" means the largest box area.

Known limitation: the regions come from a Gaussian-blurred anomaly heatmap, so area_pct OVERESTIMATES the
true defect area (in a live check a diagonal bar covering 9.9% of a tile image gave a single 41% region) and
thin defects look wider than they are, which pushes forms towards "large_area" and away from "linear".
"""

from dataclasses import dataclass

DEFECT_FORM_VERSION = "defect_form_v1"

MULTIPLE_REGIONS = "multiple_regions"
LARGE_AREA = "large_area"
LINEAR = "linear"
SMALL_SPOT = "small_spot"
LOCALIZED_PATCH = "localized_patch"

DEFECT_FORM_SCORES = {
    MULTIPLE_REGIONS: 75,
    LARGE_AREA: 85,
    LINEAR: 60,
    SMALL_SPOT: 30,
    LOCALIZED_PATCH: 50,
}

DEFECT_FORM_LABELS = {
    MULTIPLE_REGIONS: "Multiple regions",
    LARGE_AREA: "Large area",
    LINEAR: "Linear",
    SMALL_SPOT: "Small spot",
    LOCALIZED_PATCH: "Localized patch",
}

MULTIPLE_REGIONS_MIN_BOXES = 3
LARGE_AREA_MIN_PCT = 15.0
LINEAR_MIN_ASPECT = 3.0
SMALL_SPOT_MAX_PCT = 2.0


@dataclass(frozen=True)
class DefectForm:
    form: str
    score: int
    largest_box_aspect: float


def _pixel_aspect(box: dict, width: int, height: int) -> float:
    pixel_w, pixel_h = float(box["w"]) * width, float(box["h"]) * height
    shorter = min(pixel_w, pixel_h)
    if shorter <= 0:
        return float("inf")
    return max(pixel_w, pixel_h) / shorter


def classify_defect_form(boxes: list[dict], area_pct: float, width: int, height: int) -> DefectForm | None:
    """defect_form_v1 for one AI-defective inspection; None when there are no boxes (nothing to describe)
    or the image size is unknown."""
    if not boxes or width <= 0 or height <= 0:
        return None
    largest = max(boxes, key=lambda b: float(b["w"]) * width * float(b["h"]) * height)
    aspect = _pixel_aspect(largest, width, height)
    if len(boxes) >= MULTIPLE_REGIONS_MIN_BOXES:
        form = MULTIPLE_REGIONS
    elif area_pct >= LARGE_AREA_MIN_PCT:
        form = LARGE_AREA
    elif aspect >= LINEAR_MIN_ASPECT:
        form = LINEAR
    elif area_pct < SMALL_SPOT_MAX_PCT:
        form = SMALL_SPOT
    else:
        form = LOCALIZED_PATCH
    return DefectForm(form=form, score=DEFECT_FORM_SCORES[form], largest_box_aspect=aspect)


def add_defect_form(localization: dict, width: int, height: int) -> dict:
    """Store defect_form / defect_form_score / defect_form_version in a localization dict (in place) when a
    form applies; returns the dict."""
    form = classify_defect_form(localization.get("boxes") or [], float(localization.get("area_pct") or 0.0), width, height)
    if form is not None:
        localization["defect_form"] = form.form
        localization["defect_form_score"] = form.score
        localization["defect_form_version"] = DEFECT_FORM_VERSION
    return localization


def defect_form_label(form: str | None) -> str | None:
    if not form:
        return None
    return DEFECT_FORM_LABELS.get(form, form.replace("_", " ").capitalize())
