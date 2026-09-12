"""MVTec AD ground-truth defect mask evidence, for severity scoring (Milestone 3 Phase 2).

Real, measured evidence about the actual size of a labeled defect - never fabricated -
read directly from the MVTec AD dataset's own ground-truth segmentation masks (the
convention MVTec ships for every non-"good" test image, one binary mask per defect
image). This module only reads existing dataset files; it never writes to DATASET_ROOT,
modifies the dataset, or invents a value when a mask is missing.

Deliberately separate from app.inspections.severity, which stays pure and framework/DB-
independent (see its module docstring) - all Pillow/filesystem I/O for reading masks
lives here instead, one layer below the scoring engine, mirroring how app.dataset.service
already separates dataset filesystem access from anything that consumes it.

MVTec ground-truth convention - verified against every file currently in dataset/bottle/
as part of the Milestone 3 Phase 2 correction review (not assumed):

    dataset/<category>/ground_truth/<defect_type>/<stem>_mask.png
        - one 8-bit grayscale PNG per defective test image, exactly matching the pixel
          dimensions of dataset/<category>/test/<defect_type>/<stem>.png
        - binary-valued (0 = background/non-defect, 255 = defect pixel) in every mask
          inspected
        - present for every one of the 63 non-"good" Bottle test images (20 broken_large,
          22 broken_small, 21 contamination) with no exceptions - a reliable 1:1 mapping,
          not a partial/best-effort one

    "good" images never have a mask - there is no defect to segment, which is itself real
    ground truth (zero defect area), not missing evidence.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.inspections.storage import DATASET_ROOT, resolve_image_path

logger = logging.getLogger(__name__)

GOOD_DEFECT_TYPE = "good"


def _mask_relative_path(category: str, defect_type: str, filename: str) -> str:
    stem = Path(filename).stem
    return f"{category}/ground_truth/{defect_type}/{stem}_mask.png"


def compute_defect_area_ratio(category: str, defect_type: str, filename: str) -> Optional[float]:
    """The fraction of pixels marked as defect in this image's ground-truth mask.

    Returns 0.0 for `defect_type == "good"` without looking for a mask file - MVTec never
    ships one for non-defective images, and that absence IS the ground truth (no defect
    present), not missing evidence.

    For any other defect_type, returns the real measured ratio (defect pixels / total
    pixels) when a matching mask file exists under DATASET_ROOT for `category`, or None if
    it does not (unknown category/defect_type, a non-MVTec image, or any read failure) -
    never a guessed value. Never raises: a missing or unreadable mask is reported as
    "no evidence", not an error, since severity assessment must never fail inspection
    creation.
    """
    if defect_type == GOOD_DEFECT_TYPE:
        return 0.0

    try:
        mask_path = resolve_image_path(DATASET_ROOT, _mask_relative_path(category, defect_type, filename))
    except HTTPException:
        return None

    if not mask_path.is_file():
        return None

    try:
        with Image.open(mask_path) as mask_image:
            mask_array = np.array(mask_image.convert("L"))
    except Exception:  # noqa: BLE001 - an unreadable mask must never block severity assessment
        logger.warning("Could not read MVTec ground-truth mask at %s", mask_path)
        return None

    if mask_array.size == 0:
        return None

    return float(np.count_nonzero(mask_array)) / float(mask_array.size)
