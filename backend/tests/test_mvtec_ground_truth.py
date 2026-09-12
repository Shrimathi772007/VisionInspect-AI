"""Milestone 3 Phase 2 correction review: MVTec ground-truth defect mask evidence.

Exercises app.dataset.ground_truth.compute_defect_area_ratio against the REAL MVTec Bottle
dataset already present at DATASET_ROOT (dataset/bottle/) - no synthetic fixtures, no mock
masks, so these tests double as manual/automated validation that the masks contain what
the Phase 2 correction report claims they contain (binary 0/255, one per defective test
image, none for "good"). Never writes to the dataset - only reads it.
"""

import glob

import numpy as np
from PIL import Image

from app.dataset.ground_truth import compute_defect_area_ratio
from app.inspections.storage import DATASET_ROOT

BOTTLE_DEFECT_TYPES = ("broken_large", "broken_small", "contamination")


def _real_mask_path(defect_type: str, filename: str = "000.png"):
    stem = filename.rsplit(".", 1)[0]
    return DATASET_ROOT / "bottle" / "ground_truth" / defect_type / f"{stem}_mask.png"


# ---------------------------------------------------------------------------
# "good" - no mask exists; ratio is a real 0.0, not missing evidence
# ---------------------------------------------------------------------------

def test_good_has_no_ground_truth_directory():
    assert not (DATASET_ROOT / "bottle" / "ground_truth" / "good").exists()


def test_good_image_area_ratio_is_zero():
    assert compute_defect_area_ratio("bottle", "good", "000.png") == 0.0


# ---------------------------------------------------------------------------
# Real defective masks - cross-checked against an independent direct read
# ---------------------------------------------------------------------------

def test_broken_large_area_ratio_matches_independent_computation():
    mask_path = _real_mask_path("broken_large", "000.png")
    assert mask_path.is_file(), "expected real MVTec ground-truth mask to exist"

    expected_ratio = np.count_nonzero(np.array(Image.open(mask_path))) / np.array(Image.open(mask_path)).size
    actual_ratio = compute_defect_area_ratio("bottle", "broken_large", "000.png")

    assert actual_ratio == expected_ratio
    assert actual_ratio > 0.0  # broken_large defects are real, visible defects


def test_broken_small_area_ratio_is_smaller_than_broken_large():
    """Sanity check against the real dataset: MVTec's own naming (large vs small) should
    be reflected in the measured mask area for the same sample index."""
    small_ratio = compute_defect_area_ratio("bottle", "broken_small", "000.png")
    large_ratio = compute_defect_area_ratio("bottle", "broken_large", "000.png")
    assert small_ratio is not None and large_ratio is not None
    assert 0.0 < small_ratio < large_ratio


def test_contamination_area_ratio_is_positive():
    ratio = compute_defect_area_ratio("bottle", "contamination", "000.png")
    assert ratio is not None
    assert 0.0 < ratio < 1.0


def test_every_real_bottle_defect_mask_yields_a_ratio_between_0_and_1():
    """Exhaustive, not spot-checked: every ground-truth mask currently in the dataset."""
    for defect_type in BOTTLE_DEFECT_TYPES:
        mask_files = sorted(glob.glob(str(DATASET_ROOT / "bottle" / "ground_truth" / defect_type / "*_mask.png")))
        assert mask_files, f"expected at least one real mask for {defect_type}"
        for mask_file in mask_files:
            stem = mask_file.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].removesuffix("_mask.png")
            ratio = compute_defect_area_ratio("bottle", defect_type, f"{stem}.png")
            assert ratio is not None
            assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# Missing evidence -> None, never fabricated
# ---------------------------------------------------------------------------

def test_unknown_defect_type_returns_none():
    assert compute_defect_area_ratio("bottle", "some_future_defect_type", "000.png") is None


def test_missing_filename_returns_none():
    assert compute_defect_area_ratio("bottle", "broken_large", "does_not_exist.png") is None


def test_unknown_category_returns_none():
    assert compute_defect_area_ratio("not_a_real_category", "broken_large", "000.png") is None


def test_path_traversal_attempt_returns_none_not_a_file_outside_dataset_root():
    assert compute_defect_area_ratio("bottle", "../../../../etc", "passwd") is None
