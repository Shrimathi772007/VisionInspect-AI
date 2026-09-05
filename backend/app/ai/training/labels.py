"""Ground-truth label convention for MVTec AD anomaly detection.

good        -> 0
<anything else, e.g. broken_large, broken_small, contamination, ...> -> 1

This is intentionally generic (not a hard-coded per-category list of defect
names) so the same rule works for bottle, cable, capsule, and every other
MVTec category without changes.
"""

GOOD_DEFECT_TYPE = "good"
GOOD_LABEL = 0
DEFECTIVE_LABEL = 1


def label_for_defect_type(defect_type: str) -> int:
    return GOOD_LABEL if defect_type == GOOD_DEFECT_TYPE else DEFECTIVE_LABEL
