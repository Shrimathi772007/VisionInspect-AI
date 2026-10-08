"""Localization evaluation addendum (app/ai/box_eval_addendum.json): box AP@0.5 and pixel AUROC per category.

Produced by scripts/experiments/localization_map/ (PROTOCOL.md there): a second scoring of the final test sets for
localization only; no model or threshold changed; anomaly-map boxes, not a trained detector. Display metadata only -
no model, threshold, decision or quality rule reads it. Loaded once at import; a missing or unreadable file, a
missing category or an out-of-range value leaves that value None.
"""

import json
import math
from pathlib import Path

BOX_EVAL_ADDENDUM_PATH = Path(__file__).resolve().parent / "box_eval_addendum.json"
METRIC_KEYS = ("box_ap50", "box_ap50_merged", "pixel_auroc")


def _unit_interval(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def load_box_eval_addendum(path: Path = BOX_EVAL_ADDENDUM_PATH) -> dict[str, dict[str, float | None]]:
    """category -> {box_ap50, box_ap50_merged, pixel_auroc}; {} when the file is missing or unreadable."""
    try:
        categories = json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
        items = categories.items()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
    return {category: {key: _unit_interval(entry.get(key)) for key in METRIC_KEYS}
            for category, entry in items if isinstance(entry, dict)}


_BOX_EVAL = load_box_eval_addendum()


def box_eval_for(category: str) -> dict[str, float | None]:
    return _BOX_EVAL.get(category) or dict.fromkeys(METRIC_KEYS)
