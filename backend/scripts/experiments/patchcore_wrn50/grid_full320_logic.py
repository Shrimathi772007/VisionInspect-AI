"""Pure logic of Addendum 1, the Grid full320 candidate (ADDENDUM_1_GRID_FULL320.md).

New code only; patchcore.py, patchcore_preprocess.py and the study modules are used read-only:
full320 preprocessing, a PatchCoreConfig subclass that accepts full320 (and its loader), the 6 full320 rows,
the combined 18-row selection rule, and the memory-light coreset helpers. Everything here is testable without
images or weights.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import torch
from torch import nn

import wrn50_study_logic as logic
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES
from app.ai.models.patchcore import (AGGREGATIONS, CONFIG_FILENAME, STATE_FILENAME, PatchCoreConfig, PatchCoreDetector,
                                     PatchCoreModelError)
from app.ai.models.resnet18 import file_sha256
from app.ai.models.wide_resnet50 import WRN50_2_SHA256

MODE_FULL320 = "full320"
FULL320_SIZE = 320
COMBINED_MODES = ("crop224", "full256", MODE_FULL320)  # tie-break order: existing modes first
CATEGORY = "grid"


# ---------------------------------------------------------------------------
# full320 preprocessing and config
# ---------------------------------------------------------------------------

def resize_full320(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB uint8 320x320: whole-image resize (cv2.INTER_AREA), no crop."""
    resized = cv2.resize(image_bgr, (FULL320_SIZE, FULL320_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(np.ascontiguousarray(resized), cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class Full320Config(PatchCoreConfig):
    """PatchCoreConfig for full320 only (patchcore.py accepts crop224/full256 and is not edited)."""

    preprocessing: str = MODE_FULL320

    def __post_init__(self):
        if self.preprocessing != MODE_FULL320:
            raise ValueError(f"Full320Config only accepts preprocessing '{MODE_FULL320}'.")
        if self.input_size == -1:
            object.__setattr__(self, "input_size", FULL320_SIZE)
        elif self.input_size != FULL320_SIZE:
            raise ValueError(f"input_size {self.input_size} does not match preprocessing '{MODE_FULL320}'.")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'. Expected one of {AGGREGATIONS}.")
        if self.k != 1:
            raise ValueError("Only k=1 nearest-neighbour scoring is supported.")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ValueError("coreset_ratio must be in (0, 1].")
        object.__setattr__(self, "layers", tuple(self.layers))


def load_full320_detector(directory: Path, extractor: nn.Module | None = None) -> PatchCoreDetector:
    """PatchCoreDetector.load for a saved full320 model: the same integrity checks (detector type, pinned
    backbone hash, state SHA-256, bank shape, weights_only load), but the config is rebuilt as Full320Config."""
    directory = Path(directory)
    state_path, config_path = directory / STATE_FILENAME, directory / CONFIG_FILENAME
    if not state_path.is_file() or not config_path.is_file():
        raise PatchCoreModelError(f"Incomplete PatchCore model in {directory.name}: missing state or config.")
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("detector") != "patchcore" or record.get("preprocessing") != MODE_FULL320:
        raise PatchCoreModelError("Not a full320 PatchCore model configuration.")
    if record.get("weights_sha256") != WRN50_2_SHA256:
        raise PatchCoreModelError(f"Model was built for backbone weights {record.get('weights_sha256')!r}.")
    if file_sha256(state_path) != record.get("state_sha256"):
        raise PatchCoreModelError("Model state file does not match its recorded SHA-256.")
    bank = torch.load(state_path, map_location="cpu", weights_only=True)["bank"]
    if list(bank.shape) != record.get("bank_shape"):
        raise PatchCoreModelError("Model state bank shape does not match the recorded shape.")
    return PatchCoreDetector(Full320Config.from_dict(record), extractor=extractor, bank=bank)


# ---------------------------------------------------------------------------
# Rows and the combined selection rule
# ---------------------------------------------------------------------------

def build_full320_rows(results: dict) -> list[dict]:
    """6 rows (AGGREGATIONS x POLICIES) for full320, built exactly like logic.build_selection_table's rows.
    `results[agg]` = {"oof", "folds", "synthetic", "synthetic_types"}."""
    rows = []
    for agg in logic.AGGREGATIONS:
        data = results[agg]
        for policy in logic.POLICIES:
            metrics = logic.combination_metrics(data["oof"], data["folds"], data["synthetic"], policy)
            by_type = {}
            for defect_type in DEFECT_TYPES:
                picked = [s for s, t in zip(data["synthetic"], data["synthetic_types"]) if t == defect_type]
                by_type[defect_type] = float(np.mean([s is not None and s > metrics["threshold"] for s in picked])) if picked else None
            rows.append({"mode": MODE_FULL320, "aggregation": agg, "policy": policy, **metrics,
                         "synthetic_recall_by_type": by_type,
                         "meets_fpr_limit": metrics["worst_fold_fpr"] <= logic.FPR_LIMIT + logic.FPR_EPSILON})
    return rows


def _ranks(row: dict) -> tuple[int, int, int]:
    return (COMBINED_MODES.index(row["mode"]), logic.AGGREGATIONS.index(row["aggregation"]),
            logic.POLICIES.index(row["policy"]))


def select_combined(rows: list[dict]) -> dict:
    """The study's selection rule over the pooled rows, with full320 last in the mode order:

    eligible (worst-fold FPR <= 10%): highest synthetic recall -> lower worst-fold FPR -> crop224 -> full256
        -> full320 -> max -> lower mean fold FPR -> higher threshold -> policy order
    none eligible: lowest worst-fold FPR -> highest synthetic recall -> then the same order
    Exact comparisons; an existing row wins every exact tie with a full320 row."""
    eligible = [r for r in rows if r["meets_fpr_limit"]]
    if eligible:
        def key(r):
            mode, agg, policy = _ranks(r)
            return (-r["synthetic_recall"], r["worst_fold_fpr"], mode, agg, r["mean_fold_fpr"], -r["threshold"], policy)
        chosen = min(eligible, key=key)
        reason = (f"{len(eligible)} of {len(rows)} combinations meet worst-fold FPR <= {logic.FPR_LIMIT:.0%}; "
                  "highest synthetic recall among them, then the declared tie-breaks (crop224, full256, full320, max, ...)")
    else:
        def key(r):
            mode, agg, policy = _ranks(r)
            return (r["worst_fold_fpr"], -r["synthetic_recall"], mode, agg, r["mean_fold_fpr"], -r["threshold"], policy)
        chosen = min(rows, key=key)
        reason = f"no combination meets worst-fold FPR <= {logic.FPR_LIMIT:.0%}; lowest worst-fold FPR, then the declared tie-breaks"
    return {"mode": chosen["mode"], "aggregation": chosen["aggregation"], "policy": chosen["policy"],
            "threshold": chosen["threshold"], "worst_fold_fpr": chosen["worst_fold_fpr"],
            "mean_fold_fpr": chosen["mean_fold_fpr"], "synthetic_recall": chosen["synthetic_recall"],
            "any_combination_met_fpr_limit": bool(eligible), "eligible_count": len(eligible), "reason": reason}


# ---------------------------------------------------------------------------
# Memory-light coreset helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def fold_bank_from_buffer(buffer: torch.Tensor, ratio: float = logic.CORESET_RATIO, seed: int = logic.CORESET_SEED,
                          projection_dim: int = logic.PROJECTION_DIM) -> torch.Tensor:
    """Bank from a (n_train, P, D) buffer holding ONLY the other folds' features in ascending image order:
    the study's greedy_coreset_over_images on the buffer, selected rows copied out (the buffer can then be
    released). Same rows, in the same order, as the all-in-memory method on the full feature tensor."""
    n_train, patches, dim = buffer.shape
    flat = buffer.view(n_train * patches, dim)
    rows = logic.greedy_coreset_over_images(flat, list(range(n_train)), patches, ratio, seed, projection_dim)
    return flat[rows].clone()


@torch.no_grad()
def streaming_coreset_select(blocks: Iterable[torch.Tensor], n_images: int, patches: int, dim: int,
                             ratio: float = logic.CORESET_RATIO, seed: int = logic.CORESET_SEED,
                             projection_dim: int = logic.PROJECTION_DIM) -> torch.Tensor:
    """Pass 1 of the streaming final fit: global row indices of the greedy coreset over ALL images, given each
    image's (P, D) features in order. Only the 64-d projection is kept. Same RNG use, size rounding, per-image
    projection and greedy update as logic.greedy_coreset_over_images(flat, range(n_images), patches)."""
    rows = n_images * patches
    n_select = min(max(1, int(round(rows * ratio))), rows)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(dim, projection_dim, generator=generator) / math.sqrt(projection_dim) if dim > projection_dim else None
    z = torch.empty((rows, projection_dim if projection is not None else dim), dtype=torch.float32)
    seen = 0
    for slot, block in enumerate(blocks):
        if slot >= n_images or block.shape != (patches, dim):
            raise ValueError(f"Unexpected block {slot} with shape {tuple(block.shape)}.")
        z[slot * patches : (slot + 1) * patches] = block @ projection if projection is not None else block
        seen += 1
    if seen != n_images:
        raise ValueError(f"Expected {n_images} image blocks, got {seen}.")
    z_sq = (z * z).sum(1)
    first = int(torch.randint(rows, (1,), generator=generator))
    selected = [first]
    min_dist = (z_sq - 2 * (z @ z[first]) + z_sq[first]).clamp_(min=0)
    for _ in range(n_select - 1):
        index = int(torch.argmax(min_dist))
        selected.append(index)
        min_dist = torch.minimum(min_dist, (z_sq - 2 * (z @ z[index]) + z_sq[index]).clamp_(min=0))
    return torch.tensor(selected, dtype=torch.long)


@torch.no_grad()
def gather_selected_rows(blocks: Iterable[torch.Tensor], selected: torch.Tensor, patches: int, dim: int) -> torch.Tensor:
    """Pass 2 of the streaming final fit: bank = the selected global rows, in selection order, read from the
    same per-image blocks again (equal to flat[selected] on the full feature tensor)."""
    bank = torch.empty((len(selected), dim), dtype=torch.float32)
    image_of = (selected // patches).tolist()
    offset = (selected % patches).tolist()
    by_image: dict[int, list[int]] = {}
    for position, image in enumerate(image_of):
        by_image.setdefault(image, []).append(position)
    for image, block in enumerate(blocks):
        for position in by_image.get(image, ()):
            bank[position] = block[offset[position]]
    return bank


def iter_image_features(detector: PatchCoreDetector, images: Iterable[torch.Tensor], batch_size: int = 8) -> Iterator[torch.Tensor]:
    """Per-image (P, D) features of preprocessed (3, S, S) images, extracted in consecutive batches of
    `batch_size` (the same batching as PatchCoreDetector.extract_features_into)."""
    side = detector.config.input_size
    batch = torch.empty((batch_size, 3, side, side), dtype=torch.float32)
    filled = 0
    for image in images:
        batch[filled] = image
        filled += 1
        if filled == batch_size:
            yield from detector.extract_features(batch[:filled])
            filled = 0
    if filled:
        yield from detector.extract_features(batch[:filled])
