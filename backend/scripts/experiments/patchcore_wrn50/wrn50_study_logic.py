"""Pure logic of the WRN-50-2 PatchCore normal-data study (see PROTOCOL.md and STUDY_DECLARATION.md).

Everything here is deterministic and testable without images or weights: the train/good-only dataset path
guard and access audit, fold assignment, the synthetic-defect plan, the memory-bounded fold coreset, the
selection-table metrics and the selection rule. The runner (run_wrn50_study.py) only wires these together
with the backbone and the filesystem.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from app.ai.evaluation.category_phase2 import policy_threshold
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.storage import DATASET_ROOT

SCOPE = ("bottle", "capsule", "carpet", "grid", "hazelnut", "pill", "screw", "wood", "zipper")
RUN_ORDER = ("bottle", "capsule", "zipper", "wood", "grid", "pill", "carpet", "screw", "hazelnut")
MODES = ("crop224", "full256")  # tie-break order: crop224 first
AGGREGATIONS = ("max", "top1pct_mean")  # tie-break order: max first
POLICIES = ("mean_std_2.5", "mean_std_3", "percentile_99")  # last-resort tie-break order
N_FOLDS = 5
FPR_LIMIT = 0.10
FPR_EPSILON = 1e-12  # float guard for "<= 10%" only; no tolerance is applied to recall or FPR ties
DEFECTS_PER_IMAGE = 4
CORESET_RATIO = 0.10
CORESET_SEED = 0
PROJECTION_DIM = 64
FORBIDDEN_PARTS = ("test", "ground_truth")


# ---------------------------------------------------------------------------
# Dataset path guard + access audit (train/good ONLY)
# ---------------------------------------------------------------------------

class DatasetAccessError(Exception):
    """A dataset path outside dataset/<category>/train/good was requested."""


def train_good_path(category: str, filename: str | None = None, dataset_root: Path = DATASET_ROOT) -> Path:
    """THE only way the study builds dataset paths: dataset/<category>/train/good[/<filename>].

    Raises DatasetAccessError for an unknown category, a filename with any path structure, any path part
    named "test" or "ground_truth" (case-insensitive), or a resolved path that escapes train/good.
    """
    if category not in MVTEC_CATEGORIES:
        raise DatasetAccessError(f"Unknown category {category!r}.")
    parts = [category, "train", "good"]
    if filename is not None:
        if not filename or any(sep in filename for sep in ("/", "\\")) or filename in (".", ".."):
            raise DatasetAccessError(f"Filename {filename!r} must be a plain file name.")
        parts.append(filename)
    if any(part.lower() in FORBIDDEN_PARTS for part in parts):
        raise DatasetAccessError(f"Refusing a dataset path containing a forbidden part: {'/'.join(parts)}")
    base = (Path(dataset_root) / category / "train" / "good").resolve()
    path = (Path(dataset_root).joinpath(*parts)).resolve()
    if path != base and not path.is_relative_to(base):
        raise DatasetAccessError(f"Path escapes train/good: {'/'.join(parts)}")
    return path


def assert_train_good_relative(relative: str) -> None:
    """Audit-entry check: '<category>/train/good/<file>' with no forbidden part."""
    parts = relative.replace("\\", "/").split("/")
    if len(parts) < 3 or parts[1:3] != ["train", "good"] or any(p.lower() in FORBIDDEN_PARTS for p in parts):
        raise DatasetAccessError(f"Audit entry outside train/good: {relative}")


class AccessAudit:
    """Appends every dataset path the study lists or opens (relative to the dataset root) to a text file."""

    def __init__(self, audit_file: Path, dataset_root: Path = DATASET_ROOT):
        self.audit_file = Path(audit_file)
        self.dataset_root = Path(dataset_root).resolve()
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, path: Path) -> None:
        relative = Path(path).resolve().relative_to(self.dataset_root).as_posix()
        assert_train_good_relative(relative if action != "list" else relative + "/")
        with open(self.audit_file, "a", encoding="utf-8") as handle:
            handle.write(f"{action}\t{relative}\n")

    def verify(self) -> dict:
        lines = self.audit_file.read_text(encoding="utf-8").splitlines() if self.audit_file.is_file() else []
        for line in lines:
            action, relative = line.split("\t")
            assert_train_good_relative(relative if action != "list" else relative + "/")
        return {"entries": len(lines), "all_under_train_good": True,
                "opened": sum(1 for line in lines if line.startswith("open\t"))}


def list_train_good(category: str, audit: AccessAudit, dataset_root: Path = DATASET_ROOT) -> list[str]:
    """Sorted image file names in train/good (the only directory the study ever lists)."""
    directory = train_good_path(category, dataset_root=dataset_root)
    audit.record("list", directory)
    return sorted(p.name for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def file_list_fingerprint(category: str, names: list[str], dataset_root: Path = DATASET_ROOT) -> dict:
    """SHA-256 of the sorted file-name list, per-file SHA-256s, and a combined name:sha digest (train/good only)."""
    per_file = {}
    combined = hashlib.sha256()
    for name in names:
        digest = hashlib.sha256(train_good_path(category, name, dataset_root).read_bytes()).hexdigest()
        per_file[name] = digest
        combined.update(f"{name}:{digest}\n".encode())
    return {
        "file_list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "files_sha256": combined.hexdigest(),
        "per_file_sha256": per_file,
    }


# ---------------------------------------------------------------------------
# Folds and synthetic plan
# ---------------------------------------------------------------------------

def fold_of(index: int) -> int:
    """Fold of the image at `index` in the sorted train/good list."""
    return index % N_FOLDS


def fold_members(n: int, fold: int) -> list[int]:
    return [i for i in range(n) if fold_of(i) == fold]


def synthetic_plan(fold: int, position: int) -> list[dict]:
    """The 4 synthetic defects painted on the image at `position` (0-based) within `fold`'s sorted members.

    counter c = 4 * position + j (j = 0..3): defect type = DEFECT_TYPES[c mod 5], severity = SEVERITIES[c mod 3];
    seed = 1000 * fold + position (the generator additionally mixes the type and severity into its RNG).
    """
    plan = []
    for j in range(DEFECTS_PER_IMAGE):
        c = DEFECTS_PER_IMAGE * position + j
        plan.append({"defect_type": DEFECT_TYPES[c % len(DEFECT_TYPES)],
                     "severity": SEVERITIES[c % len(SEVERITIES)],
                     "seed": 1000 * fold + position})
    return plan


# ---------------------------------------------------------------------------
# Memory-bounded fold coreset (identical algorithm to patch_anomaly.greedy_coreset_indices)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_coreset_over_images(flat: torch.Tensor, image_indices: list[int], patches_per_image: int,
                               ratio: float = CORESET_RATIO, seed: int = CORESET_SEED,
                               projection_dim: int = PROJECTION_DIM) -> torch.Tensor:
    """Global row indices (into `flat`, the (N*P, D) feature view) of the greedy k-center coreset of the rows
    belonging to `image_indices`, WITHOUT materialising those rows: the seeded random projection is applied
    image by image into a small (rows, projection_dim) buffer. Same RNG use, size rounding and greedy update
    as app.ai.models.patch_anomaly.greedy_coreset_indices / PatchCoreDetector.fit_from_features.
    """
    rows = len(image_indices) * patches_per_image
    n_select = min(max(1, int(round(rows * ratio))), rows)
    dim = flat.shape[1]
    generator = torch.Generator().manual_seed(seed)
    if dim > projection_dim:
        projection = torch.randn(dim, projection_dim, generator=generator) / math.sqrt(projection_dim)
        z = torch.empty((rows, projection_dim), dtype=flat.dtype)
    else:
        projection = None
        z = torch.empty((rows, dim), dtype=flat.dtype)
    global_rows = torch.empty(rows, dtype=torch.long)
    for slot, image in enumerate(image_indices):
        start = image * patches_per_image
        block = flat[start : start + patches_per_image]
        z[slot * patches_per_image : (slot + 1) * patches_per_image] = block @ projection if projection is not None else block
        global_rows[slot * patches_per_image : (slot + 1) * patches_per_image] = torch.arange(start, start + patches_per_image)
    z_sq = (z * z).sum(1)
    first = int(torch.randint(rows, (1,), generator=generator))
    selected = [first]
    min_dist = (z_sq - 2 * (z @ z[first]) + z_sq[first]).clamp_(min=0)
    for _ in range(n_select - 1):
        index = int(torch.argmax(min_dist))
        selected.append(index)
        min_dist = torch.minimum(min_dist, (z_sq - 2 * (z @ z[index]) + z_sq[index]).clamp_(min=0))
    return global_rows[torch.tensor(selected, dtype=torch.long)]


# ---------------------------------------------------------------------------
# Metrics, selection table, selection rule
# ---------------------------------------------------------------------------

def combination_metrics(oof_scores, folds, synthetic_scores, policy: str) -> dict:
    """Threshold from the pooled out-of-fold normal scores; per-fold FPR at that pooled threshold (score >
    threshold = flagged); worst/mean fold FPR; synthetic recall pooled over all folds. A synthetic sample whose
    generation failed is passed as None and counts as NOT detected."""
    scores = np.asarray(oof_scores, dtype=np.float64)
    folds = np.asarray(folds)
    threshold = policy_threshold(policy, scores)
    fold_fpr = [float(np.mean(scores[folds == k] > threshold)) for k in range(N_FOLDS)]
    detected = [s is not None and float(s) > threshold for s in synthetic_scores]
    return {
        "threshold": threshold,
        "fold_fpr": fold_fpr,
        "worst_fold_fpr": max(fold_fpr),
        "mean_fold_fpr": float(np.mean(fold_fpr)),
        "pooled_fpr": float(np.mean(scores > threshold)),
        "synthetic_recall": float(np.mean(detected)) if detected else 0.0,
        "synthetic_count": len(detected),
        "synthetic_failed": sum(1 for s in synthetic_scores if s is None),
    }


def build_selection_table(mode_results: dict) -> list[dict]:
    """12 rows: MODES x AGGREGATIONS x POLICIES. `mode_results[mode][agg]` = {"oof", "folds", "synthetic",
    "synthetic_types"} (synthetic_types aligned with synthetic, for the per-type breakdown)."""
    rows = []
    for mode in MODES:
        for agg in AGGREGATIONS:
            data = mode_results[mode][agg]
            for policy in POLICIES:
                metrics = combination_metrics(data["oof"], data["folds"], data["synthetic"], policy)
                by_type = {}
                for defect_type in DEFECT_TYPES:
                    picked = [s for s, t in zip(data["synthetic"], data["synthetic_types"]) if t == defect_type]
                    by_type[defect_type] = float(np.mean([s is not None and s > metrics["threshold"] for s in picked])) if picked else None
                rows.append({"mode": mode, "aggregation": agg, "policy": policy, **metrics,
                             "synthetic_recall_by_type": by_type,
                             "meets_fpr_limit": metrics["worst_fold_fpr"] <= FPR_LIMIT + FPR_EPSILON})
    return rows


def _ranks(row: dict) -> tuple[int, int, int]:
    return MODES.index(row["mode"]), AGGREGATIONS.index(row["aggregation"]), POLICIES.index(row["policy"])


def select(rows: list[dict]) -> dict:
    """PROTOCOL selection rule with the declared tie-break order (exact comparisons, no tolerance):

    eligible (worst-fold FPR <= 10%): highest synthetic recall -> lower worst-fold FPR -> crop224 -> max
        -> lower mean fold FPR -> higher threshold -> policy order (mean_std_2.5, mean_std_3, percentile_99)
    none eligible: lowest worst-fold FPR -> highest synthetic recall -> crop224 -> max
        -> lower mean fold FPR -> higher threshold -> policy order
    """
    eligible = [r for r in rows if r["meets_fpr_limit"]]
    if eligible:
        def key(r):
            mode, agg, policy = _ranks(r)
            return (-r["synthetic_recall"], r["worst_fold_fpr"], mode, agg, r["mean_fold_fpr"], -r["threshold"], policy)
        chosen = min(eligible, key=key)
        reason = (f"{len(eligible)} of {len(rows)} combinations meet worst-fold FPR <= {FPR_LIMIT:.0%}; "
                  "highest synthetic recall among them, then the declared tie-breaks")
    else:
        def key(r):
            mode, agg, policy = _ranks(r)
            return (r["worst_fold_fpr"], -r["synthetic_recall"], mode, agg, r["mean_fold_fpr"], -r["threshold"], policy)
        chosen = min(rows, key=key)
        reason = f"no combination meets worst-fold FPR <= {FPR_LIMIT:.0%}; lowest worst-fold FPR, then the declared tie-breaks"
    return {"mode": chosen["mode"], "aggregation": chosen["aggregation"], "policy": chosen["policy"],
            "threshold": chosen["threshold"], "worst_fold_fpr": chosen["worst_fold_fpr"],
            "mean_fold_fpr": chosen["mean_fold_fpr"], "synthetic_recall": chosen["synthetic_recall"],
            "any_combination_met_fpr_limit": bool(eligible), "eligible_count": len(eligible), "reason": reason}


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_digest(lock: dict) -> str:
    """Canonical content hash of a lock, excluding run-environment fields (wall clock, run time, peak RAM, the
    audit counters, which grow when a run resumes) and the digest itself - same idea as the model-family study's
    lock_digest. Two runs that lock the same content get the same digest."""
    volatile = ("timestamp_utc", "timestamp_unix", "lock_digest", "run_seconds", "peak_ram_mb", "dataset_access_audit")
    body = {k: v for k, v in lock.items() if k not in volatile}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
