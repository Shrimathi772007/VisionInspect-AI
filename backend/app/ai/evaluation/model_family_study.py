"""Controlled alternative-model-family study for one category (built for Leather): pure computation on cached
patch features of the 245 train/good images. This module cannot list, decode or score a final-test image and never
imports test discovery (asserted from source by tests).

Reuses the project's anomaly-model framework: frozen ImageNet ResNet-18 features (app.ai.models.patch_anomaly),
GaussianPatchScorer / KNNPatchScorer / greedy_coreset_indices / aggregate_scores, the synthetic-defect generators
(app.ai.evaluation.synthetic_defects) and the threshold-policy arithmetic of Phase 2 (category_phase2.policy_threshold).

=====================================================================================================================
PRE-DECLARED REGISTRY AND SELECTION RULE  (written before any candidate was scored; not to be altered afterwards)
=====================================================================================================================
Candidates (exactly 8, never expanded): frozen ImageNet ResNet-18, no fine-tuning, features = 3x3 stride-1 average-
pooled maps of the chosen layers (layer 3 bilinearly aligned to layer 2 when both are used), ImageNet normalization
on top of the project's existing preprocessing (BGR->RGB, INTER_AREA resize, /255). Image score = MEAN OF THE TOP 1%
OF PATCH SCORES (fixed for every candidate, the project's original aggregation).
  Family A  gauss_l2_128, gauss_l3_128, gauss_l23_128, gauss_l23_256   one global Gaussian over all normal patches,
            Mahalanobis distance, covariance shrinkage 1e-2 (relative to the mean variance).
  Family B  knn_l2_128,   knn_l3_128,   knn_l23_128,   knn_l23_256     nearest-neighbour (k=1, Euclidean) distance
            to a memory bank of normal patches. The bank is the union of per-image greedy k-center coresets
            (10% of each normal image's patches, seed 42) - a documented, deterministic PatchCore-style
            subsampling chosen so that any fold's bank is exactly the union of its images' coresets.
Threshold policies (exactly 8): mean + K*std (population std), K in {1.5, 2.0, 2.5, 3.0}; percentiles P95/P97/P98/P99
(linear interpolation). A CONFIGURATION is a (candidate, policy) pair: 64 in total.

Normal-only evaluation: 30 deterministic splits of the 245 train/good images, each 196 calibration / 49 held out:
random 5-fold seeds 42, 43, 44 (15 splits), contiguous 5-fold on sorted file names (5), repeated 80/20 seeds 42..51
(10; the held-out images are the first 49 of default_rng(seed).permutation(n)). Per split: the scorer is fitted on
the calibration images only; each calibration image gets a LEAVE-ONE-IMAGE-OUT score (model fitted on the other
calibration images, so calibration scores are out-of-sample); held-out images are scored by the full calibration
model; threshold = policy(calibration scores); a held-out image is a false positive iff score > threshold.
Synthetic DIAGNOSTIC (never real recall): on the canonical split (80/20 seed 42) every held-out image receives
5 defect types x 3 severities = 15 synthetic defects, so the diagnostic holds held_out_images x 15 synthetic images
(existing generators, painted at full resolution before resizing; Leather: 49 x 15 = 735, Metal Nut: 44 x 15 = 660);
recall = share scoring above that split's threshold, AUROC = synthetic vs the held-out real normals of that split.

Selection, applied mechanically:
  1. GATE      worst-fold FPR over all 30 splits <= 10%.  If no configuration passes, nothing is locked, the report
               says "No candidate satisfied the predeclared normal-data robustness gate.", and the final test is
               NOT run.
  2. PREFERRED synthetic recall >= 0.75: keep the configurations that also meet it; if none does, keep all gate
               passers and flag that the preferred floor was not met.
  3. lower worst-fold FPR (exact comparison).
  4. among ties, higher synthetic recall (recalls within 0.03 are tied).
  5. among ties, lower pooled threshold CoV (within 0.01 are tied).
  6. simpler model: Gaussian before nearest-neighbour, then smaller feature dimension, then smaller image size.
  7. candidate id, then policy id.
CATEGORY NOTE  The numbers above are for Leather (245 normals -> 196 calibration / 49 held out, 30 splits). The identical
rule is applied unchanged to any other category with n normal images: each split holds out round(0.2 * n) images
(Metal Nut: n = 220 -> 176 calibration / 44 held out, 30 splits; Pill: n = 267 -> canonical 80/20 split of 214 calibration /
53 held out because round(0.2 * 267) = round(53.4) = 53, so 53 x 15 = 795 synthetic images; when 5 does not divide n the
k-fold folds hold out 54 or 53 images); the registry, policies, gate and tie bands do not change.
The LOCKED threshold is the winning policy applied to the leave-one-image-out scores of ALL n normal images under
the final model fitted on all n. Normal-only robustness demonstrates stability of a model/threshold on the
available normal pool; it does NOT establish final-test FPR (Carpet showed it can fail to transfer).
=====================================================================================================================
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from app.ai.evaluation.category_phase2 import policy_threshold
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES, SEVERITY_NAMES
from app.ai.models.patch_anomaly import (
    AGG_TOP1PCT,
    GaussianPatchScorer,
    KNNPatchScorer,
    aggregate_scores,
    greedy_coreset_indices,
)

FAMILY_GAUSSIAN = "gaussian"
FAMILY_KNN = "knn"
SHRINKAGE = 1e-2
CORESET_FRACTION = 0.10
CORESET_SEED = 42
GATE_FPR = 0.10
SYNTHETIC_FLOOR = 0.75
RECALL_TIE = 0.03
CV_TIE = 0.01
HOLDOUT_SEEDS = tuple(range(42, 52))
RANDOM_KFOLD_SEEDS = (42, 43, 44)
KFOLD_K = 5
CANONICAL_SPLIT = ("holdout_80_20", "seed42")
POLICY_KS = (1.5, 2.0, 2.5, 3.0)
POLICY_PERCENTILES = (95, 97, 98, 99)
NO_CANDIDATE_MESSAGE = "No candidate satisfied the predeclared normal-data robustness gate."
EXPECTED_CANDIDATES = 8
EXPECTED_POLICIES = 8


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    layers: tuple[int, ...]
    image_size: int
    pooling: str = "avg3x3_stride1_pad1"
    aggregation: str = AGG_TOP1PCT
    shrinkage: float | None = None
    coreset_fraction: float | None = None

    @property
    def feature_dim(self) -> int:
        return {(2,): 128, (3,): 256, (2, 3): 384}[self.layers]

    @property
    def scorer(self) -> str:
        return "mahalanobis_global_gaussian" if self.family == FAMILY_GAUSSIAN else "nearest_neighbour_k1_memory_bank"

    def config(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["feature_dim"] = self.feature_dim
        d["scorer"] = self.scorer
        d["backbone"] = "resnet18_imagenet_frozen"
        return d


def registry() -> list[CandidateSpec]:
    specs = []
    for layers, size in (((2,), 128), ((3,), 128), ((2, 3), 128), ((2, 3), 256)):
        tag = "l" + "".join(str(x) for x in layers)
        specs.append(CandidateSpec(f"gauss_{tag}_{size}", FAMILY_GAUSSIAN, layers, size, shrinkage=SHRINKAGE))
    for layers, size in (((2,), 128), ((3,), 128), ((2, 3), 128), ((2, 3), 256)):
        tag = "l" + "".join(str(x) for x in layers)
        specs.append(CandidateSpec(f"knn_{tag}_{size}", FAMILY_KNN, layers, size, coreset_fraction=CORESET_FRACTION))
    return specs


def registry_fingerprint(specs: list[CandidateSpec] | None = None) -> str:
    import json

    return hashlib.sha256(json.dumps([s.config() for s in (specs or registry())], sort_keys=True).encode()).hexdigest()


def policy_ids() -> list[str]:
    return [f"mean_std_{k:g}" for k in POLICY_KS] + [f"percentile_{p}" for p in POLICY_PERCENTILES]


def simplicity_key(spec: CandidateSpec) -> tuple:
    return (0 if spec.family == FAMILY_GAUSSIAN else 1, spec.feature_dim, spec.image_size)


# ---------------------------------------------------------------------------
# Deterministic normal-only splits (indices into the file-name-sorted train/good list)
# ---------------------------------------------------------------------------

def _split(name: str, held: np.ndarray, n: int) -> dict:
    held = np.sort(held)
    return {"name": name, "held_out": held, "calibration": np.setdiff1d(np.arange(n), held)}


def scheme_splits(n: int) -> dict[str, list[dict]]:
    held_size = int(round(n * 0.2))
    schemes: dict[str, list[dict]] = {}
    schemes["holdout_80_20"] = [_split(f"seed{s}", np.random.default_rng(s).permutation(n)[:held_size], n) for s in HOLDOUT_SEEDS]
    for seed in RANDOM_KFOLD_SEEDS:
        folds = np.array_split(np.random.default_rng(seed).permutation(n), KFOLD_K)
        schemes[f"random_kfold5_seed{seed}"] = [_split(f"fold{i}", f, n) for i, f in enumerate(folds)]
    schemes["contiguous_kfold5"] = [_split(f"block{i}", f, n) for i, f in enumerate(np.array_split(np.arange(n), KFOLD_K))]
    return schemes


def splits_fingerprint(schemes: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for name, splits in schemes.items():
        for s in splits:
            h.update(f"{name}/{s['name']}:{s['held_out'].tolist()}|{s['calibration'].tolist()}\n".encode())
    return h.hexdigest()


def split_audit(n: int, schemes: dict[str, list[dict]]) -> dict:
    all_splits = [s for v in schemes.values() for s in v]
    return {
        "split_count": len(all_splits),
        "calibration_and_held_out_disjoint_in_every_split": all(not set(s["held_out"]) & set(s["calibration"]) for s in all_splits),
        "every_split_covers_the_pool": all(set(s["held_out"]) | set(s["calibration"]) == set(range(n)) for s in all_splits),
        "kfold_held_out_folds_partition_the_pool": all(
            sorted(np.concatenate([s["held_out"] for s in v]).tolist()) == list(range(n)) for k, v in schemes.items() if "kfold" in k),
        "held_out_sizes": sorted({len(s["held_out"]) for s in all_splits}),
        "calibration_sizes": sorted({len(s["calibration"]) for s in all_splits}),
    }


# ---------------------------------------------------------------------------
# Family A: global Gaussian with exact leave-one-image-out scoring from per-image sufficient statistics
# ---------------------------------------------------------------------------

def image_statistics(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image (sum x, sum x x^T) in float64 over that image's patches."""
    n, d = patches.shape[0], patches.shape[-1]
    sums = torch.empty(n, d, dtype=torch.float64)
    outers = torch.empty(n, d, d, dtype=torch.float64)
    for i in range(n):
        x = patches[i].reshape(-1, d).double()
        sums[i] = x.sum(0)
        outers[i] = x.T @ x
    return sums, outers


def image_scores(scorer, patches: torch.Tensor, aggregation: str = AGG_TOP1PCT) -> np.ndarray:
    return aggregate_scores(scorer.patch_scores(patches), aggregation).double().numpy()


def gaussian_split_scores(patches, sums, outers, calibration, held_out, shrinkage=SHRINKAGE, aggregation=AGG_TOP1PCT):
    """(leave-one-image-out scores of the calibration images, scores of the held-out images by the full model)."""
    per_image = patches.shape[1] * patches.shape[2]
    cal = torch.as_tensor(calibration)
    total, outer_total = sums[cal].sum(0), outers[cal].sum(0)
    count = per_image * len(calibration)
    held = np.empty(0)
    if len(held_out):
        model = GaussianPatchScorer.from_statistics(count, total, outer_total, shrinkage)
        held = image_scores(model, patches[torch.as_tensor(held_out)], aggregation)
    loio = np.empty(len(calibration))
    for k, i in enumerate(calibration):
        model_i = GaussianPatchScorer.from_statistics(count - per_image, total - sums[i], outer_total - outers[i], shrinkage)
        loio[k] = image_scores(model_i, patches[int(i) : int(i) + 1], aggregation)[0]
    return loio, held


# ---------------------------------------------------------------------------
# Family B: memory bank = union of per-image coresets; pairwise minima make any fold's scores exact and cheap
# ---------------------------------------------------------------------------

def per_image_coresets(patches: torch.Tensor, fraction: float = CORESET_FRACTION, seed: int = CORESET_SEED) -> list[torch.Tensor]:
    d = patches.shape[-1]
    banks = []
    for i in range(patches.shape[0]):
        flat = patches[i].reshape(-1, d)
        n_select = max(1, int(round(flat.shape[0] * fraction)))
        banks.append(flat[greedy_coreset_indices(flat, n_select, seed)].clone())
    return banks


@torch.no_grad()
def pairwise_minima(patches: torch.Tensor, banks: list[torch.Tensor]) -> torch.Tensor:
    """M[i, j, p] = distance from patch p of image i to its nearest patch in image j's coreset.
    The distance of image i to the bank of a set S of images is then min_{j in S} M[i, j, p]. The diagonal
    (an image against its own coreset) is set to +inf so any image set containing i scores i leave-one-out."""
    n = patches.shape[0]
    p = patches.shape[1] * patches.shape[2]
    sizes = {b.shape[0] for b in banks}
    if len(sizes) != 1:
        raise ValueError("Per-image coresets must all have the same size.")
    s = sizes.pop()
    bank_all = torch.cat(banks, dim=0).contiguous()
    m = torch.empty(n, n, p, dtype=torch.float32)
    for i in range(n):
        dist = torch.cdist(patches[i].reshape(p, -1), bank_all)
        m[i] = dist.reshape(p, n, s).min(dim=2).values.T
        m[i, i, :] = float("inf")
    return m


def knn_split_scores(minima: torch.Tensor, h: int, w: int, calibration, held_out, aggregation=AGG_TOP1PCT, chunk: int = 32):
    n = minima.shape[0]
    cal = torch.as_tensor(calibration)
    patch_min = torch.empty(n, minima.shape[2])
    for start in range(0, n, chunk):
        patch_min[start : start + chunk] = minima[start : start + chunk].index_select(1, cal).min(dim=1).values
    scores = aggregate_scores(patch_min.reshape(n, h, w), aggregation).double().numpy()
    return scores[np.asarray(calibration)], scores[np.asarray(held_out)]


def knn_bank(banks: list[torch.Tensor], indices) -> KNNPatchScorer:
    return KNNPatchScorer(torch.cat([banks[int(i)] for i in indices], dim=0))


# ---------------------------------------------------------------------------
# Per-candidate summaries
# ---------------------------------------------------------------------------

def _stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "cv": float(a.std() / a.mean()) if a.mean() else float("inf"),
            "min": float(a.min()), "max": float(a.max())}


def summarize_candidate(split_scores: dict[str, list[dict]], all_normal_scores: np.ndarray, synthetic: dict | None) -> dict:
    """`split_scores[scheme]` = [{name, calibration: ndarray, held_out: ndarray}]; `synthetic` = {scores, meta}
    scored by the canonical split's model (or None)."""
    policies = {}
    for policy in policy_ids():
        per_scheme, pooled_thr, pooled_fpr, canonical = {}, [], [], None
        for scheme, rows in split_scores.items():
            out_rows = []
            for r in rows:
                thr = policy_threshold(policy, r["calibration"])
                fp = int((r["held_out"] > thr).sum())
                out_rows.append({"split": r["name"], "threshold": thr, "held_out_size": len(r["held_out"]),
                                 "false_positives": fp, "fpr": fp / len(r["held_out"]),
                                 "calibration_mean": float(np.mean(r["calibration"])), "calibration_std": float(np.std(r["calibration"]))})
                if (scheme, r["name"]) == CANONICAL_SPLIT:
                    canonical = thr
            thr_list = [x["threshold"] for x in out_rows]
            fpr_list = [x["fpr"] for x in out_rows]
            pooled_thr += thr_list
            pooled_fpr += fpr_list
            per_scheme[scheme] = {"splits": out_rows, "threshold": _stats(thr_list), "mean_fpr": float(np.mean(fpr_list)),
                                  "worst_fold_fpr": float(np.max(fpr_list))}
        entry = {"policy": policy, "per_scheme": per_scheme, "pooled_threshold": _stats(pooled_thr),
                 "mean_fpr": float(np.mean(pooled_fpr)), "worst_fold_fpr": float(np.max(pooled_fpr)),
                 "full_pool_threshold": policy_threshold(policy, all_normal_scores), "canonical_threshold": canonical}
        if synthetic is not None:
            flags = np.asarray(synthetic["scores"]) > canonical
            meta = synthetic["meta"]
            entry["synthetic_recall"] = float(flags.mean())
            entry["synthetic_recall_by_type"] = {t: float(np.mean([f for f, m in zip(flags, meta) if m["defect_type"] == t])) for t in DEFECT_TYPES}
            entry["synthetic_recall_by_severity"] = {SEVERITY_NAMES[v]: float(np.mean([f for f, m in zip(flags, meta) if m["severity"] == v])) for v in SEVERITIES}
        policies[policy] = entry
    out = {"policies": policies, "normal_score_distribution": _stats(all_normal_scores) | {
        "count": int(len(all_normal_scores)), "median": float(np.median(all_normal_scores)),
        "p95": float(np.percentile(all_normal_scores, 95)), "p99": float(np.percentile(all_normal_scores, 99))}}
    if synthetic is not None:
        canon_rows = next(r for r in split_scores[CANONICAL_SPLIT[0]] if r["name"] == CANONICAL_SPLIT[1])
        y = np.array([0] * len(canon_rows["held_out"]) + [1] * len(synthetic["scores"]))
        out["synthetic_auroc"] = float(roc_auc_score(y, np.concatenate([canon_rows["held_out"], synthetic["scores"]])))
        out["synthetic_score_mean"] = float(np.mean(synthetic["scores"]))
    return out


# ---------------------------------------------------------------------------
# Selection (the pre-declared rule)
# ---------------------------------------------------------------------------

def select_configuration(results: dict[str, dict], specs: list[CandidateSpec]) -> dict:
    by_id = {s.candidate_id: s for s in specs}
    rows = []
    for cid, res in results.items():
        for policy, e in res["policies"].items():
            rows.append({"candidate": cid, "policy": policy, "worst_fold_fpr": e["worst_fold_fpr"], "mean_fpr": e["mean_fpr"],
                         "synthetic_recall": e.get("synthetic_recall"), "synthetic_auroc": res.get("synthetic_auroc"),
                         "threshold_cv": e["pooled_threshold"]["cv"], "full_pool_threshold": e["full_pool_threshold"],
                         "passes_gate": e["worst_fold_fpr"] <= GATE_FPR + 1e-12})
    passing = [r for r in rows if r["passes_gate"]]
    base = {"rows": rows, "gate_passers": [f"{r['candidate']}|{r['policy']}" for r in passing]}
    if not passing:
        return {**base, "winner": None, "message": NO_CANDIDATE_MESSAGE, "reasoning": NO_CANDIDATE_MESSAGE, "preferred_floor_met": False}
    preferred = [r for r in passing if (r["synthetic_recall"] or 0) >= SYNTHETIC_FLOOR - 1e-12]
    floor_met = bool(preferred)
    pool = preferred if floor_met else passing
    best_fpr = min(round(r["worst_fold_fpr"], 12) for r in pool)
    pool = [r for r in pool if round(r["worst_fold_fpr"], 12) == best_fpr]
    best_recall = max(r["synthetic_recall"] or 0 for r in pool)
    pool = [r for r in pool if (r["synthetic_recall"] or 0) >= best_recall - RECALL_TIE - 1e-12]
    best_cv = min(r["threshold_cv"] for r in pool)
    pool = [r for r in pool if r["threshold_cv"] <= best_cv + CV_TIE + 1e-12]
    pool.sort(key=lambda r: (simplicity_key(by_id[r["candidate"]]), r["candidate"], r["policy"]))
    winner = pool[0]
    reasoning = (f"{len(passing)} of {len(rows)} configurations pass the gate (worst-fold FPR <= {GATE_FPR:.0%}); "
                 f"{len(preferred)} also meet the preferred synthetic-recall floor ({SYNTHETIC_FLOOR:.0%})"
                 + ("" if floor_met else " - NONE did, so the floor was not applied") +
                 f"; lowest worst-fold FPR {best_fpr:.4f}; {len(pool)} left after the recall/CoV tie bands; simplicity then id chose "
                 f"{winner['candidate']} with {winner['policy']}.")
    return {**base, "winner": f"{winner['candidate']}|{winner['policy']}", "winner_candidate": winner["candidate"],
            "winner_policy": winner["policy"], "winner_row": winner, "threshold": winner["full_pool_threshold"],
            "message": None, "reasoning": reasoning, "preferred_floor_met": floor_met, "tied_at_end": [f"{r['candidate']}|{r['policy']}" for r in pool]}

