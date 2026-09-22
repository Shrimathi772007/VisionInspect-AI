"""Calibration-robustness and score-aggregation study on GOOD images only (train/good).

Question: the frozen Carpet representation ranks defects well, but a threshold calibrated on 56 validation-good
images gave a 57% false-positive rate on the final good images. Is that (1) threshold instability, (2) too little
validation-good coverage, (3) an over-sensitive score aggregation, (4) a shift between the train/validation pool
and later good images, or a mix? This module estimates, with NO real defect and NO test image, how a threshold
calibrated on some good images behaves on OTHER unseen good images:

  * One frozen feature extractor + one Gaussian per fold; patch features are computed ONCE for all good images
    and each fold's Gaussian is fitted from per-image sufficient statistics (cheap, exact).
  * Deterministic good-only split schemes: the existing 224/56 split, random k-fold (several fixed seeds),
    repeated 80/20 - 75/25 - 70/30 holdouts, and contiguous file-order blocks (a proxy for acquisition drift).
  * Every held-out image is scored by a model that never saw it (out-of-fold), then image-level aggregations
    are derived from the same patch-score maps, so only the aggregation differs between candidates.
  * "Transfer" = calibrate a threshold policy on some held-out scores, measure the false-positive rate on other
    unseen good images. That is the val -> test situation, measured without touching the test set.

PRE-DECLARED SELECTION POLICY (written before the study was run; never changed afterwards)
------------------------------------------------------------------------------------------
Candidate = (aggregation, threshold policy). Final threshold = policy applied to the POOLED out-of-fold scores;
with two calibration schemes (random 5-fold, contiguous blocks) the more conservative (larger) threshold is used.
  Gate P (primary)    worst-fold transfer FPR <= 10% under BOTH schemes (calibrate on the other folds' pooled
                      scores, measure on the held-out fold).
  Usefulness floor    synthetic-defect recall at the final threshold >= 0.75 (validation-side diagnostic ONLY).
  Selection           among candidates passing gate P and the floor: prefer those also meeting the secondary
                      target (mean transfer FPR <= 5% under both schemes) if any exist; then highest synthetic
                      recall; recalls within 0.03 are tied -> lower out-of-fold coefficient of variation ->
                      simpler policy (mean+K*std before percentile) -> candidate id.
  Fallback            if nothing passes gate P + floor: minimise the worst-fold transfer FPR (ties: higher
                      synthetic recall) and flag the result as NOT meeting the gate - it is then kept only as an
                      experiment, never called validated.
Synthetic recall is a diagnostic, never real defect recall. Real final-test images influence nothing here; this
module does not import test discovery (a test asserts that from the source).
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import roc_auc_score

from app.ai.evaluation.threshold_experiments import generate_candidates
from app.ai.models.patch_anomaly import (
    AGG_MAX,
    AGG_TOP1PCT,
    AGG_TOP1PCT_MEDIAN,
    AGG_TOP5PCT,
    AGG_TOP5PCT_MEDIAN,
    GaussianPatchScorer,
    aggregate_scores,
    extract_patch_features,
)
from app.ai.models.resnet18 import ResNet18

STUDY_AGGREGATIONS = (AGG_TOP1PCT, AGG_TOP5PCT, AGG_MAX, AGG_TOP1PCT_MEDIAN, AGG_TOP5PCT_MEDIAN)
PRIMARY_FPR = 0.10
SECONDARY_FPR = 0.05
RECALL_FLOOR = 0.75
RECALL_TIE = 0.03
SHRINKAGE = 1e-2

# Split seeds are fixed and documented; nothing is chosen after seeing results.
RANDOM_KFOLD_SEEDS = (42, 43, 44)
RANDOM_KFOLD_K = 5
HOLDOUT_FRACTIONS = (0.20, 0.25, 0.30)
HOLDOUT_REPEATS = 10
HOLDOUT_SEED = 1000
BLOCK_K = 5

SCHEME_RANDOM = "random_5fold"
SCHEME_BLOCK = "contiguous_blocks"


# ---------------------------------------------------------------------------
# Good-image pool with cached features and per-image sufficient statistics
# ---------------------------------------------------------------------------

@dataclass
class GoodPool:
    files: list[str]  # pool order = sorted filenames
    patches: torch.Tensor  # (N, h, w, C) float32
    sums: torch.Tensor  # (N, C) float64
    outers: torch.Tensor  # (N, C, C) float64
    patches_per_image: int


def build_pool(extractor: ResNet18, images: torch.Tensor, files: list[str], layers: tuple[int, ...]) -> GoodPool:
    """Extract patch features once for every good image and precompute the Gaussian sufficient statistics."""
    patches = extract_patch_features(extractor, images, layers)
    n, h, w, c = patches.shape
    sums = torch.empty(n, c, dtype=torch.float64)
    outers = torch.empty(n, c, c, dtype=torch.float64)
    for i in range(n):
        x = patches[i].reshape(-1, c).double()
        sums[i] = x.sum(0)
        outers[i] = x.T @ x
    return GoodPool(list(files), patches, sums, outers, h * w)


def fit_scorer(pool: GoodPool, train_idx) -> GaussianPatchScorer:
    idx = list(train_idx)
    return GaussianPatchScorer.from_statistics(
        len(idx) * pool.patches_per_image, pool.sums[idx].sum(0), pool.outers[idx].sum(0), SHRINKAGE
    )


# ---------------------------------------------------------------------------
# Deterministic good-only splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    scheme: str
    fold_id: str
    train_idx: tuple[int, ...]
    heldout_idx: tuple[int, ...]


def _fold(scheme: str, fold_id: str, n: int, heldout) -> Fold:
    heldout = tuple(sorted(int(i) for i in heldout))
    held = set(heldout)
    return Fold(scheme, fold_id, tuple(i for i in range(n) if i not in held), heldout)


def random_kfold(n: int, k: int, seed: int) -> list[Fold]:
    parts = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [_fold(SCHEME_RANDOM, f"seed{seed}_fold{i}", n, part) for i, part in enumerate(parts)]


def repeated_holdout(n: int, heldout_fraction: float, repeats: int, seed: int) -> list[Fold]:
    size = int(round(n * heldout_fraction))
    rng = np.random.default_rng(seed)
    label = f"holdout{int(round((1 - heldout_fraction) * 100))}_{int(round(heldout_fraction * 100))}"
    return [_fold(label, f"{label}_rep{r}", n, rng.permutation(n)[:size]) for r in range(repeats)]


def block_kfold(n: int, k: int) -> list[Fold]:
    return [_fold(SCHEME_BLOCK, f"block{i}", n, part) for i, part in enumerate(np.array_split(np.arange(n), k))]


def existing_split(files: list[str], validation_files: list[str]) -> Fold:
    held = {files.index(f) for f in validation_files}
    return _fold("existing_224_56", "existing", len(files), held)


def all_schemes(n: int) -> dict[str, list[Fold]]:
    schemes: dict[str, list[Fold]] = {SCHEME_RANDOM: [], SCHEME_BLOCK: block_kfold(n, BLOCK_K)}
    for seed in RANDOM_KFOLD_SEEDS:
        schemes[SCHEME_RANDOM].extend(random_kfold(n, RANDOM_KFOLD_K, seed))
    for i, frac in enumerate(HOLDOUT_FRACTIONS):
        folds = repeated_holdout(n, frac, HOLDOUT_REPEATS, HOLDOUT_SEED + i)
        schemes[folds[0].scheme] = folds
    return schemes


# ---------------------------------------------------------------------------
# Out-of-fold patch-score maps and aggregation
# ---------------------------------------------------------------------------

@torch.no_grad()
def fold_maps(pool: GoodPool, fold: Fold) -> torch.Tensor:
    """(held-out images, h, w) patch anomaly maps from a Gaussian fitted on this fold's TRAINING images only."""
    scorer = fit_scorer(pool, fold.train_idx)
    return scorer.patch_scores(pool.patches[list(fold.heldout_idx)])


def aggregate_np(maps: torch.Tensor, aggregation: str) -> np.ndarray:
    return aggregate_scores(maps, aggregation).numpy().astype(np.float64)


@dataclass
class FoldScores:
    fold: Fold
    scores: dict[str, np.ndarray]  # aggregation -> held-out image scores (aligned with fold.heldout_idx)


def score_folds(pool: GoodPool, folds: list[Fold], aggregations=STUDY_AGGREGATIONS) -> list[FoldScores]:
    out = []
    for fold in folds:
        maps = fold_maps(pool, fold)
        out.append(FoldScores(fold, {a: aggregate_np(maps, a) for a in aggregations}))
    return out


# ---------------------------------------------------------------------------
# Threshold policies and transfer
# ---------------------------------------------------------------------------

def policy_thresholds(scores) -> dict[tuple[str, float], float]:
    """The nine established Phase 3 candidates (mean+K*std, K in 3..1; percentiles 95..99) on `scores`."""
    return {(c.method, c.parameter): c.threshold for c in generate_candidates([float(s) for s in scores])}


POLICIES = tuple(policy_thresholds(np.arange(10.0)).keys())


def fpr(threshold: float, scores) -> float:
    return float(np.mean(np.asarray(scores) > threshold))


def single_fold_transfer(fold_scores: list[FoldScores], aggregation: str, policy) -> list[dict]:
    """Calibrate on ONE held-out set, evaluate on every OTHER unseen good image (images of the calibration set
    excluded). Answers 'how good is a threshold picked from a validation set of this size?'."""
    rows = []
    for fs in fold_scores:
        tau = policy_thresholds(fs.scores[aggregation])[policy]
        mine = set(fs.fold.heldout_idx)
        others = np.concatenate([
            o.scores[aggregation][[j for j, idx in enumerate(o.fold.heldout_idx) if idx not in mine]]
            for o in fold_scores if o is not fs
        ])
        rows.append({"fold": fs.fold.fold_id, "threshold": tau, "fpr": fpr(tau, others), "n_eval": int(len(others)),
                     "in_sample_fpr": fpr(tau, fs.scores[aggregation])})
    return rows


def pooled_transfer(fold_scores: list[FoldScores], aggregation: str, policy, group_key=lambda f: f.fold.fold_id.rsplit("_fold", 1)[0]) -> list[dict]:
    """Leave-one-fold-out over PARTITION folds: calibrate on the other folds' pooled out-of-fold scores, measure
    on the held-out fold. This mirrors the final procedure (calibrate on ~all good scores, apply to new images).
    Folds are grouped by `group_key` so each random-seed repetition is its own partition."""
    rows = []
    groups: dict[str, list[FoldScores]] = {}
    for fs in fold_scores:
        groups.setdefault(group_key(fs), []).append(fs)
    for members in groups.values():
        for fs in members:
            calib = np.concatenate([o.scores[aggregation] for o in members if o is not fs])
            tau = policy_thresholds(calib)[policy]
            rows.append({"fold": fs.fold.fold_id, "threshold": tau, "fpr": fpr(tau, fs.scores[aggregation]),
                         "n_eval": len(fs.fold.heldout_idx)})
    return rows


def pooled_oof(fold_scores: list[FoldScores], aggregation: str) -> np.ndarray:
    """All out-of-fold scores of a partition scheme (each image once per repetition)."""
    return np.concatenate([fs.scores[aggregation] for fs in fold_scores])


def summarize_transfer(rows: list[dict]) -> dict:
    f = np.array([r["fpr"] for r in rows])
    t = np.array([r["threshold"] for r in rows])
    return {"folds": len(rows), "fpr_mean": float(f.mean()), "fpr_max": float(f.max()), "fpr_min": float(f.min()),
            "share_folds_le_10pct": float(np.mean(f <= PRIMARY_FPR)), "share_folds_le_5pct": float(np.mean(f <= SECONDARY_FPR)),
            "threshold_mean": float(t.mean()), "threshold_std": float(t.std()),
            "threshold_cv": float(t.std() / t.mean()) if t.mean() else float("inf")}


def distribution_summary(scores) -> dict:
    s = np.asarray(scores, dtype=float)
    q = np.percentile(s, [50, 90, 95, 97, 98, 99])
    return {"n": int(len(s)), "mean": float(s.mean()), "std": float(s.std()), "median": float(np.median(s)),
            "min": float(s.min()), "max": float(s.max()), "cov": float(s.std() / s.mean()),
            "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p97": float(q[3]), "p98": float(q[4]), "p99": float(q[5])}


def drift_diagnostics(block_scores: list[FoldScores], random_scores: list[FoldScores], aggregation: str, n: int) -> dict:
    """Is there structure in the good pool that a random validation split hides? (train/good images only)"""
    groups = [fs.scores[aggregation] for fs in block_scores]
    kw = kruskal(*groups)
    index_of = np.concatenate([fs.fold.heldout_idx for fs in block_scores])
    score_of = np.concatenate(groups)
    rho = spearmanr(index_of, score_of)
    first_random = [fs for fs in random_scores if fs.fold.fold_id.startswith(f"seed{RANDOM_KFOLD_SEEDS[0]}_")]
    rand_all = np.concatenate([fs.scores[aggregation] for fs in first_random])
    return {
        "block_medians": [float(np.median(g)) for g in groups],
        "block_means": [float(np.mean(g)) for g in groups],
        "block_stds": [float(np.std(g)) for g in groups],
        "kruskal_wallis_p_between_blocks": float(kw.pvalue),
        "spearman_file_index_vs_score": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
        "random_oof_distribution": distribution_summary(rand_all),
        "block_oof_distribution": distribution_summary(score_of),
    }


# ---------------------------------------------------------------------------
# Synthetic diagnostics (validation-side only)
# ---------------------------------------------------------------------------

def synthetic_recall(threshold: float, synthetic_scores) -> float:
    return float(np.mean(np.asarray(synthetic_scores) > threshold))


def synthetic_auroc(good_scores, synthetic_scores) -> float:
    y = np.r_[np.zeros(len(good_scores)), np.ones(len(synthetic_scores))]
    return float(roc_auc_score(y, np.r_[good_scores, synthetic_scores]))


# ---------------------------------------------------------------------------
# Pre-declared selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigRow:
    aggregation: str
    policy: tuple[str, float]
    fpr_max: float  # worst-fold pooled-transfer FPR, max over the two calibration schemes
    fpr_mean: float  # mean pooled-transfer FPR, max over the two calibration schemes
    final_threshold: float  # policy on the pooled OOF scores; the larger of the two schemes' thresholds
    synthetic_recall: float
    oof_cov: float

    @property
    def config_id(self) -> str:
        return f"{self.aggregation}|{self.policy[0]}_{self.policy[1]:g}"


@dataclass(frozen=True)
class ConfigSelection:
    winner: ConfigRow
    passes_primary_gate: bool
    meets_secondary_target: bool
    reasoning: str
    eligible: list[str]


def _simplicity(row: ConfigRow) -> int:
    return 0 if row.policy[0] == "mean_std" else 1


def select_configuration(rows: list[ConfigRow]) -> ConfigSelection:
    if not rows:
        raise ValueError("No candidate configurations to select from.")
    passing = [r for r in rows if r.fpr_max <= PRIMARY_FPR and r.synthetic_recall >= RECALL_FLOOR]
    if passing:
        secondary = [r for r in passing if r.fpr_mean <= SECONDARY_FPR]
        pool = secondary or passing
        best_recall = max(r.synthetic_recall for r in pool)
        tied = [r for r in pool if best_recall - r.synthetic_recall <= RECALL_TIE]
        winner = min(tied, key=lambda r: (r.oof_cov, _simplicity(r), r.config_id))
        return ConfigSelection(
            winner, True, bool(secondary),
            f"{len(passing)} configuration(s) passed gate P (worst-fold transfer FPR <= {PRIMARY_FPR:.0%} under both "
            f"schemes) and the synthetic-recall floor ({RECALL_FLOOR}); {len(secondary)} also met the secondary "
            f"target (mean transfer FPR <= {SECONDARY_FPR:.0%}). Highest synthetic recall {best_recall:.3f}; "
            f"{len(tied)} within {RECALL_TIE}; lowest OOF CoV / simplest policy -> {winner.config_id}.",
            [r.config_id for r in passing],
        )
    winner = min(rows, key=lambda r: (r.fpr_max, -r.synthetic_recall, r.config_id))
    return ConfigSelection(
        winner, False, False,
        f"NO configuration passed gate P and the recall floor. Fallback: smallest worst-fold transfer FPR "
        f"({winner.fpr_max:.1%}) -> {winner.config_id}. It does NOT meet the declared gate and is an experiment only.",
        [],
    )


__all__ = [
    "STUDY_AGGREGATIONS", "POLICIES", "Fold", "FoldScores", "GoodPool", "ConfigRow", "ConfigSelection",
    "all_schemes", "block_kfold", "build_pool", "distribution_summary", "drift_diagnostics", "existing_split",
    "fit_scorer", "fold_maps", "fpr", "pooled_oof", "pooled_transfer", "policy_thresholds", "random_kfold",
    "repeated_holdout", "score_folds", "select_configuration", "single_fold_transfer", "summarize_transfer",
    "synthetic_auroc", "synthetic_recall",
]
