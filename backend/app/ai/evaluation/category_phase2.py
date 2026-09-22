"""Phase 2 for one MVTec category: a normal-only THRESHOLD ROBUSTNESS / SELECTION study on the frozen Phase 1
ConvAutoencoder. The model, its weights, preprocessing and image size are not changed; the only variable is how the
threshold is chosen.

This module can score and split ONLY train/good images. It never imports test discovery (asserted from source by
tests). The protected final test is touched only by app.ai.evaluation.category_phase2_final, after the threshold
has been locked on disk.

=====================================================================================================================
PRE-DECLARED SELECTION RULE  (written before any calibration score was computed; not to be altered after results)
=====================================================================================================================
Calibration data: the category's train/good images, sorted by file name. Every split is deterministic:
  * holdout schemes  A 80/20 (49 held out), B 75/25 (61 held out), C 70/30 (74 held out); each has HOLDOUT_REPEATS=10
    splits, split i drawn from numpy.random.default_rng(HOLDOUT_BASE_SEED + i).permutation(n) (the first `held_out`
    permuted positions are held out). Split 0 (seed 42) is the scheme's canonical split.
  * 5-fold with seeds 42, 43, 44: permutation from numpy.random.default_rng(seed); numpy.array_split into 5 folds;
    each fold is held out once, the other four calibrate. (Folds partition the pool.)
Per split and per candidate policy: threshold = policy(calibration scores); held-out false positives = held-out
normal scores STRICTLY ABOVE the threshold. Candidate policies: mean + K*std (population std, ddof=0) for K in
{1.0, 1.5, 2.0, 2.5, 3.0}; linear-interpolation percentiles P95, P97, P98, P99 of the calibration scores.

For each policy, over ALL splits of ALL six schemes pooled together:
  worst_fpr = maximum held-out FPR;   cv = std(thresholds) / mean(thresholds)   (population std).
Selection, applied mechanically:
  1. ELIGIBLE  iff worst_fpr <= 10%.  If nothing is eligible, no threshold is selected or locked, the report states
     "No threshold candidate satisfied the predeclared calibration FPR requirement.", the requirement is NOT
     relaxed, and the final test is NOT run.
  2. Stability: keep eligible policies whose cv is within CV_TIE=0.01 (one percentage point) of the smallest eligible
     cv (i.e. "effectively equivalent" stability).
  3. Among those, lower worst_fpr wins (exact comparison).
  4. Remaining ties: simpler policy (mean+K*std before percentile), then policy id.
The LOCKED threshold is the winning policy applied to ALL train/good scores (the full pool). No test image, test
label or test score enters any step. Held-out FPR here is measured on images the model was TRAINED on (the model is
frozen and not retrained), so it measures threshold-policy behaviour and stability, not out-of-sample behaviour;
this is a stated limitation of the design, not a hidden one.
=====================================================================================================================
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from app.ai.evaluation.category_phase1 import describe, file_hashes
from app.ai.evaluation.evaluate import _sample_to_tensor, compute_reconstruction_error
from app.ai.training import artifacts
from app.ai.training.dataset import discover_train_samples
from app.ai.training.model import build_model

PHASE2_DIRNAME = "phase2_threshold"
EXPERIMENT_ID_PREFIX = "phase2-threshold-study"

MEAN_STD_KS = (1.0, 1.5, 2.0, 2.5, 3.0)
PERCENTILES = (95, 97, 98, 99)
PRIMARY_FPR = 0.10
CV_TIE = 0.01
HOLDOUT_REPEATS = 10
HOLDOUT_BASE_SEED = 42
KFOLD_SEEDS = (42, 43, 44)
KFOLD_K = 5
HOLDOUT_SCHEMES = {"A_80_20": 49, "B_75_25": 61, "C_70_30": 74}  # held-out sizes for 245 images
NO_CANDIDATE_MESSAGE = "No threshold candidate satisfied the predeclared calibration FPR requirement."

DECLARED_RULE = {
    "eligibility": f"worst held-out-normal FPR over all splits of all schemes <= {PRIMARY_FPR:.0%}",
    "stability": f"pooled threshold coefficient of variation; policies within {CV_TIE} of the smallest eligible CV are equivalent",
    "then": ["lower worst-fold FPR", "simpler policy (mean+K*std before percentile)", "policy id"],
    "locked_threshold": "the winning policy applied to all train/good scores",
    "if_none_eligible": NO_CANDIDATE_MESSAGE + " (not relaxed; final test not run)",
    "false_positive": "held-out normal score strictly above the threshold",
    "std": "population (ddof=0)", "percentile": "numpy linear interpolation",
}


def policy_ids() -> list[str]:
    return [f"mean_std_{k:g}" for k in MEAN_STD_KS] + [f"percentile_{p}" for p in PERCENTILES]


def policy_threshold(policy: str, scores) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if policy.startswith("mean_std_"):
        return float(arr.mean() + float(policy.removeprefix("mean_std_")) * arr.std())
    if policy.startswith("percentile_"):
        return float(np.percentile(arr, float(policy.removeprefix("percentile_"))))
    raise ValueError(f"Unknown threshold policy: {policy}")


def simplicity_rank(policy: str) -> int:
    return 0 if policy.startswith("mean_std_") else 1


# ---------------------------------------------------------------------------
# Deterministic normal-only splits (indices into the file-name-sorted train/good list)
# ---------------------------------------------------------------------------

def holdout_splits(n: int, held_out: int, repeats: int = HOLDOUT_REPEATS, base_seed: int = HOLDOUT_BASE_SEED) -> list[dict]:
    splits = []
    for i in range(repeats):
        perm = np.random.default_rng(base_seed + i).permutation(n)
        splits.append({"name": f"seed{base_seed + i}", "held_out": np.sort(perm[:held_out]), "calibration": np.sort(perm[held_out:])})
    return splits


def kfold_splits(n: int, seed: int, k: int = KFOLD_K) -> list[dict]:
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    splits = []
    for i, fold in enumerate(folds):
        held = np.sort(fold)
        splits.append({"name": f"fold{i}", "held_out": held, "calibration": np.sort(np.setdiff1d(np.arange(n), held))})
    return splits


def all_schemes(n: int) -> dict[str, list[dict]]:
    schemes = {name: holdout_splits(n, held) for name, held in HOLDOUT_SCHEMES.items()}
    for seed in KFOLD_SEEDS:
        schemes[f"kfold5_seed{seed}"] = kfold_splits(n, seed)
    return schemes


def splits_fingerprint(schemes: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for name, splits in schemes.items():
        for s in splits:
            h.update(f"{name}/{s['name']}:{s['held_out'].tolist()}|{s['calibration'].tolist()}\n".encode())
    return h.hexdigest()


def split_audit(n: int, schemes: dict[str, list[dict]]) -> dict:
    disjoint = all(not set(s["held_out"]) & set(s["calibration"]) and len(s["held_out"]) + len(s["calibration"]) == n
                   for splits in schemes.values() for s in splits)
    covers = all(set(s["held_out"]) | set(s["calibration"]) == set(range(n)) for splits in schemes.values() for s in splits)
    partition = all(sorted(np.concatenate([s["held_out"] for s in splits]).tolist()) == list(range(n))
                    for name, splits in schemes.items() if name.startswith("kfold5"))
    return {"calibration_and_held_out_disjoint_in_every_split": bool(disjoint),
            "every_split_covers_the_pool": bool(covers), "kfold_held_out_folds_partition_the_pool": bool(partition),
            "held_out_sizes": {name: sorted({len(s["held_out"]) for s in splits}) for name, splits in schemes.items()},
            "calibration_sizes": {name: sorted({len(s["calibration"]) for s in splits}) for name, splits in schemes.items()}}


# ---------------------------------------------------------------------------
# Scoring the frozen model on train/good only
# ---------------------------------------------------------------------------

def score_train_good(category: str, model_path: Path, expected_sha256: str, image_size: tuple[int, int] = (128, 128)) -> dict:
    """Verify the frozen model's hash, then score every train/good image (sorted by file name). Nothing else."""
    model_path = Path(model_path)
    actual = file_hashes(model_path)
    if actual["sha256"] != expected_sha256:
        raise RuntimeError(f"Frozen Phase 1 model hash mismatch: expected {expected_sha256}, found {actual['sha256']}")
    samples = discover_train_samples(category)
    if any(s.split != "train" or s.defect_type != "good" for s in samples):
        raise RuntimeError("Calibration pool contains a non-train/good sample.")
    names = [s.path.name for s in samples]
    if names != sorted(names):
        raise RuntimeError("Train/good samples are not in sorted file-name order.")
    model = artifacts.load_model(model_path, build_model())
    started = time.perf_counter()
    scores = np.array([compute_reconstruction_error(model, _sample_to_tensor(s, image_size)) for s in samples], dtype=np.float64)
    return {"names": names, "scores": scores, "model_sha256": actual["sha256"], "scoring_seconds": time.perf_counter() - started,
            "n": len(names)}


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------

def _fold_rows(scores: np.ndarray, splits: list[dict], policy: str) -> list[dict]:
    rows = []
    for s in splits:
        calib, held = scores[s["calibration"]], scores[s["held_out"]]
        thr = policy_threshold(policy, calib)
        fp = int((held > thr).sum())
        rows.append({"split": s["name"], "threshold": thr, "calibration_size": int(calib.size), "held_out_size": int(held.size),
                     "calibration_mean": float(calib.mean()), "calibration_std": float(calib.std()),
                     "held_out_false_positives": fp, "held_out_fpr": fp / held.size})
    return rows


def _threshold_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "cv": float(arr.std() / arr.mean()),
            "min": float(arr.min()), "max": float(arr.max())}


def run_study(names: list[str], scores: np.ndarray) -> dict:
    """Every scheme x policy, per-split rows, per-scheme and pooled statistics, and the full-pool thresholds."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    schemes = all_schemes(n)
    results: dict[str, dict] = {}
    for policy in policy_ids():
        per_scheme, pooled_thr, pooled_fpr = {}, [], []
        for scheme, splits in schemes.items():
            rows = _fold_rows(scores, splits, policy)
            thr = [r["threshold"] for r in rows]
            fpr = [r["held_out_fpr"] for r in rows]
            pooled_thr += thr
            pooled_fpr += fpr
            per_scheme[scheme] = {
                "folds": rows, "threshold": _threshold_stats(thr),
                "mean_held_out_fpr": float(np.mean(fpr)), "worst_fold_fpr": float(np.max(fpr)),
                "total_held_out_false_positives": int(sum(r["held_out_false_positives"] for r in rows)),
                "total_held_out_images": int(sum(r["held_out_size"] for r in rows)),
                "canonical_split": rows[0],
            }
        scheme_means = [v["threshold"]["mean"] for v in per_scheme.values()]
        results[policy] = {
            "policy": policy, "per_scheme": per_scheme,
            "pooled": {**_threshold_stats(pooled_thr), "mean_held_out_fpr": float(np.mean(pooled_fpr)),
                       "worst_fold_fpr": float(np.max(pooled_fpr)), "splits": len(pooled_thr)},
            "between_scheme_spread_of_mean_threshold": {"min": float(min(scheme_means)), "max": float(max(scheme_means)),
                                                        "relative_range": float((max(scheme_means) - min(scheme_means)) / np.mean(scheme_means))},
            "full_pool_threshold": policy_threshold(policy, scores),
            "full_pool_in_sample_false_positives": int((scores > policy_threshold(policy, scores)).sum()),
        }
    summary = {
        "n": n, "pool_score_distribution": describe(scores), "policies": results,
        "scheme_definitions": {name: {"splits": len(splits), "held_out": int(len(splits[0]["held_out"])),
                                      "calibration": int(len(splits[0]["calibration"]))} for name, splits in schemes.items()},
        "splits_sha256": splits_fingerprint(schemes), "split_audit": split_audit(n, schemes),
        "scores_sha256": hashlib.sha256(scores.tobytes()).hexdigest(), "file_list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "held_out_scores_are_in_sample_for_the_model": True,
        "declared_rule": DECLARED_RULE,
    }
    summary["selection"] = select_policy(results)
    return summary


# ---------------------------------------------------------------------------
# Selection (the pre-declared rule)
# ---------------------------------------------------------------------------

def select_policy(results: dict[str, dict]) -> dict:
    eligible = [p for p, r in results.items() if r["pooled"]["worst_fold_fpr"] <= PRIMARY_FPR + 1e-12]
    table = {p: {"worst_fold_fpr": r["pooled"]["worst_fold_fpr"], "cv": r["pooled"]["cv"], "eligible": p in eligible}
             for p, r in results.items()}
    if not eligible:
        return {"winner": None, "eligible": [], "equivalent_stability": [], "table": table, "message": NO_CANDIDATE_MESSAGE,
                "threshold": None, "reasoning": NO_CANDIDATE_MESSAGE}
    best_cv = min(results[p]["pooled"]["cv"] for p in eligible)
    stable = [p for p in eligible if results[p]["pooled"]["cv"] <= best_cv + CV_TIE]
    ordered = sorted(stable, key=lambda p: (round(results[p]["pooled"]["worst_fold_fpr"], 12), simplicity_rank(p), p))
    winner = ordered[0]
    reasoning = (f"{len(eligible)} of {len(results)} policies eligible (worst held-out FPR <= {PRIMARY_FPR:.0%}); smallest eligible CV "
                 f"{best_cv:.4f}; {len(stable)} within {CV_TIE} of it; lowest worst-fold FPR "
                 f"({results[winner]['pooled']['worst_fold_fpr']:.4f}), then simplicity, chose {winner}.")
    return {"winner": winner, "eligible": eligible, "equivalent_stability": stable, "ranking_of_equivalent": ordered, "table": table,
            "message": None, "threshold": results[winner]["full_pool_threshold"], "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def phase2_dir(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / PHASE2_DIRNAME


def reports_dir(category: str) -> Path:
    return phase2_dir(category) / "reports"


def lock_path(category: str) -> Path:
    return phase2_dir(category) / "threshold_lock.json"


def build_lock(category: str, model_path: Path, model_sha256: str, study: dict, phase1_reference: dict, timestamp: float,
               experiment_id: str) -> dict:
    sel = study["selection"]
    if sel["winner"] is None:
        raise RuntimeError(NO_CANDIDATE_MESSAGE + " Nothing to lock.")
    winner = study["policies"][sel["winner"]]
    return {
        "category": category, "experiment_id": experiment_id, "timestamp_unix": timestamp,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "source_model_path": str(model_path), "model_sha256": model_sha256,
        "threshold": sel["threshold"], "threshold_method": sel["winner"],
        "threshold_definition": "policy applied to all train/good reconstruction errors (full pool); score > threshold -> defective",
        "calibration_schemes": study["scheme_definitions"], "splits_sha256": study["splits_sha256"],
        "scores_sha256": study["scores_sha256"], "file_list_sha256": study["file_list_sha256"],
        "selection_rule": DECLARED_RULE, "selection_reasoning": sel["reasoning"],
        "selection_evidence": {"table": sel["table"], "eligible": sel["eligible"], "equivalent_stability": sel["equivalent_stability"],
                               "winner_pooled": winner["pooled"], "winner_per_scheme": {k: {"threshold": v["threshold"], "worst_fold_fpr": v["worst_fold_fpr"],
                                                                                       "mean_held_out_fpr": v["mean_held_out_fpr"]}
                                                                                   for k, v in winner["per_scheme"].items()}},
        "phase1_reference": phase1_reference,
        "final_test_data_used_for_selection": False,
    }


def lock_digest(lock: dict) -> str:
    """Canonical content hash of a lock, excluding the wall-clock fields (the file bytes differ only by those)."""
    body = {k: v for k, v in lock.items() if k not in ("timestamp_unix", "timestamp_utc")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def write_lock(path: Path, lock: dict) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing threshold lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    return file_hashes(path)


def compare_studies(a: dict, b: dict) -> dict:
    """Exact equality of two complete study runs (everything the second run must reproduce)."""
    canon = lambda x: json.loads(json.dumps(x, sort_keys=True))  # noqa: E731
    return {
        "scores_identical": a["scores_sha256"] == b["scores_sha256"],
        "file_list_identical": a["file_list_sha256"] == b["file_list_sha256"],
        "calibration_splits_identical": a["splits_sha256"] == b["splits_sha256"],
        "candidate_thresholds_and_fprs_identical": canon(a["policies"]) == canon(b["policies"]),
        "threshold_statistics_identical": canon({p: r["pooled"] for p, r in a["policies"].items()}) == canon({p: r["pooled"] for p, r in b["policies"].items()}),
        "selection_identical": canon(a["selection"]) == canon(b["selection"]),
        "selected_threshold_identical": a["selection"]["threshold"] == b["selection"]["threshold"],
    }



def study_from_model(category: str, model_path: Path, expected_sha256: str, image_size: tuple[int, int] = (128, 128)) -> dict:
    """Score train/good with the hash-verified frozen model, then run the complete study."""
    scored = score_train_good(category, model_path, expected_sha256, image_size)
    study = run_study(scored["names"], scored["scores"])
    study["scoring_seconds"] = scored["scoring_seconds"]
    study["model_sha256"] = scored["model_sha256"]
    study["train_good_scores"] = [float(x) for x in scored["scores"]]
    study["train_good_files"] = scored["names"]
    return json.loads(json.dumps(study, sort_keys=True))  # the exact form a report file round-trips to


def create_lock(category: str, model_path: Path, model_sha256: str, first: dict, second: dict, phase1_ref: dict,
                timestamp: float | None = None) -> dict:
    """Require the second, independent study run to reproduce the first exactly, then lock the single selected
    threshold. Returns {"locked": bool, ...}; when no candidate is eligible nothing is written and locked is False."""
    repro = compare_studies(first, second)
    if not all(repro.values()):
        raise RuntimeError(f"The calibration procedure is not reproducible across two runs; not locking: {repro}")
    if first["selection"]["winner"] is None:
        return {"locked": False, "message": NO_CANDIDATE_MESSAGE, "reproducibility": repro}
    if phase1_ref != first["phase1_reference_at_study_time"]:
        raise RuntimeError("A Phase 1 artifact changed since the study; not locking.")
    stamp = time.time() if timestamp is None else timestamp
    experiment_id = f"{EXPERIMENT_ID_PREFIX}-{category}-{first['splits_sha256'][:12]}"
    lock = build_lock(category, model_path, model_sha256, first, phase1_ref, stamp, experiment_id)
    lock_again = build_lock(category, model_path, model_sha256, second, phase1_ref, stamp, experiment_id)
    files = write_lock(lock_path(category), lock)
    lock_repro = {"lock_digest_first": lock_digest(lock), "lock_digest_second": lock_digest(lock_again),
                  "lock_digest_identical": lock_digest(lock) == lock_digest(lock_again),
                  "lock_content_identical_given_same_timestamp":
                      json.dumps(lock, indent=1, sort_keys=True) == json.dumps(lock_again, indent=1, sort_keys=True),
                  "lock_file": files}
    reports_dir(category).mkdir(parents=True, exist_ok=True)
    (reports_dir(category) / "lock_reproducibility.json").write_text(json.dumps({"study": repro, "lock": lock_repro}, indent=1), encoding="utf-8")
    return {"locked": True, "lock": lock, "reproducibility": repro, "lock_reproducibility": lock_repro}
