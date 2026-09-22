"""Genuinely held-out ConvAutoencoder baseline for one category (Phase 3), stage 1: split -> train -> validate ->
select -> LOCK. This module cannot score or decode a final-test image and never imports test discovery (asserted from
source by tests); the protected test set is touched only by app.ai.evaluation.category_phase3_heldout_final, after the
threshold lock has been written to disk and the model hash re-verified.

Built from the project's existing pieces, not reimplemented:
  split        app.ai.training.validation_split.split_train_validation   (sorted names, 0.8, numpy default_rng(42))
  training     app.ai.training.phase3_train.train_anomaly_model_on_samples  (ConvAutoencoder, MSE, Adam; only the
               samples it is handed ever reach the DataLoader)
  errors       app.ai.evaluation.threshold_experiments.compute_errors_for_samples
  candidates   app.ai.evaluation.threshold_experiments.generate_candidates (mean+K*std, K=1..3, population std;
               P95/P97/P98/P99 of the validation errors)
  config       app.ai.evaluation.category_phase3.phase3_training_config (128x128, batch 8, 15 epochs, lr 1e-3, seed 42, CPU)

=====================================================================================================================
PRE-DECLARED THRESHOLD-SELECTION RULE  (written before the model was trained; not to be altered after any result)
=====================================================================================================================
Data: the 245 train/good images are split deterministically into 196 training and 49 validation images; the
validation images never reach the training DataLoader. Every candidate threshold is computed from the 49 validation
reconstruction errors (established Phase 3 framework) and a validation image is a false positive iff score > threshold.
Candidates: mean + K*std (population std) for K in {1.0, 1.5, 2.0, 2.5, 3.0}; percentiles P95, P97, P98, P99.
  1. ELIGIBLE iff validation FPR <= 10%.  If nothing is eligible: nothing is selected or locked, the report says
     "No threshold candidate satisfied the predeclared validation FPR requirement.", and the final test is NOT run.
  2. Among eligible candidates prefer the LOWER validation FPR.
  3. "Effectively tied" means an identical validation false-positive count; ties go to the simpler mean+K*std family.
  4. Remaining ties use the existing deterministic Phase 3 policy: the lowest threshold, then candidate order.
The locked threshold is that candidate's threshold. The validation FPR is validation evidence only: it is measured on
images the threshold was itself computed from, so it is optimistic and is NOT an estimate of final-test FPR.
No final-test image, label or score enters any step of training, threshold selection or locking.
=====================================================================================================================
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from app.ai.evaluation.category_phase1 import describe, file_hashes, verify_dataset_counts
from app.ai.evaluation.category_phase3 import phase3_training_config
from app.ai.evaluation.threshold_experiments import METHOD_MEAN_STD, ThresholdCandidate, compute_errors_for_samples, generate_candidates
from app.ai.training import artifacts
from app.ai.training.dataset import discover_train_samples
from app.ai.training.model import build_model
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.schemas import DatasetSample
from app.ai.training.validation_split import TrainValidationSplit, split_train_validation

PHASE = "phase3_validation"
TRAINING_FRACTION = 0.8
SPLIT_SEED = 42
MAX_VALIDATION_FPR = 0.10
EXPERIMENT_ID_PREFIX = "phase3-heldout-convae"
NO_CANDIDATE_MESSAGE = "No threshold candidate satisfied the predeclared validation FPR requirement."

DECLARED_RULE = {
    "eligibility": f"validation FPR <= {MAX_VALIDATION_FPR:.0%} (validation image is a false positive iff score > threshold)",
    "preference": ["lower validation FPR", "simpler mean+K*std family when the false-positive counts are identical",
                   "existing deterministic Phase 3 policy: lowest threshold, then candidate order"],
    "candidates": "mean+K*std (population std) K=1,1.5,2,2.5,3; percentiles P95,P97,P98,P99; all computed from the validation errors",
    "if_none_eligible": NO_CANDIDATE_MESSAGE + " (nothing locked; final test not run)",
    "validation_fpr_note": "measured on the images the threshold was computed from; not an estimate of final-test FPR",
}


def phase3_dir(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / PHASE


def model_path(category: str) -> Path:
    return phase3_dir(category) / "autoencoder.pt"


def lock_path(category: str) -> Path:
    return phase3_dir(category) / "threshold_lock.json"


def reports_dir(category: str) -> Path:
    return phase3_dir(category) / "reports"


def validation_report_path(category: str) -> Path:
    return reports_dir(category) / "validation_stage.json"


def candidate_id(c: ThresholdCandidate) -> str:
    return f"mean_std_{c.parameter:g}" if c.method == METHOD_MEAN_STD else f"percentile_{c.parameter:g}"


# ---------------------------------------------------------------------------
# Split, fingerprints, prior-phase artifact references
# ---------------------------------------------------------------------------

def make_split(samples: list[DatasetSample]) -> TrainValidationSplit:
    return split_train_validation(samples, training_fraction=TRAINING_FRACTION, seed=SPLIT_SEED)


def split_fingerprint(split: TrainValidationSplit) -> dict:
    train_names = [s.path.name for s in split.training]
    val_names = [s.path.name for s in split.validation]
    return {"training_files_sha256": hashlib.sha256("\n".join(train_names).encode()).hexdigest(),
            "validation_files_sha256": hashlib.sha256("\n".join(val_names).encode()).hexdigest(),
            "training_count": len(train_names), "validation_count": len(val_names), "seed": split.seed,
            "training_fraction": split.training_fraction}


def dataset_fingerprint(samples: list[DatasetSample], counts: dict) -> str:
    """Content fingerprint of the calibration source (every train/good file's SHA-256) plus the verified counts."""
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.path.name}:{hashlib.sha256(s.path.read_bytes()).hexdigest()}\n".encode())
    h.update(json.dumps({k: counts[k] for k in ("train_good", "test_good", "test_defective", "test_total")}, sort_keys=True).encode())
    return h.hexdigest()


def prior_phase_reference(category: str) -> dict:
    """SHA-256 of every file under this category's Phase 1 and Phase 2 directories (never modified by Phase 3)."""
    root = artifacts.ARTIFACTS_ROOT / category
    ref = {}
    for sub in ("phase1_baseline", "phase2_threshold"):
        base = root / sub
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    ref[str(path.relative_to(root)).replace("\\", "/")] = file_hashes(path)["sha256"]
    return ref


# ---------------------------------------------------------------------------
# Pre-declared threshold selection (pure: validation errors only)
# ---------------------------------------------------------------------------

def _simplicity(c: ThresholdCandidate) -> int:
    return 0 if c.method == METHOD_MEAN_STD else 1


def select_candidate(candidates: list[ThresholdCandidate], validation_errors) -> dict:
    errors = np.asarray(list(validation_errors), dtype=np.float64)
    n = int(errors.size)
    rows = []
    for index, c in enumerate(candidates):
        fp = int((errors > c.threshold).sum())
        rows.append({"id": candidate_id(c), "method": c.method, "parameter": c.parameter, "threshold": c.threshold,
                     "validation_false_positives": fp, "validation_size": n, "validation_fpr": fp / n,
                     "eligible": fp / n <= MAX_VALIDATION_FPR + 1e-12, "order": index})
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return {"winner": None, "candidates": rows, "message": NO_CANDIDATE_MESSAGE, "reasoning": NO_CANDIDATE_MESSAGE}
    by_id = {candidate_id(c): c for c in candidates}
    winner = min(eligible, key=lambda r: (r["validation_false_positives"], _simplicity(by_id[r["id"]]), r["threshold"], r["order"]))
    reasoning = (f"{len(eligible)} of {len(rows)} candidates eligible (validation FPR <= {MAX_VALIDATION_FPR:.0%}); lowest validation FPR "
                 f"is {winner['validation_fpr']:.2%} ({winner['validation_false_positives']}/{n}); ties resolved by simpler mean+K*std family, "
                 f"then lowest threshold; chose {winner['id']}.")
    return {"winner": winner["id"], "winner_row": winner, "candidates": rows, "message": None, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

def config_dict(config) -> dict:
    return {"image_size": list(config.image_size), "batch_size": config.batch_size, "epochs": config.epochs,
            "learning_rate": config.learning_rate, "seed": config.seed, "device": config.device,
            "latent_channels": config.model.latent_channels, "loss": "MSELoss", "optimizer": "Adam", "augmentation": "none"}


def _digest_body(lock: dict) -> dict:
    volatile = ("timestamp_unix", "timestamp_utc", "lock_digest", "model_path", "validation_report_path")
    return {k: v for k, v in lock.items() if k not in volatile}


def lock_digest(lock: dict) -> str:
    """Canonical content hash of a lock, excluding wall-clock and location fields."""
    return hashlib.sha256(json.dumps(_digest_body(lock), sort_keys=True).encode()).hexdigest()


def build_lock(category: str, path: Path, model_hashes: dict, split: TrainValidationSplit, selection: dict, counts: dict,
               dataset_fp: str, config, prior: dict, validation_errors_sha256: str, timestamp: float) -> dict:
    fp = split_fingerprint(split)
    lock = {
        "category": category, "phase": PHASE,
        "experiment_id": f"{EXPERIMENT_ID_PREFIX}-{category}-{fp['training_files_sha256'][:12]}",
        "timestamp_unix": timestamp, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "model_path": str(path), "model_sha256": model_hashes["sha256"], "model_md5": model_hashes["md5"],
        "model_size_bytes": model_hashes["size_bytes"],
        "training_image_count": fp["training_count"], "validation_image_count": fp["validation_count"], "split_seed": fp["seed"],
        "split_fingerprint": fp, "dataset_fingerprint_sha256": dataset_fp, "dataset_counts": counts,
        "training_config": config_dict(config),
        "threshold": selection["winner_row"]["threshold"], "threshold_method": selection["winner"],
        "selected_candidate": selection["winner_row"], "all_candidates": selection["candidates"],
        "selection_rule": DECLARED_RULE, "selection_reasoning": selection["reasoning"],
        "validation_errors_sha256": validation_errors_sha256,
        "prior_phase_artifacts": prior, "final_test_data_used_for_selection": False,
    }
    lock["lock_digest"] = lock_digest(lock)
    return lock


def write_lock(path: Path, lock: dict) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing threshold lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    return file_hashes(path)


# ---------------------------------------------------------------------------
# Stage 1: split -> train -> validate -> select -> lock
# ---------------------------------------------------------------------------

def train_validate_lock(category: str, target_model_path: Path, expected_counts: dict, target_lock_path: Path | None = None,
                        config=None) -> dict:
    """Never reads, decodes or scores a final-test image. Writes the model, and the lock if a candidate is eligible."""
    target_model_path = Path(target_model_path)
    started = time.perf_counter()
    counts = verify_dataset_counts(category, expected_counts)  # lists file names/sizes only; raises before training on a mismatch
    if target_model_path.exists() or (target_lock_path is not None and Path(target_lock_path).exists()):
        raise FileExistsError(f"Refusing to overwrite an existing Phase 3 artifact: {target_model_path} / {target_lock_path}")
    config = config or phase3_training_config(category)
    events = []

    prior = prior_phase_reference(category)
    samples = discover_train_samples(category)
    split = make_split(samples)
    events.append("split")
    train_paths = {s.path for s in split.training}
    val_paths = {s.path for s in split.validation}
    if train_paths & val_paths or len(train_paths) + len(val_paths) != len(samples):
        raise AssertionError("Training/validation split overlaps or does not cover the pool.")
    dataset_fp = dataset_fingerprint(samples, counts)

    training, model = train_anomaly_model_on_samples(split.training, config, target_model_path)
    events.append("trained_on_training_subset_only")
    hashes = file_hashes(target_model_path)

    scored_model = artifacts.load_model(target_model_path, build_model(latent_channels=config.model.latent_channels))
    reload_matches = all((model.state_dict()[k] == v).all().item() for k, v in scored_model.state_dict().items())
    validation_errors = compute_errors_for_samples(scored_model, split.validation, config.image_size)
    training_errors = compute_errors_for_samples(scored_model, split.training, config.image_size)
    events.append("validation_and_training_errors_computed")

    candidates = generate_candidates(validation_errors)
    selection = select_candidate(candidates, validation_errors)
    events.append("selection_made")
    errors_sha = hashlib.sha256(np.asarray(validation_errors, dtype=np.float64).tobytes()).hexdigest()

    lock, lock_files = None, None
    if selection["winner"] is not None:
        lock = build_lock(category, target_model_path, hashes, split, selection, counts, dataset_fp, config, prior, errors_sha, time.time())
        if target_lock_path is not None:
            lock_files = write_lock(target_lock_path, lock)
        events.append("threshold_locked")
    hash_after_lock = file_hashes(target_model_path)["sha256"]

    return {
        "category": category, "counts": counts, "config": config_dict(config),
        "split": split_fingerprint(split), "dataset_fingerprint_sha256": dataset_fp,
        "training_files": [s.path.name for s in split.training], "validation_files": [s.path.name for s in split.validation],
        "training": {"epochs": training.epochs, "device": training.device, "num_training_images": training.num_training_images,
                     "final_loss": training.final_loss, "loss_history": training.loss_history, "duration_seconds": training.duration_seconds},
        "artifact": hashes, "reload_matches_trained_weights": bool(reload_matches),
        "validation_errors": validation_errors, "training_errors": training_errors, "validation_errors_sha256": errors_sha,
        "distributions": {"training": describe(training_errors), "validation": describe(validation_errors)},
        "selection": selection, "lock": lock, "lock_file": lock_files, "events": events,
        "model_hash_verified_after_lock": hash_after_lock == hashes["sha256"],
        "prior_phase_artifacts_at_start": prior, "prior_phase_artifacts_at_end": prior_phase_reference(category),
        "stage_seconds": time.perf_counter() - started,
    }


def compare_stages(a: dict, b: dict) -> dict:
    """Everything the second, independent full run must reproduce (no final-test data is involved)."""
    return {
        "split_identical": a["split"] == b["split"] and a["training_files"] == b["training_files"] and a["validation_files"] == b["validation_files"],
        "dataset_fingerprint_identical": a["dataset_fingerprint_sha256"] == b["dataset_fingerprint_sha256"],
        "loss_history_identical": a["training"]["loss_history"] == b["training"]["loss_history"],
        "loss_history_max_abs_difference": max(abs(x - y) for x, y in zip(a["training"]["loss_history"], b["training"]["loss_history"])),
        "model_sha256_identical": a["artifact"]["sha256"] == b["artifact"]["sha256"],
        "model_md5_identical": a["artifact"]["md5"] == b["artifact"]["md5"],
        "validation_scores_identical": a["validation_errors"] == b["validation_errors"],
        "validation_scores_max_abs_difference": max(abs(x - y) for x, y in zip(a["validation_errors"], b["validation_errors"])),
        "training_scores_identical": a["training_errors"] == b["training_errors"],
        "threshold_candidates_identical": a["selection"]["candidates"] == b["selection"]["candidates"],
        "selected_candidate_identical": a["selection"]["winner"] == b["selection"]["winner"],
        "selected_threshold_identical": (a["lock"] or {}).get("threshold") == (b["lock"] or {}).get("threshold"),
        "lock_digest_identical": (a["lock"] or {}).get("lock_digest") == (b["lock"] or {}).get("lock_digest") and a["lock"] is not None,
    }
