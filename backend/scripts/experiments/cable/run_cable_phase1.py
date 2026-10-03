"""Cable Phase 1 ConvAE baseline with a TRUE train/validation split of train/good.

Composes existing project functions only (no project code is modified):
  split      app.ai.training.validation_split.split_train_validation (fraction 0.8, seed 42)
  training   app.ai.training.phase3_train.train_anomaly_model_on_samples (ConvAutoencoder, Adam, MSE)
  config     app.ai.evaluation.category_phase1.phase1_config (128x128, bs 8, 15 epochs, lr 1e-3, seed 42, cpu)
  score      app.ai.evaluation.category_phase1._errors -> evaluate.compute_reconstruction_error (per-image MSE)
  threshold  app.ai.evaluation.threshold.compute_threshold (mean + 3*std, population std) on VALIDATION errors
  metrics    app.ai.evaluation.category_phase3.metrics_from_predictions / defect_type_recall

Order: verify counts -> split -> train -> threshold from validation -> LOCK -> reproducibility retrain (temp dir,
no test access) -> final test scored exactly once -> report.
"""

import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.ai.evaluation.category_phase1 import (  # noqa: E402
    _auroc, _errors, _overlap, describe, file_hashes, gate_classification, phase1_config, phase1_dir,
    preprocessing_description, verify_dataset_counts,
)
from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions, predictions_at_threshold  # noqa: E402
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold  # noqa: E402
from app.ai.training import artifacts  # noqa: E402
from app.ai.training.dataset import discover_test_samples, discover_train_samples  # noqa: E402
from app.ai.training.model import build_model  # noqa: E402
from app.ai.training.phase3_train import train_anomaly_model_on_samples  # noqa: E402
from app.ai.training.validation_split import split_train_validation  # noqa: E402

CATEGORY = "cable"
EXPECTED = {"train_good": 224, "test_good": 58, "test_defective": 92, "test_total": 150}
TRAINING_FRACTION, SPLIT_SEED = 0.8, 42


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """95% Wilson score interval for k successes in n trials (standard formula; no project utility exists)."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def other_artifact_hashes() -> dict:
    own = phase1_dir(CATEGORY).resolve()
    return {str(p.relative_to(artifacts.ARTIFACTS_ROOT)): file_hashes(p)["sha256"]
            for p in sorted(artifacts.ARTIFACTS_ROOT.rglob("*"))
            if p.is_file() and p.suffix in {".pt", ".pth", ".json"} and own not in p.resolve().parents}


def train_and_threshold(split, config, model_path: Path) -> dict:
    training, model = train_anomaly_model_on_samples(split.training, config, model_path)
    scored = artifacts.load_model(model_path, build_model(latent_channels=config.model.latent_channels))
    reload_ok = all(torch.equal(model.state_dict()[k], v) for k, v in scored.state_dict().items())
    val_errors, _, _ = _errors(scored, split.validation, config.image_size)
    train_errors, _, _ = _errors(scored, split.training, config.image_size)
    threshold = compute_threshold(val_errors, k=DEFAULT_THRESHOLD_K)
    return {"training": training, "model": scored, "reload_ok": reload_ok, "val_errors": val_errors,
            "train_errors": train_errors, "threshold": threshold, "artifact": file_hashes(model_path)}


def main() -> None:
    out_dir = phase1_dir(CATEGORY)
    model_path = out_dir / "autoencoder.pt"
    lock_path = out_dir / "threshold_lock.json"
    report_path = out_dir / "reports" / "phase1_report.json"
    if out_dir.exists():
        sys.exit(f"STOP: {out_dir} already exists; refusing to overwrite.")

    counts = verify_dataset_counts(CATEGORY, EXPECTED)
    print("Dataset counts verified:", json.dumps(counts))
    before = other_artifact_hashes()

    config = phase1_config(CATEGORY)
    all_train = discover_train_samples(CATEGORY)
    split = split_train_validation(all_train, training_fraction=TRAINING_FRACTION, seed=SPLIT_SEED)
    print(f"Split: {len(split.training)} training / {len(split.validation)} validation (seed {split.seed})")

    # ---- RUN 1: real artifact ----
    r1 = train_and_threshold(split, config, model_path)
    val_stats = describe(r1["val_errors"])
    val_fp = int((np.asarray(r1["val_errors"]) > r1["threshold"]).sum())
    lock = {
        "status": "MODEL LOCKED / THRESHOLD LOCKED",
        "category": CATEGORY, "artifact_path": str(model_path), "artifact_size_bytes": r1["artifact"]["size_bytes"],
        "model_md5": r1["artifact"]["md5"], "model_sha256": r1["artifact"]["sha256"],
        "threshold": r1["threshold"], "threshold_k": DEFAULT_THRESHOLD_K,
        "threshold_policy": "mean + 3 * std (population) of per-image reconstruction MSE on the VALIDATION normal subset",
        "validation_error_mean": val_stats["mean"], "validation_error_std": val_stats["std"],
        "split": {"source": "train/good only", "training_fraction": TRAINING_FRACTION, "seed": SPLIT_SEED,
                  "training": len(split.training), "validation": len(split.validation),
                  "validation_files": sorted(s.path.name for s in split.validation)},
        "locked_at_unix": time.time(), "final_test_images_listed_before_lock": False,
    }
    lock_path.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    print(f"MODEL LOCKED  sha256={lock['model_sha256']} md5={lock['model_md5']}")
    print(f"THRESHOLD LOCKED  {lock['threshold']!r}  (val mean {val_stats['mean']:.6g}, std {val_stats['std']:.6g})")

    # ---- RUN 2: reproducibility retrain in a temp dir; the final test set is NOT touched ----
    with tempfile.TemporaryDirectory() as tmp:
        r2 = train_and_threshold(split, config, Path(tmp) / "autoencoder.pt")
    repro = {
        "model_sha256_identical": r1["artifact"]["sha256"] == r2["artifact"]["sha256"],
        "model_md5_identical": r1["artifact"]["md5"] == r2["artifact"]["md5"],
        "loss_history_identical": r1["training"].loss_history == r2["training"].loss_history,
        "validation_errors_identical": r1["val_errors"] == r2["val_errors"],
        "threshold_identical": r1["threshold"] == r2["threshold"],
        "run2_sha256": r2["artifact"]["sha256"], "note": "run 2 never listed or scored final-test images",
    }
    print("Reproducibility (pre-test):", json.dumps(repro))

    # ---- FINAL TEST: listed and scored exactly once, after the lock ----
    assert file_hashes(model_path)["sha256"] == lock["model_sha256"]
    test_listed_at = time.time()
    test_samples = discover_test_samples(CATEGORY)
    test_errors, _, _ = _errors(r1["model"], test_samples, config.image_size)
    predictions = predictions_at_threshold(test_samples, test_errors, lock["threshold"])
    metrics = metrics_from_predictions(predictions)
    good = [p["reconstruction_error"] for p in predictions if p["label"] == 0]
    bad = [p["reconstruction_error"] for p in predictions if p["label"] == 1]
    per_defect = {k: {**v, "missed": v["total"] - v["detected"]} for k, v in sorted(defect_type_recall(predictions).items())}
    by_type_scores: dict[str, list[float]] = {}
    for p in predictions:
        if p["label"] == 1:
            by_type_scores.setdefault(p["defect_type"], []).append(p["reconstruction_error"])

    # ---- post-test integrity (no re-scoring) ----
    recomputed = metrics_from_predictions(predictions_at_threshold(test_samples, test_errors, lock["threshold"]))
    after = other_artifact_hashes()
    train_paths = {s.path for s in split.training}
    val_paths = {s.path for s in split.validation}
    test_paths = {s.path for s in test_samples}
    sha = lambda ps: {hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}  # noqa: E731
    leakage = {
        "defective_test_images_not_used_for_training": all(s.split == "train" and s.defect_type == "good" for s in split.training)
        and not (train_paths & test_paths),
        "final_test_not_used_for_threshold_selection": set(lock["split"]["validation_files"]) == {p.name for p in val_paths}
        and not (val_paths & test_paths),
        "final_test_not_used_for_model_selection": "single fixed configuration; no candidates compared",
        "no_test_derived_threshold_tuning": lock["locked_at_unix"] <= test_listed_at and lock_path.stat().st_mtime <= test_listed_at,
        "no_final_test_result_used_to_modify_model": file_hashes(model_path)["sha256"] == lock["model_sha256"],
        "validation_and_final_test_distinct": not (val_paths & test_paths) and not (sha(val_paths) & sha(test_paths)),
        "training_and_validation_disjoint": not (train_paths & val_paths),
        "training_and_test_contents_disjoint": not (sha(train_paths) & sha(test_paths)),
        "saved_artifact_is_what_was_scored": r1["reload_ok"],
        "other_category_artifacts_unchanged": before == after,
    }
    fpr = metrics["false_defect_detection_rate"]
    report = {
        "category": CATEGORY, "counts": counts, "lock": lock,
        "config": {"model": "ConvAutoencoder (app.ai.training.model.build_model)", "latent_channels": config.model.latent_channels,
                   "image_size": list(config.image_size), "batch_size": config.batch_size, "epochs": config.epochs,
                   "learning_rate": config.learning_rate, "seed": config.seed, "device": config.device,
                   "optimizer": "Adam", "loss": "MSELoss", "determinism": "torch.manual_seed(42) via app.ai.training.train.set_seed; DataLoader shuffle, num_workers=0"},
        "preprocessing": preprocessing_description(config.image_size),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
        "parameter_count": int(sum(p.numel() for p in r1["model"].parameters())),
        "training": {"num_training_images": r1["training"].num_training_images, "epochs": r1["training"].epochs,
                     "final_loss": r1["training"].final_loss, "lowest_loss": min(r1["training"].loss_history),
                     "loss_history": r1["training"].loss_history, "duration_seconds": r1["training"].duration_seconds},
        "artifact": r1["artifact"],
        "validation": {"stats": val_stats, "false_positives": val_fp, "of": len(r1["val_errors"]),
                       "fpr": val_fp / len(r1["val_errors"]), "training_subset_stats": describe(r1["train_errors"])},
        "final_test": {"metrics": metrics, "fpr_wilson95": wilson(metrics["false_positives"], len(good)),
                       "recall_wilson95": wilson(metrics["true_positives"], len(bad)),
                       "auroc": _auroc(good, bad), "test_good_stats": describe(good), "test_defective_stats": describe(bad),
                       "by_type_stats": {k: describe(v) for k, v in sorted(by_type_scores.items())},
                       "overlap": _overlap(good, bad, lock["threshold"]),
                       "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
                       "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0]},
        "per_defect": per_defect,
        "gate": gate_classification(metrics["recall"], metrics["f1_score"], fpr),
        "leakage_audit": leakage,
        "reproducibility": {**repro, "metrics_recomputed_from_stored_scores_identical": recomputed == metrics,
                            "final_test_scoring_passes": 1},
        "predictions": predictions,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("predictions", "preprocessing")}, indent=1, default=str))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
