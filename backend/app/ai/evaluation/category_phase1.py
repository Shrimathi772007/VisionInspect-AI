"""Phase 1 baseline for one MVTec category: the existing ConvAutoencoder, trained once on ALL train/good images
with the established configuration, a threshold of mean + 3*std of the TRAIN reconstruction errors, and one final
test scoring.

Reuses (does not reimplement) the project's pieces:
  * discovery/preprocessing  app.ai.training.dataset (discover_*_samples, load_sample -> process_image)
  * model / loss / optimizer app.ai.training.phase3_train.train_anomaly_model_on_samples (a line-for-line mirror of
                             app.ai.training.train.train_anomaly_model that accepts an explicit sample list and an
                             explicit save path; with all train/good samples it is the Phase 1 training procedure)
  * threshold                app.ai.evaluation.threshold.compute_threshold (K=3, the Phase 1 rule)
  * metrics                  app.ai.evaluation.category_phase3.metrics_from_predictions et al.

Order of operations is fixed and enforced in code:
  1. verify dataset counts (raise before training on a mismatch);
  2. refuse to overwrite an existing artifact;
  3. train on train/good only;
  4. score train/good -> threshold, then write a hash-locked lock record (model SHA-256 + threshold);
  5. only then list and score the test images; the threshold is applied as-is.

Nothing here selects, tunes or compares thresholds, and no Phase 2/3/4 machinery is used.
"""

import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions, predictions_at_threshold
from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, _sample_to_tensor, compute_reconstruction_error
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold
from app.ai.training import artifacts
from app.ai.training.dataset import discover_test_samples, discover_train_samples
from app.ai.training.model import build_model
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.schemas import TrainingConfig

PHASE1_DIRNAME = "phase1_baseline"
PERCENTILES = (5, 25, 50, 75, 90, 95, 99)


def phase1_config(category: str) -> TrainingConfig:
    """The established baseline configuration, spelled out (nothing here is tuned)."""
    config = TrainingConfig(category=category, image_size=DEFAULT_IMAGE_SIZE, batch_size=8, epochs=15,
                            learning_rate=1e-3, seed=42, device="cpu")
    return config


def phase1_dir(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / PHASE1_DIRNAME


def default_model_path(category: str) -> Path:
    return phase1_dir(category) / "autoencoder.pt"


def file_hashes(path: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def dataset_fingerprint(samples, counts: dict) -> str:
    """Content fingerprint of the training data (every train/good file's SHA-256, in sorted order) plus the counts."""
    h = hashlib.sha256()
    for sample in samples:
        h.update(f"{sample.path.name}:{hashlib.sha256(sample.path.read_bytes()).hexdigest()}\n".encode())
    h.update(json.dumps({k: counts[k] for k in ("train_good", "test_good", "test_defective", "test_total")}, sort_keys=True).encode())
    return h.hexdigest()


def gate_classification(recall: float, f1: float, fpr: float) -> str:
    """Project evidence gates (interpretation only): EXCELLENT / GOOD / ACCEPTABLE / NOT PRODUCTION READY."""
    if recall >= 0.90 and f1 >= 0.85 and fpr <= 0.10:
        return "EXCELLENT"
    if recall >= 0.85 and f1 >= 0.80 and fpr <= 0.10:
        return "GOOD"
    if recall >= 0.75 and f1 >= 0.70 and fpr <= 0.15:
        return "ACCEPTABLE"
    return "NOT PRODUCTION READY"


def describe(values) -> dict:
    """count/mean/std(population)/median/min/max plus the requested percentiles."""
    arr = np.asarray(list(values), dtype=np.float64)
    out = {"count": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)),
           "min": float(arr.min()), "max": float(arr.max())}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def verify_dataset_counts(category: str, expected: dict) -> dict:
    """Actual filesystem counts; raises BEFORE any training when they differ from `expected`."""
    train = discover_train_samples(category)
    test = discover_test_samples(category)
    by_type: dict[str, int] = {}
    for sample in test:
        by_type[sample.defect_type] = by_type.get(sample.defect_type, 0) + 1
    actual = {"train_good": len(train), "test_good": by_type.get("good", 0),
              "test_defective": len(test) - by_type.get("good", 0), "test_total": len(test),
              "test_by_type": dict(sorted(by_type.items()))}
    mismatches = {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v}
    if mismatches:
        raise ValueError(f"Dataset counts differ from the expected counts: {mismatches}")
    return actual


def preprocessing_description(image_size: tuple[int, int]) -> dict:
    """What app.ai.preprocessing.pipeline actually does (verified against its source by a test)."""
    return {"loader": "cv2.imread(IMREAD_COLOR) -> BGR uint8", "color_conversion": "BGR -> RGB (cv2.cvtColor)",
            "resize": f"cv2.resize to {image_size[0]}x{image_size[1]} (WxH), interpolation=INTER_AREA",
            "float_conversion": "astype(float32)", "normalization": "divide by 255.0 -> [0, 1] (no mean/std standardization)",
            "tensor": "HWC float32 -> CHW tensor via torch.from_numpy(...).permute(2, 0, 1).contiguous()",
            "augmentation": "none"}


def _errors(model: torch.nn.Module, samples, image_size) -> tuple[list[float], list[float], list[float]]:
    """(errors, preprocessing ms, inference ms) - preprocessing and inference are timed separately."""
    errors, prep_ms, inf_ms = [], [], []
    for sample in samples:
        t0 = time.perf_counter()
        tensor = _sample_to_tensor(sample, image_size)
        t1 = time.perf_counter()
        errors.append(compute_reconstruction_error(model, tensor))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t0) * 1000)
        inf_ms.append((t2 - t1) * 1000)
    return errors, prep_ms, inf_ms


def _auroc(good: list[float], bad: list[float]) -> float:
    return float(roc_auc_score([0] * len(good) + [1] * len(bad), list(good) + list(bad)))


def _overlap(good: list[float], bad: list[float], threshold: float) -> dict:
    g, b = np.asarray(good), np.asarray(bad)
    lo, hi = max(g.min(), b.min()), min(g.max(), b.max())
    return {
        "good_range": [float(g.min()), float(g.max())], "defective_range": [float(b.min()), float(b.max())],
        "overlap_interval": [float(lo), float(hi)] if lo <= hi else None,
        "good_images_inside_overlap": int(((g >= lo) & (g <= hi)).sum()) if lo <= hi else 0,
        "defective_images_inside_overlap": int(((b >= lo) & (b <= hi)).sum()) if lo <= hi else 0,
        "defective_images_at_or_below_good_median": int((b <= np.median(g)).sum()),
        "defective_images_at_or_below_good_max": int((b <= g.max()).sum()),
        "good_images_at_or_above_defective_min": int((g >= b.min()).sum()),
        "good_above_threshold": int((g > threshold).sum()), "defective_above_threshold": int((b > threshold).sum()),
    }


def run_phase1(category: str, model_path: Path, expected_counts: dict, *, lock_path: Path | None = None) -> dict:
    """Train, threshold, lock, then score the test set once. Returns a complete, JSON-serializable record."""
    model_path = Path(model_path)
    started = time.perf_counter()
    counts = verify_dataset_counts(category, expected_counts)
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact: {model_path}")

    config = phase1_config(category)
    train_samples = discover_train_samples(category)

    # ---- training (train/good only) ----
    training, model = train_anomaly_model_on_samples(train_samples, config, model_path)
    artifact = file_hashes(model_path)

    # ---- model reload timing + verification that the saved file is what gets scored ----
    t0 = time.perf_counter()
    scored_model = artifacts.load_model(model_path, build_model(latent_channels=config.model.latent_channels))
    model_load_ms = (time.perf_counter() - t0) * 1000
    in_memory_state = {k: v.clone() for k, v in model.state_dict().items()}
    reload_matches_trained = all(torch.equal(in_memory_state[k], v) for k, v in scored_model.state_dict().items())

    # ---- threshold from TRAIN errors only, then lock ----
    train_errors, train_prep_ms, train_inf_ms = _errors(scored_model, train_samples, config.image_size)
    threshold = compute_threshold(train_errors, k=DEFAULT_THRESHOLD_K)
    train_stats = describe(train_errors)
    lock = {"category": category, "model_sha256": artifact["sha256"], "threshold_k": DEFAULT_THRESHOLD_K, "threshold": threshold,
            "train_error_mean": train_stats["mean"], "train_error_std": train_stats["std"],
            "locked_at_unix": time.time(), "locked_before_test_images_were_listed": True}
    if lock_path is not None:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        Path(lock_path).write_text(json.dumps(lock, indent=1), encoding="utf-8")

    # ---- final test: first touched HERE ----
    test_listed_at = time.time()
    eval_started = time.perf_counter()
    test_samples = discover_test_samples(category)
    test_errors, prep_ms, inf_ms = _errors(scored_model, test_samples, config.image_size)
    eval_total_ms = (time.perf_counter() - eval_started) * 1000
    predictions = predictions_at_threshold(test_samples, test_errors, threshold)
    metrics = metrics_from_predictions(predictions)

    good = [p["reconstruction_error"] for p in predictions if p["label"] == 0]
    bad = [p["reconstruction_error"] for p in predictions if p["label"] == 1]
    by_type: dict[str, dict] = {}
    for p in predictions:
        if p["label"] == 1:
            by_type.setdefault(p["defect_type"], []).append(p["reconstruction_error"])
    recall_by_type = defect_type_recall(predictions)
    per_defect = {}
    for name in sorted(by_type):
        scores = by_type[name]
        entry = recall_by_type[name]
        per_defect[name] = {"total": entry["total"], "detected": entry["detected"], "missed": entry["total"] - entry["detected"],
                            "recall": entry["recall"], "mean": statistics.fmean(scores), "median": statistics.median(scores),
                            "min": min(scores), "max": max(scores)}

    train_paths = {s.path for s in train_samples}
    test_paths = {s.path for s in test_samples}
    train_content = {hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths}
    test_content = {hashlib.sha256(p.read_bytes()).hexdigest() for p in test_paths}
    leakage = {
        "1_only_train_good_used_for_training": all(s.split == "train" and s.defect_type == "good" for s in train_samples)
        and training.num_training_images == len(train_samples),
        "2_no_test_good_in_training": not any(p in train_paths for p in test_paths if p.parent.name == "good"),
        "3_no_defective_test_image_in_training": not any(p in train_paths for s in test_samples if s.label == 1 for p in [s.path]),
        "4_no_defect_mask_used": not any("ground_truth" in str(p) for p in train_paths | test_paths),
        "5_threshold_computed_from_train_errors_only": abs(threshold - (statistics.fmean(train_errors) + 3 * float(np.std(train_errors)))) < 1e-15,
        "7_threshold_locked_before_test_images_were_scored": lock["locked_at_unix"] <= test_listed_at
        and (lock_path is None or Path(lock_path).stat().st_mtime <= test_listed_at),
        "8_train_and_test_paths_disjoint": not (train_paths & test_paths),
        "8b_train_and_test_contents_disjoint": not (train_content & test_content),
        "saved_artifact_is_what_was_scored": reload_matches_trained,
    }

    return {
        "category": category,
        "counts": counts,
        "config": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items() if k != "model"},
                   "model": asdict(config.model), "loss": "MSELoss", "optimizer": "Adam", "threshold_rule": f"mean + {DEFAULT_THRESHOLD_K:g} * std (train/good reconstruction errors, population std)"},
        "preprocessing": preprocessing_description(config.image_size),
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                        "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "cpu_count": os.cpu_count(),
                        "processor": platform.processor(), "platform": platform.platform()},
        "dataset_fingerprint_sha256": dataset_fingerprint(train_samples, counts),
        "training_files": [s.path.name for s in train_samples], "training_files_sha256": hashlib.sha256("\n".join(s.path.name for s in train_samples).encode()).hexdigest(),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "gate_classification": gate_classification(metrics["recall"], metrics["f1_score"], metrics["false_defect_detection_rate"]),
        "training": {"epochs": training.epochs, "device": training.device, "num_training_images": training.num_training_images,
                     "final_loss": training.final_loss, "loss_history": training.loss_history, "duration_seconds": training.duration_seconds},
        "artifact": artifact,
        "threshold": {"k": DEFAULT_THRESHOLD_K, "value": threshold, "train_error_stats": train_stats},
        "lock": lock,
        "metrics": metrics,
        "per_defect": per_defect,
        "distributions": {"train_good": train_stats, "test_good": describe(good), "test_defective": describe(bad),
                          "test_defective_by_type": {n: describe(v) for n, v in sorted(by_type.items())}},
        "separation": {"good_mean": statistics.fmean(good), "defective_mean": statistics.fmean(bad),
                       "good_median": statistics.median(good), "defective_median": statistics.median(bad),
                       "difference_of_means": statistics.fmean(bad) - statistics.fmean(good),
                       "difference_of_medians": statistics.median(bad) - statistics.median(good),
                       "overlap": _overlap(good, bad, threshold), "auroc_descriptive_only": _auroc(good, bad),
                       "train_good_vs_test_good_mean_ratio": statistics.fmean(good) / train_stats["mean"]},
        "test_good_above_threshold": {"count": int((np.asarray(good) > threshold).sum()), "of": len(good)},
        "defective_above_threshold": {"count": int((np.asarray(bad) > threshold).sum()), "of": len(bad)},
        "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "timing": {"training_seconds": training.duration_seconds, "model_load_ms": model_load_ms,
                   "preprocessing_ms_per_image": statistics.fmean(prep_ms), "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms),
                   "final_test_evaluation_total_ms": eval_total_ms, "train_scoring_ms_per_image":
                       statistics.fmean(train_prep_ms) + statistics.fmean(train_inf_ms),
                   "end_to_end_run_seconds": time.perf_counter() - started},
        "leakage_partial": leakage,
        "predictions": predictions,
        "train_errors": train_errors,
    }


def compare_runs(a: dict, b: dict) -> dict:
    """Field-by-field comparison of two complete runs (exact equality; also reports the largest numeric gap)."""
    pa = [(p["file"], p["reconstruction_error"], p["predicted_label"]) for p in a["predictions"]]
    pb = [(p["file"], p["reconstruction_error"], p["predicted_label"]) for p in b["predictions"]]
    err_gap = max((abs(x[1] - y[1]) for x, y in zip(pa, pb)), default=0.0)
    train_gap = max((abs(x - y) for x, y in zip(a["train_errors"], b["train_errors"])), default=0.0)
    return {
        "dataset_fingerprint_identical": a["dataset_fingerprint_sha256"] == b["dataset_fingerprint_sha256"],
        "training_order_identical": a["training_files"] == b["training_files"] and a["training_files_sha256"] == b["training_files_sha256"],
        "model_configuration_identical": a["config"] == b["config"] and a["parameter_count"] == b["parameter_count"],
        "loss_history_identical": a["training"]["loss_history"] == b["training"]["loss_history"],
        "loss_history_max_abs_difference": max(abs(x - y) for x, y in zip(a["training"]["loss_history"], b["training"]["loss_history"])),
        "final_loss_identical": a["training"]["final_loss"] == b["training"]["final_loss"],
        "model_sha256_identical": a["artifact"]["sha256"] == b["artifact"]["sha256"],
        "model_md5_identical": a["artifact"]["md5"] == b["artifact"]["md5"],
        "threshold_identical": a["threshold"]["value"] == b["threshold"]["value"],
        "threshold_absolute_difference": abs(a["threshold"]["value"] - b["threshold"]["value"]),
        "train_errors_identical": a["train_errors"] == b["train_errors"], "train_errors_max_abs_difference": train_gap,
        "test_errors_identical": [x[1] for x in pa] == [x[1] for x in pb], "test_errors_max_abs_difference": err_gap,
        "predicted_labels_identical": [x[2] for x in pa] == [x[2] for x in pb],
        "confusion_matrix_identical": a["metrics"]["confusion_matrix"] == b["metrics"]["confusion_matrix"],
        "metrics_identical": {k: v for k, v in a["metrics"].items()} == {k: v for k, v in b["metrics"].items()},
        "per_defect_recall_identical": {k: v["detected"] for k, v in a["per_defect"].items()} == {k: v["detected"] for k, v in b["per_defect"].items()},
    }
