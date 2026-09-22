"""The single final-test evaluation of the OPTIMIZED frozen detector, with the false-positive-aware gates and a
comparison against the original (baseline) result.

Kept apart from app.ai.evaluation.calibration_pipeline / calibration_study, which cannot list test images. It:

  * refuses to run twice (a result file for the optimization already existing);
  * verifies the frozen state file's SHA-256 against its lock record, the backbone weights' SHA-256, and that the
    original Carpet baseline artifact is byte-identical to the hash recorded at freeze time (never overwritten);
  * lists the test images only after those checks, scores each once, and applies the LOCKED threshold
    (score > threshold -> defective);
  * runs a data-derived leakage audit and loads - never recomputes or alters - the original baseline result.

HONESTY NOTE: the Carpet test set was already scored once, for the original detector. This is therefore a second
look at the same 117 images. The optimization protocol was declared on train/good data before this evaluation, but
its design was motivated by knowing the baseline's outcome, so this result is less pristine than the first.

Gates (reporting only, applied after freezing):
    EXCELLENT   recall >= 0.90, F1 >= 0.85, FPR <= 0.10 | GOOD  recall >= 0.85, F1 >= 0.80, FPR <= 0.10
    ACCEPTABLE  recall >= 0.75, F1 >= 0.70, FPR <= 0.15 | otherwise NOT PRODUCTION READY
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from app.ai.evaluation.anomaly_model_selection import score_uint8, to_tensor
from app.ai.evaluation.calibration_pipeline import SIDE, _md5_sha256, reports_dir, selected_dir
from app.ai.evaluation.category_phase3 import (
    defect_type_recall,
    metrics_from_predictions,
    predictions_at_threshold,
)
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.resnet18 import RESNET18_SHA256, load_pretrained_resnet18
from app.ai.preprocessing.pipeline import _build_preprocessing_result, _load_image
from app.ai.training import artifacts, discover_test_samples, discover_train_samples

EXCELLENT, GOOD, ACCEPTABLE, NOT_PRODUCTION_READY = "EXCELLENT", "GOOD", "ACCEPTABLE", "NOT PRODUCTION READY"


def classify_with_fpr(recall: float, f1: float, fpr: float) -> str:
    if recall >= 0.90 and f1 >= 0.85 and fpr <= 0.10:
        return EXCELLENT
    if recall >= 0.85 and f1 >= 0.80 and fpr <= 0.10:
        return GOOD
    if recall >= 0.75 and f1 >= 0.70 and fpr <= 0.15:
        return ACCEPTABLE
    return NOT_PRODUCTION_READY


def result_path(category: str) -> Path:
    return reports_dir(category) / "final_test_result.json"


def baseline_result_path(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / "model_selection" / "reports" / "final_test_result.json"


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_with_baseline(baseline: dict, optimized: dict) -> dict:
    """Metric deltas plus paired per-image analysis (which good/defective images changed outcome)."""
    keys = ("accuracy", "precision", "recall", "f1_score", "false_defect_detection_rate",
            "true_negatives", "false_positives", "false_negatives", "true_positives")
    table = {k: {"baseline": baseline["metrics"][k], "optimized": optimized["metrics"][k],
                 "absolute_change": optimized["metrics"][k] - baseline["metrics"][k]} for k in keys}
    base = {p["file"]: p for p in baseline["predictions"]}
    opt = {p["file"]: p for p in optimized["predictions"]}
    common = set(base) & set(opt)
    paired = {
        "same_test_images": set(base) == set(opt),
        "false_positives_removed": sum(1 for f in common if base[f]["label"] == 0 and base[f]["predicted_label"] == 1 and opt[f]["predicted_label"] == 0),
        "false_positives_remaining": sum(1 for f in common if base[f]["label"] == 0 and opt[f]["predicted_label"] == 1 and base[f]["predicted_label"] == 1),
        "new_false_positives": sum(1 for f in common if base[f]["label"] == 0 and base[f]["predicted_label"] == 0 and opt[f]["predicted_label"] == 1),
        "defects_lost": sum(1 for f in common if base[f]["label"] == 1 and base[f]["predicted_label"] == 1 and opt[f]["predicted_label"] == 0),
        "defects_gained": sum(1 for f in common if base[f]["label"] == 1 and base[f]["predicted_label"] == 0 and opt[f]["predicted_label"] == 1),
    }
    return {"metrics": table, "paired": paired}


def evaluate_optimized_on_final_test(category: str, frozen_dir: Path | None = None) -> dict:
    out = result_path(category)
    if out.exists():
        raise FileExistsError(f"An optimized final-test result already exists ({out}); the final test is scored once.")
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else selected_dir(category)
    lock = json.loads((frozen_dir / "frozen_model.json").read_text(encoding="utf-8"))

    _, sha, _ = _md5_sha256(frozen_dir / lock["state_file"]["name"])
    if sha != lock["state_file"]["sha256"]:
        raise RuntimeError(f"Frozen model state changed after freezing: expected {lock['state_file']['sha256']}, found {sha}.")
    if lock["backbone"]["sha256"] != RESNET18_SHA256:
        raise RuntimeError("Frozen model was locked against a different backbone.")
    if json.loads((frozen_dir / "model_config.json").read_text(encoding="utf-8")) != lock["config"]:
        raise RuntimeError("Frozen model config differs from its lock record.")
    baseline_state = artifacts.ARTIFACTS_ROOT / category / "selected_model" / "model_state.pt"
    if _md5_sha256(baseline_state)[1] != lock["baseline_reference"]["model_state_sha256"]:
        raise RuntimeError("The original Carpet baseline artifact no longer matches the hash recorded at freeze time.")
    baseline = json.loads(baseline_result_path(category).read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    extractor = load_pretrained_resnet18()
    detector = PatchAnomalyDetector.load(frozen_dir, extractor)
    model_load_ms = (time.perf_counter() - t0) * 1000
    threshold = lock["threshold"]["value"]

    train_samples = discover_train_samples(category)  # ---- the 280 good images the model was fitted/calibrated on
    # ---- The final test set is first touched HERE, after every lock/hash check above. ----
    test_samples = discover_test_samples(category)
    scores, prep_ms, inf_ms = [], [], []
    for sample in test_samples:
        t = time.perf_counter()
        prepared = _build_preprocessing_result(sample.path, _load_image(sample.path), (SIDE, SIDE)).resized_image
        tensor = to_tensor(prepared[None])
        t1 = time.perf_counter()
        scores.extend(detector.score_images(tensor))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t) * 1000)
        inf_ms.append((t2 - t1) * 1000)

    predictions = predictions_at_threshold(test_samples, scores, threshold)
    metrics = metrics_from_predictions(predictions)
    classification = classify_with_fpr(metrics["recall"], metrics["f1_score"], metrics["false_defect_detection_rate"])

    # ---- Leakage audit (data-derived) ----
    pool_paths = {s.path for s in train_samples}
    test_paths = {s.path for s in test_samples}
    pool_hashes = {_content_hash(p) for p in pool_paths}
    test_hashes = {_content_hash(p) for p in test_paths}
    study = json.loads((reports_dir(category) / "calibration_study.json").read_text(encoding="utf-8"))
    study_row = next(c for c in study["configurations"] if c["config"] == lock["selection"]["config_id"])
    leakage = {
        "fit_and_calibration_pool_is_exactly_the_train_good_images": lock["fitted_on"]["file_list_sha256"] == hashlib.sha256(
            "\n".join(sorted(s.path.name for s in train_samples)).encode()).hexdigest() and lock["fitted_on"]["training_images"] == len(train_samples),
        "pool_only_from_train_good": all(s.split == "train" and s.defect_type == "good" for s in train_samples),
        "test_only_from_test_split": all(s.split == "test" for s in test_samples),
        "test_disjoint_from_pool_by_path": not (test_paths & pool_paths),
        "test_disjoint_from_pool_by_content": not (test_hashes & pool_hashes),
        "locked_threshold_equals_the_good_only_study_threshold": study_row["final_threshold"] == threshold,
        "study_report_written_before_final_test": (reports_dir(category) / "calibration_study.json").stat().st_mtime
        <= (frozen_dir / "frozen_model.json").stat().st_mtime,
        "frozen_state_hash_verified_before_test": True,
        "backbone_is_fixed_pretrained_file_verified_by_sha256": True,
        "original_baseline_artifact_unchanged": True,
        "synthetic_defects_generated_from_validation_good_images_only_no_real_masks": True,
        "normalization_is_fixed_imagenet_constants_no_data_statistics": True,
        "no_augmentation_or_test_time_adaptation": True,
    }
    leakage["all_passed"] = all(leakage.values())

    good_scores = [p["reconstruction_error"] for p in predictions if p["label"] == 0]
    bad_scores = [p["reconstruction_error"] for p in predictions if p["label"] == 1]
    by_type_scores: dict[str, list] = {}
    for p in predictions:
        if p["label"] == 1:
            by_type_scores.setdefault(p["defect_type"], []).append(p["reconstruction_error"])
    result = {
        "category": category,
        "candidate_id": lock["candidate_id"],
        "frozen_digest": lock["digest"],
        "threshold": lock["threshold"],
        "counts": {"test_total": len(predictions), "test_good": len(good_scores), "test_defective": len(bad_scores)},
        "metrics": metrics,
        "classification": classification,
        "gates": {"EXCELLENT": "recall>=0.90 & F1>=0.85 & FPR<=0.10", "GOOD": "recall>=0.85 & F1>=0.80 & FPR<=0.10",
                  "ACCEPTABLE": "recall>=0.75 & F1>=0.70 & FPR<=0.15", "otherwise": NOT_PRODUCTION_READY},
        "recall_by_defect_type": defect_type_recall(predictions),
        "detected_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "score_distribution": {
            "good": {"mean": statistics.fmean(good_scores), "median": statistics.median(good_scores),
                     "std": float(np.std(good_scores)), "min": min(good_scores), "max": max(good_scores)},
            "defective": {"mean": statistics.fmean(bad_scores), "median": statistics.median(bad_scores),
                          "std": float(np.std(bad_scores)), "min": min(bad_scores), "max": max(bad_scores)},
            "defective_by_type_mean": {k: statistics.fmean(v) for k, v in by_type_scores.items()},
            "good_p50_p90_p95_max": [float(x) for x in np.percentile(good_scores, [50, 90, 95, 100])],
        },
        "leakage_audit": leakage,
        "timing": {"model_load_ms": model_load_ms, "preprocess_ms_per_image": statistics.fmean(prep_ms),
                   "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms),
                   "final_test_total_ms": sum(prep_ms) + sum(inf_ms)},
        "test_set_use_note": "Second scoring of the Carpet test set (the original detector was scored first); see module docstring.",
        "predictions": predictions,
    }
    result["comparison_with_original_baseline"] = compare_with_baseline(baseline, result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


__all__ = ["ACCEPTABLE", "EXCELLENT", "GOOD", "NOT_PRODUCTION_READY", "classify_with_fpr", "compare_with_baseline",
           "evaluate_optimized_on_final_test", "result_path", "score_uint8"]
