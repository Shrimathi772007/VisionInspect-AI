"""The ONE final-test evaluation of a frozen anomaly detector, plus the success classification.

Kept apart from app.ai.evaluation.anomaly_model_selection (which cannot list test images). The single
entry point, `evaluate_frozen_on_final_test`, accepts only a directory holding a frozen model, and:

  * refuses to run if a final-test result already exists for the category (exactly once, ever);
  * verifies the frozen state file's SHA-256 against the lock record written before the test set was
    unlocked, and the backbone weights' SHA-256 (via the verified loader);
  * lists the test images only now, scores each once, and applies the LOCKED threshold
    (score > threshold -> defective) - nothing about the test images can change the model or threshold;
  * runs a data-derived leakage audit and writes the full result (predictions included) to JSON.

Project target bands (defect recall and F1 matter more than accuracy - a missed defect is the critical failure):
    EXCELLENT recall >= 0.90 and F1 >= 0.85 | GOOD recall >= 0.85 and F1 >= 0.80
    ACCEPTABLE recall >= 0.75 and F1 >= 0.70 | otherwise BELOW TARGET
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

from app.ai.evaluation.anomaly_model_selection import _md5_sha256, score_uint8, to_tensor
from app.ai.evaluation.category_phase3 import (
    PHASE3_SPLIT_SEED,
    PHASE3_TRAINING_FRACTION,
    defect_type_recall,
    metrics_from_predictions,
    predictions_at_threshold,
    split_leakage_checks,
)
from app.ai.evaluation.phase3_threshold_selection import select_threshold_from_validation
from app.ai.evaluation.threshold_experiments import generate_candidates
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.resnet18 import RESNET18_SHA256, load_pretrained_resnet18
from app.ai.preprocessing.pipeline import _build_preprocessing_result, _load_image
from app.ai.training import artifacts, discover_test_samples, discover_train_samples
from app.ai.training.validation_split import split_train_validation

EXCELLENT, GOOD, ACCEPTABLE, BELOW_TARGET = "EXCELLENT", "GOOD", "ACCEPTABLE", "BELOW TARGET"


def classify_result(recall: float, f1: float) -> str:
    if recall >= 0.90 and f1 >= 0.85:
        return EXCELLENT
    if recall >= 0.85 and f1 >= 0.80:
        return GOOD
    if recall >= 0.75 and f1 >= 0.70:
        return ACCEPTABLE
    return BELOW_TARGET


def final_result_path(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / "model_selection" / "reports" / "final_test_result.json"


def _training_list_hash(split) -> str:
    return hashlib.sha256("\n".join(s.path.name for s in split.training).encode()).hexdigest()


def evaluate_frozen_on_final_test(category: str, frozen_dir: Path) -> dict:
    result_path = final_result_path(category)
    if result_path.exists():
        raise FileExistsError(f"A final-test result already exists for '{category}' ({result_path}); the final "
                              "test is evaluated exactly once.")
    frozen_dir = Path(frozen_dir)
    lock = json.loads((frozen_dir / "frozen_model.json").read_text(encoding="utf-8"))

    md5, sha, size = _md5_sha256(frozen_dir / lock["state_file"]["name"])
    if sha != lock["state_file"]["sha256"]:
        raise RuntimeError(f"Frozen model state changed after freezing: expected {lock['state_file']['sha256']}, found {sha}.")
    if lock["backbone"]["sha256"] != RESNET18_SHA256:
        raise RuntimeError("Frozen model was locked against a different backbone than the one this code verifies.")
    if json.loads((frozen_dir / "model_config.json").read_text(encoding="utf-8")) != lock["config"]:
        raise RuntimeError("Frozen model config differs from its lock record.")

    t0 = time.perf_counter()
    extractor = load_pretrained_resnet18()
    detector = PatchAnomalyDetector.load(frozen_dir, extractor)
    model_load_ms = (time.perf_counter() - t0) * 1000
    side = detector.image_size
    threshold = lock["threshold"]["value"]

    split = split_train_validation(discover_train_samples(category), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)

    # ---- The final test set is first touched HERE, after the lock record and hashes are verified. ----
    test_samples = discover_test_samples(category)
    scores, prep_ms, inf_ms = [], [], []
    for sample in test_samples:
        t = time.perf_counter()
        prepared = _build_preprocessing_result(sample.path, _load_image(sample.path), (side, side)).resized_image
        tensor = to_tensor(prepared[None])
        t1 = time.perf_counter()
        scores.extend(detector.score_images(tensor))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t) * 1000)
        inf_ms.append((t2 - t1) * 1000)

    predictions = predictions_at_threshold(test_samples, scores, threshold)  # score > threshold -> defective
    metrics = metrics_from_predictions(predictions)
    by_type = defect_type_recall(predictions)
    classification = classify_result(metrics["recall"], metrics["f1_score"])

    # ---- Leakage audit (data-derived) ----
    val_scores = score_uint8(detector, _validation_uint8(split, side))
    rederived = select_threshold_from_validation(generate_candidates(val_scores), val_scores).selected
    leakage = split_leakage_checks(category, split, test_samples)
    leakage.update({
        "model_fitted_on_exactly_the_training_list": lock["fitted_on"]["file_list_sha256"] == _training_list_hash(split)
        and lock["fitted_on"]["training_images"] == len(split.training),
        "locked_threshold_rederived_from_validation_good_scores_only": rederived is not None
        and rederived.threshold == threshold,
        "frozen_state_hash_verified_before_test": True,
        "backbone_is_fixed_pretrained_file_verified_by_sha256": True,
        "normalization_is_fixed_imagenet_constants_no_data_statistics": True,
        "no_augmentation_or_test_time_adaptation": True,
    })
    leakage["all_passed"] = all(leakage.values())

    good_scores = [p["reconstruction_error"] for p in predictions if p["label"] == 0]
    bad_scores = [p["reconstruction_error"] for p in predictions if p["label"] == 1]
    per_type_scores = {}
    for p in predictions:
        if p["label"] == 1:
            per_type_scores.setdefault(p["defect_type"], []).append(p["reconstruction_error"])

    result = {
        "category": category,
        "candidate_id": lock["candidate_id"],
        "frozen_digest": lock["digest"],
        "threshold": lock["threshold"],
        "counts": {"test_total": len(predictions), "test_good": len(good_scores), "test_defective": len(bad_scores)},
        "metrics": metrics,
        "classification": classification,
        "targets": {"ACCEPTABLE": "recall>=0.75 & F1>=0.70", "GOOD": "recall>=0.85 & F1>=0.80",
                    "EXCELLENT": "recall>=0.90 & F1>=0.85"},
        "recall_by_defect_type": by_type,
        "detected_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "score_distribution": {
            "good": {"mean": statistics.fmean(good_scores), "median": statistics.median(good_scores),
                     "min": min(good_scores), "max": max(good_scores)},
            "defective": {"mean": statistics.fmean(bad_scores), "median": statistics.median(bad_scores),
                          "min": min(bad_scores), "max": max(bad_scores)},
            "defective_by_type_mean": {k: statistics.fmean(v) for k, v in per_type_scores.items()},
        },
        "leakage_audit": leakage,
        "timing": {"model_load_ms": model_load_ms, "preprocess_ms_per_image": statistics.fmean(prep_ms),
                   "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms),
                   "final_test_total_ms": sum(prep_ms) + sum(inf_ms)},
        "predictions": predictions,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _validation_uint8(split, side: int):
    import numpy as np

    return np.stack([
        _build_preprocessing_result(s.path, _load_image(s.path), (side, side)).resized_image for s in split.validation
    ])


__all__ = ["ACCEPTABLE", "BELOW_TARGET", "EXCELLENT", "GOOD", "classify_result", "evaluate_frozen_on_final_test",
           "final_result_path"]
