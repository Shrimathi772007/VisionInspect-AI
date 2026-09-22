"""Stage 2 of the held-out Phase 3 baseline: the single, protected final-test evaluation of the LOCKED threshold.

Kept apart from app.ai.evaluation.category_phase3_heldout, which cannot decode or score a test image. Before any test
image is listed, `verify_lock` requires all of:

  * no final-test result already exists (the test is scored exactly once);
  * the threshold lock's own digest matches its content (a hand-edited lock is rejected);
  * the lock is exactly what the pre-declared rule selects from the saved VALIDATION errors (recomputed here, so a
    lock and digest edited together are rejected too), and the validation-stage report holds the same digest;
  * the model file's SHA-256 and MD5 equal the lock's; the train/good dataset fingerprint is unchanged;
  * every Phase 1 / Phase 2 artifact is byte-identical to the hashes recorded in the lock.

Only then are the 124 test images scored, once, with the locked threshold (score > threshold -> defective). No other
threshold is applied to the test scores. AUROC and all distribution statistics are descriptive and are computed
after the predictions; the earlier phases' recorded results are loaded read-only for comparison.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from app.ai.evaluation.calibration_final_test import classify_with_fpr
from app.ai.evaluation.category_phase1 import _errors, describe, file_hashes
from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions, predictions_at_threshold
from app.ai.evaluation.category_phase3_heldout import (
    dataset_fingerprint,
    lock_digest,
    lock_path,
    model_path,
    prior_phase_reference,
    reports_dir,
    select_candidate,
    validation_report_path,
)
from app.ai.evaluation.threshold_experiments import generate_candidates
from app.ai.training import artifacts
from app.ai.training.dataset import discover_test_samples, discover_train_samples
from app.ai.training.model import build_model


def result_path(category: str) -> Path:
    return reports_dir(category) / "final_test_result.json"


def verify_lock(category: str) -> tuple[dict, dict]:
    """Every pre-test check. Returns (lock, validation-stage report). Never lists a test image."""
    if result_path(category).exists():
        raise FileExistsError(f"A Phase 3 final-test result already exists ({result_path(category)}); the final test is scored exactly once.")
    lock = json.loads(lock_path(category).read_text(encoding="utf-8"))
    report = json.loads(validation_report_path(category).read_text(encoding="utf-8"))

    if lock_digest(lock) != lock.get("lock_digest"):
        raise RuntimeError("Threshold lock content does not match its digest (lock was modified).")
    if lock["final_test_data_used_for_selection"] is not False:
        raise RuntimeError("The lock does not certify that final-test data was unused.")
    if report["lock"]["lock_digest"] != lock["lock_digest"]:
        raise RuntimeError("Threshold lock differs from the lock recorded in the validation-stage report.")

    errors = report["validation_errors"]
    if hashlib.sha256(np.asarray(errors, dtype=np.float64).tobytes()).hexdigest() != lock["validation_errors_sha256"]:
        raise RuntimeError("Validation errors in the report do not match the lock's recorded hash.")
    reselected = select_candidate(generate_candidates(errors), errors)
    if reselected["winner"] != lock["threshold_method"] or reselected["winner_row"]["threshold"] != lock["threshold"]:
        raise RuntimeError("The locked threshold is not what the pre-declared rule selects from the validation errors.")

    hashes = file_hashes(model_path(category))
    if hashes["sha256"] != lock["model_sha256"] or hashes["md5"] != lock["model_md5"]:
        raise RuntimeError("The Phase 3 model file no longer matches the hashes in the threshold lock (model was modified).")
    if dataset_fingerprint(discover_train_samples(category), lock["dataset_counts"]) != lock["dataset_fingerprint_sha256"]:
        raise RuntimeError("The train/good dataset no longer matches the fingerprint in the threshold lock.")
    if prior_phase_reference(category) != lock["prior_phase_artifacts"]:
        raise RuntimeError("A Phase 1 or Phase 2 artifact no longer matches the hashes recorded in the lock.")
    return lock, report


def _rank(values: np.ndarray, threshold: float) -> dict:
    return {"count_above_threshold": int((values > threshold).sum()), "of": int(values.size),
            "fraction_above_threshold": float((values > threshold).mean()),
            "threshold_percentile_rank": float((values <= threshold).mean() * 100)}


def evaluate_locked_on_final_test(category: str, image_size: tuple[int, int] = (128, 128)) -> dict:
    lock, report = verify_lock(category)
    threshold = lock["threshold"]
    samples = discover_train_samples(category)
    lock_file = lock_path(category)
    model = artifacts.load_model(model_path(category), build_model(latent_channels=lock["training_config"]["latent_channels"]))

    # ---- The protected final test set is first touched HERE, after every check above. ----
    test_listed_at = time.time()
    started = time.perf_counter()
    test_samples = discover_test_samples(category)
    errors, prep_ms, inf_ms = _errors(model, test_samples, image_size)
    eval_ms = (time.perf_counter() - started) * 1000
    predictions = predictions_at_threshold(test_samples, errors, threshold)
    metrics = metrics_from_predictions(predictions)

    good = np.array([p["reconstruction_error"] for p in predictions if p["label"] == 0])
    bad = np.array([p["reconstruction_error"] for p in predictions if p["label"] == 1])
    auroc = float(roc_auc_score([0] * good.size + [1] * bad.size, list(good) + list(bad)))
    by_type: dict[str, list[float]] = {}
    for p in predictions:
        if p["label"] == 1:
            by_type.setdefault(p["defect_type"], []).append(p["reconstruction_error"])
    recall_by_type = defect_type_recall(predictions)
    per_defect = {n: {"total": recall_by_type[n]["total"], "detected": recall_by_type[n]["detected"],
                      "missed": recall_by_type[n]["total"] - recall_by_type[n]["detected"], "recall": recall_by_type[n]["recall"],
                      "mean": statistics.fmean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
                  for n, v in sorted(by_type.items())}

    val = np.asarray(report["validation_errors"])
    train = np.asarray(report["training_errors"])
    position = {
        "test_good": _rank(good, threshold), "test_defective": _rank(bad, threshold),
        "threshold_over_validation_max": float(threshold / val.max()), "threshold_over_training_max": float(threshold / train.max()),
        "test_good_above_validation_max": int((good > val.max()).sum()), "test_good_above_training_max": int((good > train.max()).sum()),
        "test_good_mean_over_validation_mean": float(good.mean() / val.mean()), "test_good_mean_over_training_mean": float(good.mean() / train.mean()),
        "test_good_median_over_validation_median": float(np.median(good) / np.median(val)),
    }

    prior = _prior_results(category, threshold)
    phase3 = {"threshold": threshold, "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"],
              "f1_score": metrics["f1_score"], "false_defect_detection_rate": metrics["false_defect_detection_rate"], "auroc": auroc,
              "true_negatives": metrics["true_negatives"], "false_positives": metrics["false_positives"],
              "false_negatives": metrics["false_negatives"], "true_positives": metrics["true_positives"]}
    comparison = {"phase1": prior["phase1"], "phase2": prior["phase2"], "phase3": phase3,
                  "per_defect": {n: {"phase1_recall": prior["per_defect_phase1"][n]["recall"], "phase2_recall": prior["per_defect_phase2"][n]["recall"],
                                     "phase3_recall": v["recall"], "phase1_detected": prior["per_defect_phase1"][n]["detected"],
                                     "phase3_detected": v["detected"], "total": v["total"]} for n, v in per_defect.items()},
                  "distributions": {"phase1": prior["distributions_phase1"],
                                    "phase3": {"training_196": report["distributions"]["training"], "validation_49": report["distributions"]["validation"],
                                               "test_good": describe(good), "test_defective": describe(bad)}}}

    # ---- Leakage audit ----
    train_paths = {s.path for s in samples}
    test_paths = {s.path for s in test_samples}
    trained_on = set(report["training_files"])
    validation_files = set(report["validation_files"])
    hashes_after = file_hashes(model_path(category))
    leakage = {
        "1_only_the_196_training_images_used_for_training": report["training"]["num_training_images"] == lock["training_image_count"]
        == len(trained_on) and trained_on <= {p.name for p in train_paths},
        "2_no_validation_image_used_for_training": not (trained_on & validation_files) and len(validation_files) == lock["validation_image_count"],
        "3_no_test_image_used_for_calibration": not (test_paths & train_paths)
        and not ({hashlib.sha256(p.read_bytes()).hexdigest() for p in test_paths} & {hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths}),
        "4_threshold_selected_from_validation_errors_only": select_candidate(generate_candidates(report["validation_errors"]), report["validation_errors"])["winner"] == lock["threshold_method"],
        "5_threshold_locked_before_the_final_test_was_listed": lock["timestamp_unix"] <= test_listed_at and lock_file.stat().st_mtime <= test_listed_at,
        "6_model_hash_verified_before_and_after_the_final_test": hashes_after["sha256"] == lock["model_sha256"] and hashes_after["md5"] == lock["model_md5"],
        "7_final_test_scored_exactly_once": not result_path(category).exists(),
        "8_phase1_and_phase2_artifacts_unchanged": prior_phase_reference(category) == lock["prior_phase_artifacts"],
        "9_no_masks_or_synthetic_defects_or_augmentation": not any("ground_truth" in str(p) for p in train_paths | test_paths)
        and lock["training_config"]["augmentation"] == "none",
        "10_lock_digest_verified": lock_digest(lock) == lock["lock_digest"],
    }
    leakage["all_passed"] = all(leakage.values())

    result = {
        "category": category, "phase": lock["phase"], "experiment_id": lock["experiment_id"], "locked_threshold": threshold,
        "threshold_method": lock["threshold_method"], "model_sha256": hashes_after["sha256"],
        "counts": {"test_total": len(predictions), "test_good": int(good.size), "test_defective": int(bad.size)},
        "metrics": metrics, "auroc_descriptive_only": auroc, "defect_identification_accuracy_is_recall": metrics["defect_identification_accuracy"] == metrics["recall"],
        "gate_classification": classify_with_fpr(metrics["recall"], metrics["f1_score"], metrics["false_defect_detection_rate"]),
        "per_defect": per_defect, "distributions": {"test_good": describe(good), "test_defective": describe(bad),
                                                     "test_defective_by_type": {n: describe(v) for n, v in sorted(by_type.items())}},
        "threshold_position": position, "comparison": comparison, "leakage_audit": leakage,
        "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "timing": {"preprocessing_ms_per_image": statistics.fmean(prep_ms), "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms), "final_test_total_ms": eval_ms},
        "test_listed_at_unix": test_listed_at, "predictions": predictions,
    }
    out = result_path(category)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _prior_results(category: str, phase3_threshold: float) -> dict:
    """Read-only load of Phase 1's and Phase 2's recorded final results (used only after the Phase 3 scoring)."""
    root = artifacts.ARTIFACTS_ROOT / category
    p1 = json.loads((root / "phase1_baseline" / "reports" / "phase1_report.json").read_text(encoding="utf-8"))["run1"]
    p2 = json.loads((root / "phase2_threshold" / "reports" / "final_test_result.json").read_text(encoding="utf-8"))
    p2_scores = [q for q in p2["predictions"]]
    p2_auroc = float(roc_auc_score([q["label"] for q in p2_scores], [q["reconstruction_error"] for q in p2_scores]))

    def row(threshold, m, auroc):
        return {"threshold": threshold, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1_score": m["f1_score"],
                "false_defect_detection_rate": m["false_defect_detection_rate"], "auroc": auroc, "true_negatives": m["true_negatives"],
                "false_positives": m["false_positives"], "false_negatives": m["false_negatives"], "true_positives": m["true_positives"]}

    return {"phase1": row(p1["threshold"]["value"], p1["metrics"], p1["separation"]["auroc_descriptive_only"]),
            "phase2": row(p2["locked_threshold"], p2["metrics"], p2_auroc),
            "per_defect_phase1": p1["per_defect"], "per_defect_phase2": p2["per_defect"],
            "distributions_phase1": {"train_good_245": p1["distributions"]["train_good"], "test_good": p1["distributions"]["test_good"],
                                     "test_defective": p1["distributions"]["test_defective"]}}
