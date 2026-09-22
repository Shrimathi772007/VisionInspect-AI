"""The single, protected final-test evaluation of a category's Phase 2 LOCKED threshold.

Kept apart from app.ai.evaluation.category_phase2, which cannot list test images. This module:

  * refuses to run twice (a final-test result file already existing);
  * verifies, BEFORE any test image is listed, the frozen model's SHA-256, that the threshold lock names that model,
    that the lock records no final-test use, that the calibration study predates the lock, that the locked threshold
    is the study's selected threshold, and that the Phase 1 artifacts are byte-identical to the hashes recorded in
    the lock;
  * lists and scores the test images once and applies the locked threshold as-is (score > threshold -> defective);
  * re-verifies the model and Phase 1 artifacts afterwards, runs a leakage audit and loads - never recomputes or
    alters - the recorded Phase 1 result for a purely descriptive comparison.

No threshold other than the locked one is applied to the test scores.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from app.ai.evaluation.category_phase1 import _errors, describe, file_hashes
from app.ai.evaluation.category_phase2 import lock_path, phase2_dir, reports_dir
from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions, predictions_at_threshold
from app.ai.training import artifacts
from app.ai.training.dataset import discover_test_samples, discover_train_samples
from app.ai.training.model import build_model


def result_path(category: str) -> Path:
    return reports_dir(category) / "final_test_result.json"


def phase1_paths(category: str) -> dict[str, Path]:
    base = artifacts.ARTIFACTS_ROOT / category / "phase1_baseline"
    return {"model": base / "autoencoder.pt", "threshold_lock": base / "threshold_lock.json", "report": base / "reports" / "phase1_report.json"}


def phase1_reference(category: str) -> dict:
    """Hashes of Phase 1's artifacts (bytes only - the Phase 1 report is not opened here)."""
    paths = phase1_paths(category)
    ref = {name: file_hashes(p)["sha256"] for name, p in paths.items()}
    ref["threshold"] = json.loads(paths["threshold_lock"].read_text(encoding="utf-8"))["threshold"]
    return ref


def _rank_within(values: np.ndarray, threshold: float) -> dict:
    return {"count_above_threshold": int((values > threshold).sum()), "of": int(values.size),
            "fraction_above_threshold": float((values > threshold).mean()),
            "threshold_percentile_rank": float((values <= threshold).mean() * 100)}


def compare_with_phase1(phase1: dict, phase2_metrics: dict, phase2_per_defect: dict, phase2_predictions: list[dict],
                        phase1_threshold: float, phase2_threshold: float) -> dict:
    keys = ("accuracy", "precision", "recall", "f1_score", "false_defect_detection_rate", "true_negatives", "false_positives",
            "false_negatives", "true_positives")
    table = {"threshold": {"phase1": phase1_threshold, "phase2": phase2_threshold}}
    for k in keys:
        table[k] = {"phase1": phase1["metrics"][k], "phase2": phase2_metrics[k], "change": phase2_metrics[k] - phase1["metrics"][k]}
    per_defect = {t: {"phase1_recall": phase1["per_defect"][t]["recall"], "phase1_detected": phase1["per_defect"][t]["detected"],
                      "phase2_recall": v["recall"], "phase2_detected": v["detected"], "total": v["total"],
                      "change_in_detected": v["detected"] - phase1["per_defect"][t]["detected"]} for t, v in phase2_per_defect.items()}
    p1 = {p["file"]: p for p in phase1["predictions"]}
    p2 = {p["file"]: p for p in phase2_predictions}
    common = set(p1) & set(p2)
    paired = {
        "same_test_images": set(p1) == set(p2),
        "max_abs_score_difference_vs_phase1": max(abs(p1[f]["reconstruction_error"] - p2[f]["reconstruction_error"]) for f in common),
        "false_positives_removed": sum(1 for f in common if p1[f]["label"] == 0 and p1[f]["predicted_label"] == 1 and p2[f]["predicted_label"] == 0),
        "new_false_positives": sum(1 for f in common if p1[f]["label"] == 0 and p1[f]["predicted_label"] == 0 and p2[f]["predicted_label"] == 1),
        "defects_lost": sum(1 for f in common if p1[f]["label"] == 1 and p1[f]["predicted_label"] == 1 and p2[f]["predicted_label"] == 0),
        "defects_gained": sum(1 for f in common if p1[f]["label"] == 1 and p1[f]["predicted_label"] == 0 and p2[f]["predicted_label"] == 1),
        "predictions_changed": sum(1 for f in common if p1[f]["predicted_label"] != p2[f]["predicted_label"]),
    }
    return {"metrics": table, "per_defect": per_defect, "paired": paired,
            "threshold_identical_to_phase1": phase1_threshold == phase2_threshold}


def evaluate_locked_on_final_test(category: str, expected_model_sha256: str, image_size: tuple[int, int] = (128, 128)) -> dict:
    out = result_path(category)
    if out.exists():
        raise FileExistsError(f"A Phase 2 final-test result already exists ({out}); the final test is scored exactly once.")
    lock_file = lock_path(category)
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    study_file = reports_dir(category) / "calibration_study.json"
    study = json.loads(study_file.read_text(encoding="utf-8"))

    model_path = phase1_paths(category)["model"]
    sha_before = file_hashes(model_path)["sha256"]
    if sha_before != expected_model_sha256 or lock["model_sha256"] != sha_before:
        raise RuntimeError("The frozen model does not match the expected hash / the lock's recorded hash.")
    if lock["final_test_data_used_for_selection"] is not False:
        raise RuntimeError("The lock does not certify that final-test data was unused.")
    if study["selection"]["threshold"] != lock["threshold"] or study["selection"]["winner"] != lock["threshold_method"]:
        raise RuntimeError("The locked threshold is not the calibration study's selected threshold.")
    if not study_file.stat().st_mtime <= lock_file.stat().st_mtime:
        raise RuntimeError("The calibration study was written after the threshold lock.")
    ref_now = phase1_reference(category)
    if ref_now != lock["phase1_reference"]:
        raise RuntimeError("A Phase 1 artifact no longer matches the hashes recorded in the lock.")
    threshold = lock["threshold"]
    train_samples = discover_train_samples(category)

    model = artifacts.load_model(model_path, build_model())

    # ---- The protected final test set is first touched HERE, after every check above. ----
    test_listed_at = time.time()
    started = time.perf_counter()
    test_samples = discover_test_samples(category)
    errors, prep_ms, inf_ms = _errors(model, test_samples, image_size)
    eval_total_ms = (time.perf_counter() - started) * 1000
    predictions = predictions_at_threshold(test_samples, errors, threshold)
    metrics = metrics_from_predictions(predictions)

    good = np.array([p["reconstruction_error"] for p in predictions if p["label"] == 0])
    bad = np.array([p["reconstruction_error"] for p in predictions if p["label"] == 1])
    by_type: dict[str, list[float]] = {}
    for p in predictions:
        if p["label"] == 1:
            by_type.setdefault(p["defect_type"], []).append(p["reconstruction_error"])
    recall_by_type = defect_type_recall(predictions)
    per_defect = {}
    for name in sorted(by_type):
        s = by_type[name]
        per_defect[name] = {"total": recall_by_type[name]["total"], "detected": recall_by_type[name]["detected"],
                            "missed": recall_by_type[name]["total"] - recall_by_type[name]["detected"], "recall": recall_by_type[name]["recall"],
                            "mean": statistics.fmean(s), "median": statistics.median(s), "min": min(s), "max": max(s)}

    phase1 = json.loads(phase1_paths(category)["report"].read_text(encoding="utf-8"))["run1"]
    comparison = compare_with_phase1(phase1, metrics, per_defect, predictions, ref_now["threshold"], threshold)

    # ---- Leakage audit ----
    sha_after = file_hashes(model_path)["sha256"]
    train_paths = {s.path for s in train_samples}
    test_paths = {s.path for s in test_samples}
    leakage = {
        "1_no_final_test_labels_used_during_threshold_selection": lock["final_test_data_used_for_selection"] is False
        and lock["file_list_sha256"] == study["file_list_sha256"]
        == hashlib.sha256("\n".join(sorted(s.path.name for s in train_samples)).encode()).hexdigest(),
        "2_no_final_test_scores_used_to_select_the_threshold": study["selection"]["threshold"] == threshold
        and study_file.stat().st_mtime <= lock_file.stat().st_mtime <= test_listed_at,
        "3_only_train_good_used_for_calibration": all(s.split == "train" and s.defect_type == "good" for s in train_samples)
        and len(train_samples) == study["n"],
        "4_calibration_splits_disjoint_where_required": all(study["split_audit"][k] for k in (
            "calibration_and_held_out_disjoint_in_every_split", "every_split_covers_the_pool", "kfold_held_out_folds_partition_the_pool")),
        "5_threshold_locked_before_final_testing": lock["timestamp_unix"] <= test_listed_at and lock_file.stat().st_mtime <= test_listed_at,
        "6_final_test_evaluated_once": not out.exists(),
        "7_phase1_model_weights_not_modified": sha_before == sha_after == expected_model_sha256,
        "8_phase1_artifacts_not_overwritten": phase1_reference(category) == lock["phase1_reference"],
        "calibration_and_test_paths_disjoint": not (train_paths & test_paths),
    }
    leakage["all_passed"] = all(leakage.values())

    result = {
        "category": category, "experiment_id": lock["experiment_id"], "locked_threshold": threshold, "threshold_method": lock["threshold_method"],
        "lock_file": str(lock_file), "model_sha256": sha_after, "counts": {"test_total": len(predictions), "test_good": int(good.size), "test_defective": int(bad.size)},
        "metrics": metrics, "defect_identification_accuracy_is_recall": metrics["defect_identification_accuracy"] == metrics["recall"],
        "per_defect": per_defect, "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "distributions": {"test_good": describe(good), "test_defective": describe(bad),
                          "test_defective_by_type": {n: describe(v) for n, v in sorted(by_type.items())}},
        "threshold_position": {"test_good": _rank_within(good, threshold), "test_defective": _rank_within(bad, threshold),
                               "threshold_over_test_good_min": float(threshold / good.min()), "threshold_over_test_good_median": float(threshold / np.median(good))},
        "comparison_with_phase1": comparison, "leakage_audit": leakage,
        "timing": {"preprocessing_ms_per_image": statistics.fmean(prep_ms), "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms), "final_test_total_ms": eval_total_ms},
        "test_listed_at_unix": test_listed_at, "predictions": predictions,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


__all__ = ["compare_with_phase1", "evaluate_locked_on_final_test", "phase1_reference", "phase2_dir", "result_path"]
