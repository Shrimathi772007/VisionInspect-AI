"""The single, protected final-test evaluation of the LOCKED model-family study winner.

Kept apart from app.ai.evaluation.model_family_pipeline / model_family_study, which cannot list or score test images.
Before any test image is listed, `verify_lock` requires all of:

  * no final-test result already exists (the test is scored exactly once);
  * the selection lock's digest matches its content (a hand-edited lock is rejected);
  * the candidate registry is exactly the declared 8 candidates (configurations and fingerprint), the study report
    matches the lock, and the pre-declared selection rule, recomputed from the saved normal-only/synthetic study,
    selects exactly the locked candidate, policy and threshold (so a lock and digest edited together are rejected);
  * the selected model artifact, its copy in candidates/, and the frozen ImageNet backbone match their locked
    SHA-256 hashes (a tampered artifact or backbone is rejected);
  * the normal training dataset fingerprint is unchanged and every Phase 1/2/3 artifact is byte-identical.

Only then are the 124 test images scored, once, with the locked threshold (score > threshold -> defective). No other
candidate or threshold is applied to the test scores. AUROC and distribution statistics are descriptive and are
computed after the predictions; earlier phases' recorded results are read-only comparison inputs.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from app.ai.evaluation import model_family_study as mfs
from app.ai.evaluation.anomaly_model_selection import to_tensor
from app.ai.evaluation.calibration_final_test import classify_with_fpr
from app.ai.evaluation.category_phase1 import describe, file_hashes
from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions, predictions_at_threshold
from app.ai.evaluation.model_family_pipeline import (
    candidates_dir,
    dataset_fingerprint,
    lock_digest,
    lock_path,
    prior_phase_reference,
    reports_dir,
    selected_dir,
    study_report_path,
)
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.resnet18 import RESNET18_SHA256, load_pretrained_resnet18, pretrained_weights_path
from app.ai.preprocessing.pipeline import _build_preprocessing_result, _load_image
from app.ai.training import artifacts
from app.ai.training.dataset import discover_test_samples, discover_train_samples


def result_path(category: str) -> Path:
    return reports_dir(category) / "final_test_result.json"


def _canon(x):
    return json.loads(json.dumps(x, sort_keys=True))


def verify_lock(category: str) -> tuple[dict, dict]:
    """Every pre-test check. Returns (lock, study report). Never lists a test image."""
    if result_path(category).exists():
        raise FileExistsError(f"A final-test result already exists ({result_path(category)}); the final test is scored exactly once.")
    lock = json.loads(lock_path(category).read_text(encoding="utf-8"))
    study = json.loads(study_report_path(category).read_text(encoding="utf-8"))

    if lock_digest(lock) != lock.get("lock_digest"):
        raise RuntimeError("Selection lock content does not match its digest (lock was modified).")
    if lock["final_test_data_used_for_selection"] is not False:
        raise RuntimeError("The lock does not certify that final-test data was unused.")

    specs = mfs.registry()
    if len(specs) != mfs.EXPECTED_CANDIDATES or len(lock["candidate_list"]) != mfs.EXPECTED_CANDIDATES:
        raise RuntimeError("Candidate count differs from the declared registry.")
    if lock["registry_fingerprint"] != mfs.registry_fingerprint(specs) or _canon(lock["candidate_configurations"]) != _canon([s.config() for s in specs]):
        raise RuntimeError("Candidate configurations differ from the declared registry.")
    if study["registry_fingerprint"] != lock["registry_fingerprint"] or study["splits_sha256"] != lock["split_fingerprint_sha256"] \
            or study["dataset_fingerprint_sha256"] != lock["normal_dataset_fingerprint_sha256"]:
        raise RuntimeError("The study report does not match the selection lock.")

    reselected = mfs.select_configuration({cid: r["summary"] for cid, r in study["candidates"].items()}, specs)
    if reselected["winner"] != f"{lock['selected_candidate']}|{lock['selected_policy']}" or reselected["threshold"] != lock["selected_threshold"]:
        raise RuntimeError("The locked candidate/threshold is not what the pre-declared rule selects from the saved study.")

    backbone_sha = file_hashes(pretrained_weights_path())["sha256"]
    if backbone_sha != RESNET18_SHA256 or backbone_sha != lock["backbone"]["sha256"]:
        raise RuntimeError("The frozen ResNet-18 backbone does not match the locked hash (backbone was modified).")
    cid = lock["selected_candidate"]
    chosen = file_hashes(selected_dir(category) / "model_state.pt")["sha256"]
    if chosen != lock["selected_artifact"]["sha256"]:
        raise RuntimeError("The selected model artifact no longer matches the hash in the selection lock (artifact was modified).")
    if file_hashes(candidates_dir(category) / cid / "model_state.pt")["sha256"] != chosen:
        raise RuntimeError("The selected artifact differs from the fitted candidate artifact.")
    if _canon(json.loads((selected_dir(category) / "model_config.json").read_text(encoding="utf-8"))) != \
            _canon(json.loads((candidates_dir(category) / cid / "model_config.json").read_text(encoding="utf-8"))):
        raise RuntimeError("The selected model configuration differs from the fitted candidate configuration.")

    if dataset_fingerprint(discover_train_samples(category), lock["dataset_counts"]) != lock["normal_dataset_fingerprint_sha256"]:
        raise RuntimeError("The train/good dataset no longer matches the fingerprint in the selection lock.")
    if prior_phase_reference(category) != lock["prior_phase_artifacts"]:
        raise RuntimeError("A Phase 1/2/3 artifact no longer matches the hashes recorded in the lock.")
    return lock, study


def verify_lock_unchanged_since_final_test(category: str) -> bool:
    """True iff the lock file is byte-identical to the one the final test was run against."""
    result = json.loads(result_path(category).read_text(encoding="utf-8"))
    return file_hashes(lock_path(category))["sha256"] == result["selection_lock_file_sha256"]


def _rank(values: np.ndarray, threshold: float) -> dict:
    return {"count_above_threshold": int((values > threshold).sum()), "of": int(values.size),
            "fraction_above_threshold": float((values > threshold).mean()),
            "threshold_percentile_rank": float((values <= threshold).mean() * 100)}


def _optional_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _metric_row(m: dict, threshold: float, auroc: float) -> dict:
    return {"threshold": threshold, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1_score": m["f1_score"],
            "false_defect_detection_rate": m["false_defect_detection_rate"], "auroc": auroc, "true_negatives": m["true_negatives"],
            "false_positives": m["false_positives"], "false_negatives": m["false_negatives"], "true_positives": m["true_positives"]}


def _build_comparison(category: str, metrics: dict, threshold: float, auroc: float, per_defect: dict) -> dict:
    """Read-only comparison with the category's earlier ConvAutoencoder phases - whichever of Phase 1/2/3 exist."""
    root = artifacts.ARTIFACTS_ROOT / category
    p1 = _optional_json(root / "phase1_baseline" / "reports" / "phase1_report.json")
    p2 = _optional_json(root / "phase2_threshold" / "reports" / "final_test_result.json")
    p3 = _optional_json(root / "phase3_validation" / "reports" / "final_test_result.json")
    comparison, earlier = {}, {}
    if p1:
        r1 = p1["run1"]
        comparison["phase1_convae"] = _metric_row(r1["metrics"], r1["threshold"]["value"], r1["separation"]["auroc_descriptive_only"])
        earlier["phase1"] = r1["per_defect"]
    if p2:
        scores = p2["predictions"]
        comparison["phase2_convae"] = _metric_row(p2["metrics"], p2["locked_threshold"], float(roc_auc_score(
            [q["label"] for q in scores], [q["reconstruction_error"] for q in scores])))
        earlier["phase2"] = p2["per_defect"]
    if p3:
        comparison["phase3_convae"] = _metric_row(p3["metrics"], p3["locked_threshold"], p3["auroc_descriptive_only"])
        earlier["phase3"] = p3["per_defect"]
    comparison["alternative"] = _metric_row(metrics, threshold, auroc)
    comparison["per_defect"] = {n: {**{phase: d[n]["recall"] for phase, d in earlier.items() if n in d}, "alternative": v["recall"],
                                    "alternative_detected": v["detected"], "total": v["total"]} for n, v in per_defect.items()}
    return comparison


def evaluate_locked_on_final_test(category: str) -> dict:
    lock, study = verify_lock(category)
    threshold = lock["selected_threshold"]
    cid = lock["selected_candidate"]
    size = lock["selected_configuration"]["image_size"]
    train_samples = discover_train_samples(category)
    lock_file = lock_path(category)
    lock_file_sha = file_hashes(lock_file)["sha256"]

    t0 = time.perf_counter()
    detector = PatchAnomalyDetector.load(selected_dir(category), load_pretrained_resnet18())
    load_ms = (time.perf_counter() - t0) * 1000

    # ---- The protected final test set is first touched HERE, after every check above. ----
    test_listed_at = time.time()
    started = time.perf_counter()
    test_samples = discover_test_samples(category)
    scores, prep_ms, inf_ms = [], [], []
    for sample in test_samples:
        t = time.perf_counter()
        prepared = _build_preprocessing_result(sample.path, _load_image(sample.path), (size, size)).resized_image
        tensor = to_tensor(prepared[None])
        t1 = time.perf_counter()
        scores.extend(detector.score_images(tensor))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t) * 1000)
        inf_ms.append((t2 - t1) * 1000)
    eval_ms = (time.perf_counter() - started) * 1000
    predictions = predictions_at_threshold(test_samples, scores, threshold)
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

    normals = study["candidates"][cid]["summary"]["normal_score_distribution"]
    position = {"test_good": _rank(good, threshold), "test_defective": _rank(bad, threshold),
                "normal_loio_distribution": normals,
                "test_good_above_normal_loio_p99": int((good > normals["p99"]).sum()),
                "test_good_mean_over_normal_loio_mean": float(good.mean() / normals["mean"]),
                "test_good_median_over_normal_loio_median": float(np.median(good) / normals["median"])}

    comparison = _build_comparison(category, metrics, threshold, auroc, per_defect)

    train_paths = {s.path for s in train_samples}
    test_paths = {s.path for s in test_samples}
    hashes_after = {"backbone": file_hashes(pretrained_weights_path())["sha256"], "artifact": file_hashes(selected_dir(category) / "model_state.pt")["sha256"]}
    leakage = {
        "1_final_test_paths_never_used_for_fitting_or_threshold_selection": not (test_paths & train_paths)
        and lock["dataset_counts"]["train_good"] == len(train_samples) and all(s.split == "train" and s.defect_type == "good" for s in train_samples),
        "2_test_contents_disjoint_from_the_normal_pool": not ({hashlib.sha256(p.read_bytes()).hexdigest() for p in test_paths}
                                                              & {hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths}),
        "3_no_test_labels_or_masks_used": lock["final_test_data_used_for_selection"] is False and not any("ground_truth" in str(p) for p in train_paths | test_paths),
        "4_selection_reproduced_from_the_saved_normal_and_synthetic_study_only": mfs.select_configuration(
            {c: r["summary"] for c, r in study["candidates"].items()}, mfs.registry())["winner"] == f"{cid}|{lock['selected_policy']}",
        "5_lock_written_before_the_final_test_was_listed": lock["timestamp_unix"] <= test_listed_at and lock_file.stat().st_mtime <= test_listed_at,
        "6_artifact_and_backbone_hashes_verified_before_and_after": hashes_after["artifact"] == lock["selected_artifact"]["sha256"]
        and hashes_after["backbone"] == lock["backbone"]["sha256"],
        "7_final_test_scored_exactly_once": not result_path(category).exists(),
        "8_phase1_2_3_artifacts_unchanged": prior_phase_reference(category) == lock["prior_phase_artifacts"],
        "9_selection_lock_unchanged_during_the_final_test": file_hashes(lock_file)["sha256"] == lock_file_sha,
    }
    leakage["all_passed"] = all(leakage.values())

    result = {
        "category": category, "experiment_id": lock["experiment_id"], "selected_candidate": cid, "selected_policy": lock["selected_policy"],
        "selected_configuration": lock["selected_configuration"], "locked_threshold": threshold,
        "selection_lock_file_sha256": lock_file_sha, "selection_lock_digest": lock["lock_digest"], "artifact_sha256": hashes_after["artifact"],
        "backbone_sha256": hashes_after["backbone"], "counts": {"test_total": len(predictions), "test_good": int(good.size), "test_defective": int(bad.size)},
        "metrics": metrics, "auroc": auroc, "defect_identification_accuracy_is_recall": metrics["defect_identification_accuracy"] == metrics["recall"],
        "gate_classification": classify_with_fpr(metrics["recall"], metrics["f1_score"], metrics["false_defect_detection_rate"]),
        "per_defect": per_defect, "distributions": {"test_good": describe(good), "test_defective": describe(bad),
                                                     "test_defective_by_type": {n: describe(v) for n, v in sorted(by_type.items())}},
        "threshold_position": position, "comparison": comparison, "leakage_audit": leakage,
        "false_positive_files": [p["file"] for p in predictions if p["label"] == 0 and p["predicted_label"] == 1],
        "missed_files": [p["file"] for p in predictions if p["label"] == 1 and p["predicted_label"] == 0],
        "timing": {"model_load_ms": load_ms, "preprocessing_ms_per_image": statistics.fmean(prep_ms), "inference_ms_per_image": statistics.fmean(inf_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms), "final_test_total_ms": eval_ms},
        "test_listed_at_unix": test_listed_at, "predictions": predictions,
    }
    out = result_path(category)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
