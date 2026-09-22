"""Phase 2 threshold-robustness / calibration study for an ALREADY-SELECTED, FROZEN model-family candidate.

This is not a new model-family study and not a final-test optimization. The candidate (backbone, layers, image
size, pooling, aggregation, scorer family, shrinkage) is read from the category's existing, hash-locked
`selection_lock.json` and is never changed here: no fine-tuning, no refitting outside the established per-split
calibration procedure, no new candidates. The module reuses app.ai.evaluation.model_family_study (splits, threshold
policies, Gaussian/kNN scoring, synthetic diagnostic) and app.ai.evaluation.model_family_pipeline (`load_inputs`,
`evaluate_group`) exactly as the model-family study does; it adds only the threshold-selection rule below.

It cannot reach the final test set: it never imports test discovery (asserted from source by tests). The only touch
of the test tree is via `load_inputs` -> `count_dataset_entries`, which counts directory entries per folder (no
image is opened, no file name is kept) so the mandatory dataset-count verification can run - identical to what the
model-family study already does before its own lock.

=====================================================================================================================
PRE-DECLARED THRESHOLD-SELECTION RULE  (written before any Phase 2 result was computed; not to be altered afterwards)
=====================================================================================================================
Candidates for the threshold: the established 8 policies already used by the model-family study - mean + K*std
(population std) for K in {1.5, 2.0, 2.5, 3.0}, and percentiles P95/P97/P98/P99 (linear interpolation) - evaluated
for the ONE frozen candidate on the SAME 30 normal-only splits and the SAME synthetic diagnostic as the model-family
study (identical split fingerprint). No new threshold values are introduced. The currently locked policy (mean+3sigma)
is the BASELINE.

For each policy, "conservativeness" is measured by its FULL-POOL threshold (fitted on all 267 train/good images'
leave-one-image-out scores) - a lower threshold is less conservative (flags more images, higher expected recall AND
higher expected FPR) - a single, comparable scale because every policy scores the same frozen candidate.

Selection is sequential and mechanical:
  1. GATE       keep only policies whose worst-fold FPR over the 30 splits is <= 10% (the same normal-data
                robustness gate used by the model-family study). If none pass, classification B (no stable
                threshold): no threshold change is justified by the calibration data.
  2. STABILITY  among the gate-passing policies, find the lowest worst-fold FPR achieved ("best_worst_fold_fpr").
                Keep only policies within a 2-percentage-point BAND of that best value: these are "comparably
                stable" to the most robust gate-passing policy.
  3. LEAST CONSERVATIVE  among the comparably-stable policies, select the one with the lowest full-pool threshold
                (deterministic tie-break: policy id, alphabetical). This is the least conservative threshold that
                remains supported by the normal-only stability evidence.
  4. CLASSIFY   if the selected policy IS the locked baseline (mean+3sigma), classification B: the baseline is
                already the least conservative policy within the comparably-stable band, so no threshold change is
                justified. Otherwise classification A: the selected, less-conservative policy is calibration-
                supported for FUTURE validation on a new held-out defect set - it is NOT validated against, and must
                never be validated against, the already-spent Pill final test.
  5. Synthetic diagnostic recall/AUROC are recorded for every policy for reference only; they play no role in steps
     1-4 and never override the normal-only stability evidence (case C in the brief: if a lower-FPR-supported
     policy does not exist, an attractive synthetic recall alone is NOT grounds to relax the threshold).

MANDATORY STATEMENT  A policy passing worst-fold FPR <= 10% on the normal training pool does not establish that the
real Pill final-test FPR would be <= 10% - the final test was already spent selecting the candidate and cannot be
reused to check this. Likewise, synthetic recall is a diagnostic on synthetic corruptions of real Pill NORMAL images
and is not real Pill defect recall. Nothing computed here touches, decodes, or is validated against the protected
final test; the frozen selection lock, the frozen selected artifact and the frozen final-test result are read only
to verify they are unchanged, and are never rewritten.
=====================================================================================================================
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from app.ai.evaluation import model_family_study as mfs
from app.ai.evaluation.category_phase1 import file_hashes
from app.ai.evaluation.model_family_pipeline import (
    StudyInputs,
    evaluate_group,
    lock_digest,
    lock_path,
    reports_dir as study_reports_dir,
    selected_dir,
)
from app.ai.training import artifacts

PHASE2_DIRNAME = "phase2_threshold_calibration"
STABILITY_BAND_FPR = 0.02  # 2 percentage points


# ---------------------------------------------------------------------------
# Paths (a directory separate from model_family_study/, never written to by it)
# ---------------------------------------------------------------------------

def phase2_root(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / PHASE2_DIRNAME


def phase2_candidates_dir(category: str) -> Path:
    return phase2_root(category) / "candidates"


def phase2_reports_dir(category: str) -> Path:
    return phase2_root(category) / "reports"


def phase2_report_path(category: str) -> Path:
    return phase2_reports_dir(category) / "phase2_threshold_report.json"


# ---------------------------------------------------------------------------
# Frozen-state verification (before and after) - never mutates anything it reads
# ---------------------------------------------------------------------------

def read_frozen_state(category: str) -> dict:
    """Hashes of the three artifacts that Phase 2 must never change. Reads only; no test image is touched."""
    lock = json.loads(lock_path(category).read_text(encoding="utf-8"))
    result_file = study_reports_dir(category) / "final_test_result.json"
    return {
        "selection_lock_sha256": file_hashes(lock_path(category))["sha256"],
        "selection_lock_digest_valid": lock_digest(lock) == lock["lock_digest"],
        "selected_candidate": lock["selected_candidate"], "selected_policy": lock["selected_policy"],
        "selected_threshold": lock["selected_threshold"],
        "selected_artifact_sha256": file_hashes(selected_dir(category) / "model_state.pt")["sha256"],
        "final_test_result_sha256": file_hashes(result_file)["sha256"] if result_file.is_file() else None,
        "final_test_confusion_matrix": json.loads(result_file.read_text(encoding="utf-8"))["metrics"]["confusion_matrix"]
        if result_file.is_file() else None,
    }


def verify_frozen_state_unchanged(before: dict, after: dict) -> dict:
    checks = {
        "selection_lock_unchanged": before["selection_lock_sha256"] == after["selection_lock_sha256"],
        "selected_artifact_unchanged": before["selected_artifact_sha256"] == after["selected_artifact_sha256"],
        "selected_candidate_and_policy_unchanged": (before["selected_candidate"], before["selected_policy"], before["selected_threshold"])
        == (after["selected_candidate"], after["selected_policy"], after["selected_threshold"]),
        "final_test_result_unchanged": before["final_test_result_sha256"] == after["final_test_result_sha256"],
        "final_test_confusion_matrix_unchanged": before["final_test_confusion_matrix"] == after["final_test_confusion_matrix"],
    }
    checks["all_passed"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Threshold selection (pure; operates on the already-computed per-policy summary of the ONE frozen candidate)
# ---------------------------------------------------------------------------

def _pooled_fpr(e: dict) -> np.ndarray:
    return np.array([x["fpr"] for sch in e["per_scheme"].values() for x in sch["splits"]])


def select_calibration_threshold(policies: dict, baseline_policy: str, gate_fpr: float = mfs.GATE_FPR,
                                 band: float = STABILITY_BAND_FPR) -> dict:
    rows = []
    for p, e in policies.items():
        fpr = _pooled_fpr(e)
        rows.append({
            "policy": p, "threshold_full_pool": e["full_pool_threshold"], "mean_fpr": e["mean_fpr"],
            "median_fpr": float(np.median(fpr)), "worst_fold_fpr": e["worst_fold_fpr"], "fpr_std": float(np.std(fpr)),
            "threshold_mean": e["pooled_threshold"]["mean"], "threshold_std": e["pooled_threshold"]["std"],
            "threshold_cv": e["pooled_threshold"]["cv"], "threshold_min": e["pooled_threshold"]["min"],
            "threshold_max": e["pooled_threshold"]["max"], "passes_gate": e["worst_fold_fpr"] <= gate_fpr,
            "splits_fpr_le_5pct": int((fpr <= 0.05).sum()), "splits_fpr_le_10pct": int((fpr <= 0.10).sum()),
            "splits_fpr_le_15pct": int((fpr <= 0.15).sum()), "splits_total": int(fpr.size),
            "synthetic_recall": e.get("synthetic_recall"), "synthetic_recall_by_type": e.get("synthetic_recall_by_type"),
            "synthetic_recall_by_severity": e.get("synthetic_recall_by_severity"),
        })
    rows_in_registry_order = [r for pid in mfs.policy_ids() for r in rows if r["policy"] == pid]
    baseline = next(r for r in rows if r["policy"] == baseline_policy)
    gate_passers = [r for r in rows if r["passes_gate"]]

    if not gate_passers:
        return {"rows": rows_in_registry_order, "gate_passers": [], "stability_supported": [], "best_worst_fold_fpr_among_gate_passers": None,
                "baseline_policy": baseline_policy, "baseline": baseline, "selected_policy": None, "selected_threshold": None,
                "classification": "B_no_stable_threshold",
                "reasoning": f"No threshold policy satisfied the worst-fold FPR <= {gate_fpr:.0%} gate on the 30 normal-only splits; "
                             "no threshold change is justified by the calibration data."}

    best_worst_fpr = min(r["worst_fold_fpr"] for r in gate_passers)
    stability_supported = sorted((r for r in gate_passers if r["worst_fold_fpr"] <= best_worst_fpr + band),
                                 key=lambda r: (r["threshold_full_pool"], r["policy"]))
    chosen = stability_supported[0]
    out = {"rows": rows_in_registry_order, "gate_passers": [r["policy"] for r in gate_passers],
           "stability_supported": [r["policy"] for r in stability_supported], "best_worst_fold_fpr_among_gate_passers": best_worst_fpr,
           "baseline_policy": baseline_policy, "baseline": baseline, "selected_policy": chosen["policy"],
           "selected_threshold": chosen["threshold_full_pool"]}
    if chosen["policy"] == baseline_policy:
        out["classification"] = "B_no_stable_threshold"
        out["reasoning"] = (f"The locked baseline ({baseline_policy}, threshold {baseline['threshold_full_pool']:.5g}, worst-fold FPR "
                            f"{baseline['worst_fold_fpr']:.2%}) already has the lowest worst-fold FPR among the gate-passing policies; "
                            f"the alternative gate-passing policies within {band:.0%} of that value do not include a less-conservative one. "
                            "No threshold change is justified by the calibration data.")
    else:
        out["classification"] = "A_calibration_supported_lower_threshold"
        out["reasoning"] = (f"{chosen['policy']} (threshold {chosen['threshold_full_pool']:.5g}, worst-fold FPR {chosen['worst_fold_fpr']:.2%}) "
                            f"passes the worst-fold FPR <= {gate_fpr:.0%} gate and is within {band:.0%} of the best gate-passing worst-fold FPR "
                            f"({best_worst_fpr:.2%}); it is less conservative than the locked baseline {baseline_policy} (threshold "
                            f"{baseline['threshold_full_pool']:.5g}, worst-fold FPR {baseline['worst_fold_fpr']:.2%}). This is calibration-"
                            "supported evidence for FUTURE validation only; it has NOT been evaluated against, and is not proven better on, "
                            "the already-spent Pill final test.")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def frozen_candidate_spec(category: str) -> mfs.CandidateSpec:
    """The single candidate this Phase 2 study evaluates, read from the existing (untouched) selection lock."""
    lock = json.loads(lock_path(category).read_text(encoding="utf-8"))
    cid = lock["selected_candidate"]
    return next(s for s in mfs.registry() if s.candidate_id == cid), lock["selected_policy"], lock


def run_threshold_study(inputs: StudyInputs, spec: mfs.CandidateSpec, baseline_policy: str, out_dir: Path,
                        log=print) -> dict:
    t0 = time.perf_counter()
    result = evaluate_group(inputs, [spec], out_dir, log)[spec.candidate_id]
    summary = result["summary"]
    selection = select_calibration_threshold(summary["policies"], baseline_policy)
    return {
        "category": inputs.category, "frozen_candidate": spec.config(), "baseline_policy": baseline_policy,
        "dataset_fingerprint_sha256": inputs.dataset_fingerprint, "splits_sha256": mfs.splits_fingerprint(inputs.schemes),
        "split_audit": mfs.split_audit(len(inputs.names), inputs.schemes),
        "synthetic": {"images": len(inputs.synthetic_meta), "digest_sha256": inputs.synthetic_digest,
                      "held_out_images": len(inputs.canonical_held_out), "variants_per_image": len(mfs.DEFECT_TYPES) * len(mfs.SEVERITIES),
                      "label": "DIAGNOSTIC ONLY - not real Pill defect recall"},
        "synthetic_auroc": summary["synthetic_auroc"], "recomputed_artifact_sha256": result["artifact"]["sha256"],
        "declared_rule": __doc__, "selection_rule_sha256": hashlib.sha256(__doc__.encode("utf-8")).hexdigest(),
        "selection": selection, "cost": result["cost"], "study_seconds": time.perf_counter() - t0,
    }


def compare_threshold_studies(a: dict, b: dict) -> dict:
    canon = lambda x: json.loads(json.dumps(x, sort_keys=True))  # noqa: E731
    return {
        "dataset_fingerprint_identical": a["dataset_fingerprint_sha256"] == b["dataset_fingerprint_sha256"],
        "split_fingerprint_identical": a["splits_sha256"] == b["splits_sha256"],
        "synthetic_digest_identical": a["synthetic"]["digest_sha256"] == b["synthetic"]["digest_sha256"],
        "recomputed_artifact_identical": a["recomputed_artifact_sha256"] == b["recomputed_artifact_sha256"],
        "selection_identical": canon(a["selection"]) == canon(b["selection"]),
        "selected_policy_identical": a["selection"]["selected_policy"] == b["selection"]["selected_policy"],
        "selected_threshold_identical": a["selection"]["selected_threshold"] == b["selection"]["selected_threshold"],
        "classification_identical": a["selection"]["classification"] == b["selection"]["classification"],
    }
