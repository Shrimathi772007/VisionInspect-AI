"""Build the per-category Phase 4 optimization report (JSON, Git-ignored with backend/ai_models/).

Records, for every trained candidate: configuration, model hashes and size, training loss, validation
statistics (incl. CoV), every threshold candidate with its validation false-positive count/rate, the
threshold it would select, and timings - plus the stage selections, the frozen selection, the single
final-test result (only when a non-baseline candidate was selected), the baseline comparison and
adoption decision, reproducibility and leakage audit.
"""

import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.ai.evaluation.category_phase4 import CandidateOutcome, ExperimentResult
from app.ai.evaluation.category_phase4_final import (
    ADOPTION_MAX_FALSE_POSITIVE_RATE,
    ADOPTION_SIGN_TEST_ALPHA,
    FinalTestResult,
)
from app.ai.training import artifacts

PHASE4_DIR_NAME = "phase4_optimization"


def phase4_root(category: str) -> Path:
    """backend/ai_models/<category>/phase4_optimization (root resolved at call time)."""
    return artifacts.ARTIFACTS_ROOT / category / PHASE4_DIR_NAME


def ensure_no_phase4_overwrite(category: str) -> None:
    """Refuse to start if this category already has Phase 4 output - nothing is ever overwritten."""
    root = phase4_root(category)
    if root.exists():
        raise FileExistsError(f"Phase 4 output already exists for category '{category}': {root}. Refusing to overwrite.")


def phase4_report_path(category: str) -> Path:
    return phase4_root(category) / "reports" / "phase4_optimization_report.json"


def _candidate_record(o: CandidateOutcome) -> dict:
    tc = o.config.training_config
    stability = {(s.candidate.method, s.candidate.parameter): s for s in o.selection.stability_by_candidate}
    return {
        "candidate_id": o.candidate_id,
        "stage_first_used": o.config.stage,
        "configuration": {
            "architecture": tc.model.architecture,
            "latent_channels": tc.model.latent_channels,
            "image_size": list(tc.image_size),
            "epochs": tc.epochs,
            "learning_rate": tc.learning_rate,
            "batch_size": tc.batch_size,
            "seed": tc.seed,
            "device": tc.device,
        },
        "reused_protected_phase3_artifact": o.reused_phase3_artifact,
        "model": {
            "path": str(o.model_path),
            "md5": o.model_metadata.md5,
            "sha256": o.model_metadata.sha256,
            "size_bytes": o.model_metadata.artifact_size_bytes,
        },
        "training": {
            "duration_seconds": o.training_duration_s,
            "final_loss": o.final_training_loss,
            "num_training_images": o.num_training_images,
            "loss_history": o.loss_history,
        },
        "validation": {
            "sample_count": len(o.validation_errors),
            "mean_error": o.validation_mean,
            "std_error": o.validation_std,
            "min_error": o.validation_min,
            "max_error": o.validation_max,
            "coefficient_of_variation": o.coefficient_of_variation,
            "threshold_candidates": [
                {
                    "method": c.method,
                    "parameter": c.parameter,
                    "threshold": c.threshold,
                    "validation_false_positive_count": stability[(c.method, c.parameter)].validation_false_positive_count,
                    "validation_false_positive_rate": stability[(c.method, c.parameter)].validation_false_positive_rate,
                    "passes_10_percent_ceiling": stability[(c.method, c.parameter)].passes_stability_check,
                }
                for c in o.threshold_candidates
            ],
            "threshold_selection_status": o.selection.status,
            "selected_threshold": None if o.selection.selected is None else {
                "method": o.selection.selected.method,
                "parameter": o.selection.selected.parameter,
                "threshold": o.selection.selected.threshold,
            },
        },
        "timing": {
            "training_duration_s": o.training_duration_s,
            "model_load_ms": o.model_load_ms,
            "validation_evaluation_ms": o.validation_evaluation_ms,
            "validation_ms_per_image_total": o.validation_evaluation_ms / len(o.validation_errors),
            "validation_preprocess_ms_per_image": o.validation_preprocess_ms_per_image,
            "validation_inference_ms_per_image": o.validation_inference_ms_per_image,
        },
    }


def build_category_phase4_report(
    *,
    category: str,
    git_branch: str,
    git_head: str,
    baseline: dict,
    experiment: ExperimentResult,
    reproducibility: dict,
    final: FinalTestResult | None,
    comparison: dict | None,
    decision: str,
    leakage: dict,
    experiment2_total_ms: float,
    protected_artifacts: dict,
) -> dict:
    locked = experiment.locked
    return {
        "report_type": "category_phase4_optimization",
        "category": category,
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git": {"branch": git_branch, "head": git_head},
        "decision": decision,
        "serving_status": "NOT registered for production serving (app.ai.inference.serving).",
        "protected_baseline": baseline,
        "protected_artifacts_unchanged": protected_artifacts,
        "selection_protocol": {
            "stages": {
                "stage1_input_size": "128 / 160 / 192 at epochs 15, lr 1e-3",
                "stage2_epochs": "15 / 25 / 35 at the Stage 1 winner's size, lr 1e-3",
                "stage3_learning_rate": "1e-3 / 5e-4 at the Stage 1+2 winning configuration",
            },
            "eligibility": "candidates with no threshold option under the 10% validation FP ceiling are disqualified",
            "ranking": "ascending validation coefficient of variation (std/mean of reconstruction error on good-only validation images)",
            "tie_rule": "within 2% relative CoV: fewer epochs, smaller input size, smaller model file, learning rate nearest baseline, candidate id (wall-clock time is never a tie-breaker)",
            "threshold_rule": "lowest threshold candidate with validation FP rate <= 10% (existing Phase 3 rule), on the winner's own validation errors",
            "adoption_criteria_vs_baseline": {
                "A1": "final-test F1 strictly higher than baseline",
                "A2": f"final-test FPR <= {ADOPTION_MAX_FALSE_POSITIVE_RATE:.0%}",
                "A3": f"one-sided exact sign test on disagreeing defective images, p < {ADOPTION_SIGN_TEST_ALPHA}",
            },
            "limitation": "Validation holds only good images, so it cannot measure recall/F1. CoV is an engineering "
            "consistency signal, not a proven predictor of defect detection. Training loss is never used to select.",
        },
        "candidates": [_candidate_record(o) for o in experiment.outcomes.values()],
        "stage_selections": [asdict(s) for s in experiment.stage_selections],
        "frozen_selection": {
            "candidate_id": locked.candidate_id,
            "image_size": list(locked.image_size),
            "epochs": locked.epochs,
            "learning_rate": locked.learning_rate,
            "model_md5": locked.model_md5,
            "threshold_method": locked.threshold_method,
            "threshold_parameter": locked.threshold_parameter,
            "threshold": locked.threshold,
            "is_baseline_configuration": locked.is_baseline_configuration,
            "digest": locked.digest,
        },
        "final_test": None
        if final is None
        else {
            "note": "Scored exactly once, after the selection was frozen and both experiment runs were verified identical.",
            "metrics": final.metrics,
            "recall_by_defect_type": final.recall_by_defect_type,
            "predictions": final.predictions,
            "timing": {
                "total_ms": final.total_ms,
                "ms_per_image_total": final.total_ms / len(final.predictions),
                "preprocess_ms_per_image": final.preprocess_ms_per_image,
                "inference_ms_per_image": final.inference_ms_per_image,
            },
        },
        "baseline_comparison": comparison,
        "reproducibility": {"experiment_runs": 2, **reproducibility},
        "leakage_verification": leakage,
        "timing": {"experiment_run1_total_ms": experiment.total_ms, "experiment_run2_total_ms": experiment2_total_ms},
        "mAP": "Not applicable: reconstruction-error classifier with no localization/detection output.",
        "limitations": [
            "The hypothesis that input size, epochs or learning rate limit Hazelnut detection is untested "
            "beyond this experiment; no mechanism is claimed.",
            "Selection uses a good-only validation signal (CoV); it can pick a candidate that is not better at "
            "finding defects.",
            "110 final-test images (40 good, 70 defective) is a small sample; small differences are not "
            "statistically meaningful.",
            "Timings are CPU-only wall-clock on this machine for Hazelnut and are not extrapolated.",
        ],
    }
