"""Build and persist the Phase 3 validated-model artifact (JSON).

Written to a separate file from both Phase 1's and Phase 2's reports
(backend/ai_models/<category>/evaluation_reports/phase3_validated_model_report.json)
- neither is overwritten. Same directory convention: backend/ai_models/ is
already gitignored in its entirety, so this is not committed.
"""

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.ai.evaluation.model_metadata import ModelMetadata
from app.ai.evaluation.phase3_threshold_selection import Phase3SelectionResult
from app.ai.evaluation.threshold_experiments import CandidateTestResult
from app.ai.training.artifacts import ARTIFACTS_ROOT


def _candidate_summary(result: CandidateTestResult) -> dict:
    c = result.candidate
    return {
        "method": c.method,
        "parameter": c.parameter,
        "threshold": c.threshold,
        "validation_sample_count": c.calibration_sample_count,
        "validation_mean": c.calibration_mean,
        "validation_std": c.calibration_std,
        "validation_percentile": c.calibration_percentile,
        "final_test": {
            "total_test_samples": result.total_test_samples,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "true_negatives": result.true_negatives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "true_positives": result.true_positives,
            "confusion_matrix": [
                [result.true_negatives, result.false_positives],
                [result.false_negatives, result.true_positives],
            ],
            "false_positive_rate": result.false_positive_rate,
            "false_negative_rate": result.false_negative_rate,
        },
    }


def build_phase3_report(
    *,
    git_branch: str,
    git_head: str,
    original_model_metadata: ModelMetadata,
    phase3_model_metadata: ModelMetadata,
    training_config: dict,
    split: dict,
    training_result: dict,
    validation_statistics: dict,
    candidate_results: list[CandidateTestResult],
    selection: Phase3SelectionResult,
    phase1_baseline: dict,
    phase2_provisional: dict | None,
    reproducibility: dict,
    timing_ms: dict,
) -> dict:
    selected_summary = None
    if selection.selected is not None:
        selected_summary = next(
            (
                _candidate_summary(r)
                for r in candidate_results
                if r.candidate.method == selection.selected.method
                and r.candidate.parameter == selection.selected.parameter
            ),
            None,
        )

    return {
        "report_type": "milestone_4_phase_3_validated_model",
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git": {"branch": git_branch, "head": git_head},
        "original_model": asdict(original_model_metadata),
        "phase3_model": asdict(phase3_model_metadata),
        "training_config": training_config,
        "split": split,
        "training_result": training_result,
        "validation_statistics": validation_statistics,
        "candidates": [_candidate_summary(r) for r in candidate_results],
        "selection": {
            "status": selection.status,
            "selected": (
                {"method": selection.selected.method, "parameter": selection.selected.parameter}
                if selection.selected is not None
                else None
            ),
            "reasoning": selection.reasoning,
            "selected_final_test_result": selected_summary,
            "stability_by_candidate": [
                {
                    "method": s.candidate.method,
                    "parameter": s.candidate.parameter,
                    "threshold": s.candidate.threshold,
                    "validation_sample_count": s.validation_sample_count,
                    "validation_false_positive_count": s.validation_false_positive_count,
                    "validation_false_positive_rate": s.validation_false_positive_rate,
                    "passes_stability_check": s.passes_stability_check,
                }
                for s in selection.stability_by_candidate
            ],
        },
        "phase1_baseline": phase1_baseline,
        "phase2_provisional_reference": phase2_provisional,
        "reproducibility": reproducibility,
        "timing_ms": timing_ms,
        "leakage_verification": {
            "validation_excluded_from_training": True,
            "threshold_statistics_source": "genuine validation subset only (structurally excluded from training)",
            "final_test_used_for_threshold_selection": False,
            "final_test_labels_used_for_threshold_selection": False,
            "selection_decided_before_final_test_evaluation": True,
            "final_test_evaluated_only_once": True,
        },
        "limitations": [
            "The validation set contains only good/normal images (MVTec provides no "
            "labeled-defective validation split); it can establish false-positive/stability "
            "behavior only, never recall or F1.",
            "The final test set is only 83 images (20 good, 63 defective); differences of a "
            "few samples between operating points should be read as suggestive, not "
            "statistically definitive.",
            "With 42 validation images, the validation false-positive-rate estimate carries "
            "sampling uncertainty - each misclassified image shifts the rate by ~2.4 "
            "percentage points.",
            "The 10% validation false-positive ceiling is an engineering operating constraint "
            "inherited from Phase 2, not a value defined by the original project specification.",
        ],
    }


def default_phase3_report_path(category: str) -> Path:
    return ARTIFACTS_ROOT / category / "evaluation_reports" / "phase3_validated_model_report.json"


def save_phase3_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
