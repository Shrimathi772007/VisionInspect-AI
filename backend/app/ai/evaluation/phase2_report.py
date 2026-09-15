"""Build and persist the Phase 2 threshold-experiment artifact (JSON).

Written to a separate file from the Phase 1 report
(backend/ai_models/<category>/evaluation_reports/phase2_threshold_experiment_report.json)
- Phase 1's phase1_validation_report.json is never overwritten. Same
directory convention as Phase 1: backend/ai_models/ is already gitignored in
its entirety, so this is not committed.
"""

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.ai.evaluation.model_metadata import ModelMetadata
from app.ai.evaluation.threshold_experiments import CandidateTestResult
from app.ai.evaluation.threshold_selection import SelectionResult
from app.ai.training.artifacts import ARTIFACTS_ROOT
from app.ai.training.schemas import DatasetStatistics


def _candidate_result_summary(result: CandidateTestResult) -> dict:
    c = result.candidate
    return {
        "method": c.method,
        "parameter": c.parameter,
        "threshold": c.threshold,
        "calibration_sample_count": c.calibration_sample_count,
        "calibration_mean": c.calibration_mean,
        "calibration_std": c.calibration_std,
        "calibration_percentile": c.calibration_percentile,
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


def build_phase2_report(
    *,
    model_metadata: ModelMetadata,
    dataset_stats: DatasetStatistics,
    git_branch: str,
    git_head: str,
    calibration_fraction: float,
    calibration_seed: int,
    calibration_sample_count: int,
    held_out_sample_count: int,
    phase1_baseline: dict,
    candidate_results: list[CandidateTestResult],
    selection: SelectionResult,
    reproducibility: dict,
    timing_ms: dict,
) -> dict:
    """Assemble the full Phase 2 JSON artifact.

    `candidate_results` includes every candidate's final-test performance,
    included here purely for reporting/comparison - `selection` (built by
    app.ai.evaluation.threshold_selection.select_threshold) was already
    decided from calibration/held-out data alone before these results were
    computed; nothing here retroactively changes that decision.
    """
    selected_summary = None
    if selection.selected is not None:
        selected_summary = next(
            (
                _candidate_result_summary(r)
                for r in candidate_results
                if r.candidate.method == selection.selected.method
                and r.candidate.parameter == selection.selected.parameter
            ),
            None,
        )

    return {
        "report_type": "milestone_4_phase_2_threshold_experiment",
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git": {"branch": git_branch, "head": git_head},
        "model": asdict(model_metadata),
        "dataset": {
            "category": dataset_stats.category,
            "train_good_count": dataset_stats.train_good_count,
            "test_good_count": dataset_stats.test_good_count,
            "test_defective_count": dataset_stats.test_defective_count,
            "test_total_count": dataset_stats.test_total_count,
            "test_defect_type_counts": dataset_stats.test_defect_type_counts,
        },
        "calibration_methodology": {
            "description": (
                "Threshold candidates are computed ONLY from a deterministic subset of "
                "train/good reconstruction errors (the same calibration split established "
                "in Phase 1: fraction=0.8, seed=42). This is a calibration/stability "
                "methodology, not an independent validation methodology - the model was "
                "trained on all 209 train/good images, including the held-out-42 subset, "
                "so held-out-42 is not unseen data. The final 83-image test set "
                "(test/good + test/broken_large + test/broken_small + test/contamination) "
                "is completely separate and untouched until after threshold selection."
            ),
            "calibration_fraction": calibration_fraction,
            "calibration_seed": calibration_seed,
            "calibration_sample_count": calibration_sample_count,
            "held_out_sample_count": held_out_sample_count,
        },
        "phase1_baseline": phase1_baseline,
        "candidates": [_candidate_result_summary(r) for r in candidate_results],
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
                    "held_out_sample_count": s.held_out_sample_count,
                    "held_out_false_positive_count": s.held_out_false_positive_count,
                    "held_out_false_positive_rate": s.held_out_false_positive_rate,
                    "passes_stability_check": s.passes_stability_check,
                }
                for s in selection.stability_by_candidate
            ],
        },
        "reproducibility": reproducibility,
        "timing_ms": timing_ms,
        "leakage_verification": {
            "threshold_statistics_source": "train/good calibration subset only (167 of 209 images)",
            "final_test_used_for_threshold_selection": False,
            "final_test_labels_used_for_threshold_selection": False,
            "selection_decided_before_final_test_evaluation": True,
            "held_out_42_described_as_independent_validation": False,
        },
        "limitations": [
            "The held-out-42 calibration subset is not an independent/unseen validation "
            "set - all 209 train/good images, including these 42, were used to train the "
            "model. Its false-positive rate is a calibration-stability proxy only.",
            "With only 42 held-out calibration images, the held-out false-positive rate "
            "estimate itself has wide sampling uncertainty (each misclassified image moves "
            "the rate by ~2.4 percentage points).",
            "No threshold candidate in this experiment can be called a fully validated "
            "production threshold; genuine validation would require retraining without the "
            "calibration/held-out images in the training set.",
        ],
    }


def default_phase2_report_path(category: str) -> Path:
    return ARTIFACTS_ROOT / category / "evaluation_reports" / "phase2_threshold_experiment_report.json"


def save_phase2_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
