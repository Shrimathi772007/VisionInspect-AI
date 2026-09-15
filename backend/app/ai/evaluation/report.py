"""Build and persist the Phase 1 evaluation artifact (JSON).

Deliberately concise: aggregate metrics, counts, timing, and model/dataset
identity only - not a per-sample dump (209 + 83 rows would bloat the file
without adding audit value beyond what the aggregate stats already show).

Written under backend/ai_models/<category>/evaluation_reports/, alongside
the model artifact itself - backend/ai_models/ is already gitignored in its
entirety, so this follows the project's existing convention for generated,
non-committed AI artifacts without needing a new ignore rule.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.ai.evaluation.schemas import EvaluationResult, ValidatedEvaluationResult
from app.ai.evaluation.model_metadata import ModelMetadata
from app.ai.training.artifacts import ARTIFACTS_ROOT
from app.ai.training.schemas import DatasetStatistics


def _evaluation_summary(result: EvaluationResult) -> dict:
    return {
        "threshold": result.threshold,
        "threshold_k": result.threshold_k,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "confusion_matrix": result.confusion_matrix,
        "true_negatives": result.true_negatives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "true_positives": result.true_positives,
        "total_test_samples": result.total_test_samples,
        "good_test_count": result.good_test_count,
        "defective_test_count": result.defective_test_count,
        "num_correct": result.num_correct,
        "num_incorrect": result.num_incorrect,
        "min_error": result.min_error,
        "max_error": result.max_error,
        "mean_error": result.mean_error,
        "mean_good_error": result.mean_good_error,
        "mean_defective_error": result.mean_defective_error,
    }


def _validated_summary(result: ValidatedEvaluationResult) -> dict:
    summary = {
        "threshold": result.threshold,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "confusion_matrix": result.confusion_matrix,
        "true_negatives": result.true_negatives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "true_positives": result.true_positives,
        "total_test_samples": result.total_test_samples,
        "good_test_count": result.good_test_count,
        "defective_test_count": result.defective_test_count,
        "num_correct": result.num_correct,
        "num_incorrect": result.num_incorrect,
        "min_error": result.min_error,
        "max_error": result.max_error,
        "mean_error": result.mean_error,
        "mean_good_error": result.mean_good_error,
        "mean_defective_error": result.mean_defective_error,
        "calibration": asdict(result.calibration),
    }
    if result.timing is not None:
        summary["timing_ms"] = asdict(result.timing)
    return summary


def build_phase1_report(
    model_metadata: ModelMetadata,
    dataset_stats: DatasetStatistics,
    training_seed: int,
    baseline: EvaluationResult,
    validated: ValidatedEvaluationResult,
) -> dict:
    """Assemble the full Phase 1 evaluation artifact as a JSON-serializable dict.

    `baseline` reproduces the original, documented, train-good-based
    threshold methodology unchanged (app.ai.evaluation.evaluate_model);
    `validated` is the Phase 1 calibration-split protocol
    (app.ai.evaluation.validated_evaluate.evaluate_model_validated). Both are
    included, clearly labeled, so the artifact never silently substitutes one
    for the other.
    """
    return {
        "report_type": "milestone_4_phase_1_ai_evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": asdict(model_metadata),
        "dataset": {
            "category": dataset_stats.category,
            "train_good_count": dataset_stats.train_good_count,
            "test_good_count": dataset_stats.test_good_count,
            "test_defective_count": dataset_stats.test_defective_count,
            "test_total_count": dataset_stats.test_total_count,
            "test_defect_type_counts": dataset_stats.test_defect_type_counts,
        },
        "training_seed": training_seed,
        "protocol_notes": (
            "TRAIN = category/train/good (all images; unchanged, not retrained "
            "for this report). CALIBRATION = a deterministic subset of "
            "train/good reconstruction errors (see app.ai.evaluation.calibration), "
            "used only to compute the anomaly threshold. FINAL TEST = "
            "category/test/* (good + every defect-type folder), identical for "
            "both protocols below and never used to select the threshold. "
            "'baseline' reproduces the original threshold methodology (all "
            "train/good errors); 'validated' uses the calibration-subset "
            "threshold. Numbers are not necessarily identical - see each "
            "section's own threshold value."
        ),
        "baseline_reproduction": _evaluation_summary(baseline),
        "validated_phase1_evaluation": _validated_summary(validated),
    }


def default_report_path(category: str) -> Path:
    return ARTIFACTS_ROOT / category / "evaluation_reports" / "phase1_validation_report.json"


def save_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
