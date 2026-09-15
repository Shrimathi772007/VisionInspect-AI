"""Evaluate an already-trained anomaly-detection model on held-out test data.

    trained model -> unseen test images -> reconstruction error -> threshold
    (from training-normal errors only) -> good/defective prediction -> real metrics

Does not train, retrain, or modify any model. Ground-truth labels are used
only to score predictions after the fact - never to compute reconstruction
error or to choose the threshold.

Public API:
    compute_reconstruction_error(model, image_tensor) -> float
    evaluate_model(category, model, image_size, threshold_k) -> EvaluationResult
    compute_threshold(normal_errors, k) -> float

Milestone 4 Phase 1 additions (leakage-free, timed, reproducible evaluation -
see app.ai.evaluation.validated_evaluate for the full protocol docstring):
    split_for_calibration(samples, fraction, seed) -> CalibrationSplit
    evaluate_model_validated(category, model, ...) -> ValidatedEvaluationResult
    compute_model_metadata(model_path, category, ...) -> ModelMetadata
    build_phase1_report(...) -> dict / save_report(report, path) -> Path
"""

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_CALIBRATION_SEED,
    CalibrationSplit,
    split_for_calibration,
)
from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, compute_reconstruction_error, evaluate_model
from app.ai.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from app.ai.evaluation.model_metadata import ModelMetadata, compute_model_metadata
from app.ai.evaluation.report import build_phase1_report, default_report_path, save_report
from app.ai.evaluation.schemas import (
    CalibrationInfo,
    EvaluationResult,
    EvaluationSample,
    TimingSummary,
    ValidatedEvaluationResult,
)
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold
from app.ai.evaluation.validated_evaluate import evaluate_model_validated

__all__ = [
    "compute_reconstruction_error",
    "evaluate_model",
    "compute_threshold",
    "compute_classification_metrics",
    "ClassificationMetrics",
    "EvaluationResult",
    "EvaluationSample",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_THRESHOLD_K",
    "split_for_calibration",
    "CalibrationSplit",
    "DEFAULT_CALIBRATION_FRACTION",
    "DEFAULT_CALIBRATION_SEED",
    "evaluate_model_validated",
    "ValidatedEvaluationResult",
    "CalibrationInfo",
    "TimingSummary",
    "compute_model_metadata",
    "ModelMetadata",
    "build_phase1_report",
    "save_report",
    "default_report_path",
]
