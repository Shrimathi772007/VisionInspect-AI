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
"""

from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, compute_reconstruction_error, evaluate_model
from app.ai.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from app.ai.evaluation.schemas import EvaluationResult, EvaluationSample
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold

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
]
