"""AI defect prediction (inference) for one inspection image at a time.

    inspection image -> the category's preprocessing -> category's configured trained model
    -> anomaly score -> category's configured locked threshold -> good/defective

The model and threshold for each category come from app.ai.inference.serving - every MVTec
category has one registered final model, and its `gate` says how reliable it is. Reuses, rather
than duplicates, the existing preprocessing pipelines and the Phase 5 reconstruction-error logic. Does not train,
retrain, or otherwise modify any model, and never uses MVTec ground-truth labels to
influence a prediction - see app.ai.evaluation for the separate checkpoint that measures
AI-prediction-vs-ground-truth agreement.

Public API:
    predict_image(image_path, category, ...) -> PredictionResult
"""

from app.ai.inference.errors import InferenceError, ModelArtifactNotFoundError, ModelIntegrityError
from app.ai.inference.predict import SUPPORTED_CATEGORIES, predict_image
from app.ai.inference.schemas import DEFECTIVE_PREDICTION, GOOD_PREDICTION, PredictionResult

__all__ = [
    "predict_image",
    "SUPPORTED_CATEGORIES",
    "PredictionResult",
    "GOOD_PREDICTION",
    "DEFECTIVE_PREDICTION",
    "InferenceError",
    "ModelArtifactNotFoundError",
    "ModelIntegrityError",
]
