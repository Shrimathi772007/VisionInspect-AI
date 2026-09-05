"""AI defect prediction (inference) for one inspection image at a time.

    inspection image -> existing preprocessing -> existing trained model
    -> reconstruction error -> existing Phase 5 threshold -> good/defective prediction

Reuses, rather than duplicates, the Phase 2 preprocessing pipeline, the Phase 4 trained
model artifacts, and the Phase 5 reconstruction-error/threshold logic. Does not train,
retrain, or otherwise modify any model, and never uses MVTec ground-truth labels to
influence a prediction - see app.ai.evaluation for the separate checkpoint that measures
AI-prediction-vs-ground-truth agreement.

Public API:
    predict_image(image_path, category, ...) -> PredictionResult
"""

from app.ai.inference.errors import InferenceError, ModelArtifactNotFoundError
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
]
