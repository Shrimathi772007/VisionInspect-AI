"""Data structures produced by AI inference (single-image prediction).

Kept intentionally separate from app.ai.evaluation.schemas: an EvaluationResult
measures how well the AI agrees with MVTec ground truth across a whole test
set; a PredictionResult is just the AI's own decision for one new image, with
no ground-truth label involved at all.
"""

from dataclasses import dataclass
from typing import Any

GOOD_PREDICTION = "good"
DEFECTIVE_PREDICTION = "defective"


@dataclass
class PredictionResult:
    """One AI anomaly-detection prediction - no ground truth, no confidence score.

    `reconstruction_error` and `threshold` are the real, measured/derived
    values the decision was made from; `prediction` is a deterministic
    function of the two (see app.ai.inference.predict.predict_image), not a
    calibrated probability. For a patch-anomaly category (e.g. Tile)
    `reconstruction_error` holds that model's anomaly score - the field keeps
    its name because it is the existing API/database contract.
    """

    category: str
    prediction: str  # "good" or "defective" - see GOOD_PREDICTION / DEFECTIVE_PREDICTION
    reconstruction_error: float
    threshold: float
    model_name: str = "autoencoder"
    input_size: tuple[int, int] = (128, 128)
    processing_time_ms: float = 0.0
    # Only with predict_image(..., return_patch_scores=True) for a patch model: the (h, w) float32
    # torch grid of per-patch anomaly scores the image score was aggregated from, and how the
    # model's input maps onto the original image ("crop224", "full256", "full320", or "resize"
    # for the ResNet-18 whole-image resize). None otherwise (ConvAE, or not requested).
    patch_scores: Any = None
    input_mode: str | None = None
