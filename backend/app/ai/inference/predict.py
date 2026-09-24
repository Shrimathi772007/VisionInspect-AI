"""Single-image AI inference using a category's configured, validated model and threshold.

    inspection image -> existing preprocessing -> category's configured trained model
    (cached) -> reconstruction error -> category's configured validated threshold
    -> good/defective

Which model and which threshold serve a category is decided entirely by
app.ai.inference.serving (SERVING_CONFIGS) - a constant-time lookup. Inference never
derives a threshold from data: it does not touch train/good (or any other dataset) images,
so its cost no longer grows with the size of a category's training set.

Reuses, rather than duplicates:
- app.ai.preprocessing.process_image for image loading/validation/preprocessing
  (same decode -> RGB -> resize -> normalize pipeline used everywhere else).
- app.ai.evaluation.evaluate.compute_reconstruction_error for the reconstruction-error
  definition - identical to the one used by every evaluation.

Does not train, retrain, or otherwise modify any model or its architecture, and never
uses MVTec ground-truth labels to influence a prediction. See app.ai.evaluation for the
separate checkpoint that measures AI-prediction-vs-ground-truth agreement.
"""

import time
from pathlib import Path

import torch

from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, compute_reconstruction_error
from app.ai.inference.errors import ModelArtifactNotFoundError
from app.ai.inference.schemas import DEFECTIVE_PREDICTION, GOOD_PREDICTION, PredictionResult
from app.ai.inference.serving import get_serving_config, get_supported_categories, load_serving_model
from app.ai.preprocessing import process_image

# Categories with a configured, validated model today (a snapshot of SERVING_CONFIGS at
# import time; get_supported_categories() is the live view). predict_image itself is
# generic: a category is served if and only if it has a serving configuration.
SUPPORTED_CATEGORIES = get_supported_categories()


def _image_to_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Model-ready CHW tensor for one image, via the existing preprocessing pipeline."""
    result = process_image(path, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def predict_image(
    image_path: Path,
    category: str,
    model_name: str = "autoencoder",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    threshold_k: float | None = None,
) -> PredictionResult:
    """Run AI anomaly-detection inference for one inspection image.

    Looks up `category`'s serving configuration, reconstructs the image with its configured
    autoencoder, and compares the reconstruction error against its configured validated
    threshold (error <= threshold -> "good", otherwise "defective"). Takes no ground-truth
    label and produces none - this is an AI prediction, not an evaluation against MVTec
    ground truth.

    `model_name` and `threshold_k` exist only so pre-serving-registry callers keep working
    (same names and positions as before). Neither can select a different model or threshold -
    the serving configuration is authoritative:
      - `model_name` must equal the category's configured model name ("autoencoder", which
        is also the default). Any other value is refused, never used to pick an artifact.
      - `threshold_k` must be None. The old K-sigma threshold (recomputed from all train/good
        images on every call) no longer exists; silently ignoring a K would hand the caller a
        different threshold than the one it asked for, so a non-None value is rejected.

    Raises:
        ValueError: `threshold_k` was given (see above).
        ModelArtifactNotFoundError: `category` has no configured model, `model_name` is not
            the configured one, or the configured artifact is missing on disk. Never falls
            back to any other model.
        ModelIntegrityError: the configured artifact is not the validated model (MD5 mismatch).
        app.ai.preprocessing.errors.ImageProcessingError: image missing/unsupported/undecodable.
    """
    if threshold_k is not None:
        raise ValueError(
            "threshold_k is no longer supported: the decision threshold comes from the category's "
            "serving configuration (app.ai.inference.serving) and is never recomputed per call."
        )

    start = time.perf_counter()

    config = get_serving_config(category)
    if model_name != config.model_name:
        raise ModelArtifactNotFoundError(
            f"No model named '{model_name}' is configured for category '{category}' "
            f"(configured model: '{config.model_name}')."
        )
    model = load_serving_model(config)

    image_tensor = _image_to_tensor(Path(image_path), image_size)
    reconstruction_error = compute_reconstruction_error(model, image_tensor)
    threshold = config.threshold

    prediction = GOOD_PREDICTION if reconstruction_error <= threshold else DEFECTIVE_PREDICTION

    return PredictionResult(
        category=category,
        prediction=prediction,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        model_name=config.model_name,
        input_size=image_size,
        processing_time_ms=(time.perf_counter() - start) * 1000,
    )
