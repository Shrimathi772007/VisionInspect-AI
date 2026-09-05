"""Single-image AI inference using an existing trained autoencoder + the Phase 5 threshold rule.

    inspection image -> existing preprocessing -> existing trained model
    -> reconstruction error -> existing Phase 5 threshold -> good/defective

Reuses, rather than duplicates:
- app.ai.preprocessing.process_image for image loading/validation/preprocessing
  (same decode -> RGB -> resize -> normalize pipeline used everywhere else).
- app.ai.training.artifacts.load_model_for_category for loading the trained model.
- app.ai.evaluation.evaluate.compute_reconstruction_error for the reconstruction-error
  definition - identical formula to the one used in Phase 5 evaluation.
- app.ai.evaluation.threshold.compute_threshold and app.ai.training.dataset for deriving
  the same train/good-only threshold used in Phase 5 - never test images or labels.

Does not train, retrain, or otherwise modify any model or its architecture, and never
uses MVTec ground-truth labels to influence a prediction. See app.ai.evaluation for the
separate checkpoint that measures AI-prediction-vs-ground-truth agreement.
"""

import time
from pathlib import Path

import torch

from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, compute_reconstruction_error
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold
from app.ai.inference.errors import ModelArtifactNotFoundError
from app.ai.inference.schemas import DEFECTIVE_PREDICTION, GOOD_PREDICTION, PredictionResult
from app.ai.preprocessing import process_image
from app.ai.training.artifacts import load_model_for_category
from app.ai.training.dataset import discover_train_samples

# Category with a trained artifact today. predict_image itself is generic - it works for
# any category that has both a saved model under ai_models/<category>/ and a matching
# dataset/<category>/train/good directory to derive the threshold from.
SUPPORTED_CATEGORIES = ("bottle",)


def _image_to_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Model-ready CHW tensor for one image, via the existing preprocessing pipeline."""
    result = process_image(path, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def _get_threshold(
    category: str,
    model: torch.nn.Module,
    image_size: tuple[int, int],
    threshold_k: float,
) -> float:
    """The Phase 5 threshold rule (mean + k*std of train/good reconstruction error),
    recomputed from `model`'s own train/good reconstruction error - never from test data."""
    train_samples = discover_train_samples(category)
    train_errors = [compute_reconstruction_error(model, _image_to_tensor(s.path, image_size)) for s in train_samples]
    return compute_threshold(train_errors, k=threshold_k)


def predict_image(
    image_path: Path,
    category: str,
    model_name: str = "autoencoder",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    threshold_k: float = DEFAULT_THRESHOLD_K,
) -> PredictionResult:
    """Run AI anomaly-detection inference for one inspection image.

    Loads the existing trained autoencoder for `category`, reconstructs the image, and
    compares its reconstruction error against the existing Phase 5 threshold rule to
    decide "good" vs "defective". Takes no ground-truth label and produces none - this is
    an AI prediction, not an evaluation against MVTec ground truth.

    Raises:
        ModelArtifactNotFoundError: no trained model for `category`/`model_name`.
        app.ai.preprocessing.errors.ImageProcessingError: image missing/unsupported/undecodable.
        app.ai.training.errors.TrainingDataError: dataset misconfigured (used only to derive
            the train/good threshold - never to look at the input image's label).
    """
    start = time.perf_counter()

    try:
        model = load_model_for_category(category, model_name=model_name)
    except FileNotFoundError as exc:
        raise ModelArtifactNotFoundError(
            f"No trained model artifact available for category '{category}' (model '{model_name}')."
        ) from exc

    image_tensor = _image_to_tensor(Path(image_path), image_size)
    reconstruction_error = compute_reconstruction_error(model, image_tensor)
    threshold = _get_threshold(category, model, image_size, threshold_k)

    prediction = GOOD_PREDICTION if reconstruction_error <= threshold else DEFECTIVE_PREDICTION

    return PredictionResult(
        category=category,
        prediction=prediction,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        model_name=model_name,
        input_size=image_size,
        processing_time_ms=(time.perf_counter() - start) * 1000,
    )
