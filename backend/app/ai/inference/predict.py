"""Single-image AI inference using a category's configured, validated model and threshold.

    inspection image -> the category's preprocessing (existing pipeline at the configured input
    size; WRN-50 PatchCore: its locked crop224 / full256 / full320 mode) -> category's configured
    trained model (cached) -> anomaly score (reconstruction error for ConvAE, patch nearest-neighbour
    score for patch models) -> category's configured locked threshold -> good/defective

Which model and which threshold serve a category is decided entirely by
app.ai.inference.serving (SERVING_CONFIGS) - a constant-time lookup. Inference never
derives a threshold from data: it does not touch train/good (or any other dataset) images,
so its cost no longer grows with the size of a category's training set.

Reuses, rather than duplicates:
- app.ai.preprocessing.process_image for image loading/validation/preprocessing
  (same decode -> RGB -> resize -> normalize pipeline used everywhere else).
- app.ai.preprocessing.patchcore_preprocess (and, for Grid, the Addendum 1 resize_full320) for
  WRN-50 PatchCore models - exactly the preprocessing their final test was scored with.
- app.ai.evaluation.evaluate.compute_reconstruction_error for the reconstruction-error
  definition - identical to the one used by every evaluation.

Does not train, retrain, or otherwise modify any model or its architecture, and never
uses MVTec ground-truth labels to influence a prediction. See app.ai.evaluation for the
separate checkpoint that measures AI-prediction-vs-ground-truth agreement.
"""

import time
from pathlib import Path

import torch

from app.ai.evaluation.evaluate import compute_reconstruction_error
from app.ai.inference.errors import ModelArtifactNotFoundError
from app.ai.inference.schemas import DEFECTIVE_PREDICTION, GOOD_PREDICTION, PredictionResult
from app.ai.inference.serving import (
    INPUT_MODE_FULL320,
    MODEL_FAMILY_CONVAE,
    MODEL_FAMILY_PATCHCORE_WRN50,
    full320_module,
    get_serving_config,
    get_supported_categories,
    load_serving_model,
)
from app.ai.models.patch_anomaly import aggregate_scores, extract_patch_features
from app.ai.models.patchcore import FEATURE_STRIDE
from app.ai.preprocessing import process_image
from app.ai.preprocessing.patchcore_preprocess import decode_image, normalize, resize_and_crop

# PredictionResult.input_mode of the ResNet-18 patch models: the whole image resized to the input size.
INPUT_MODE_RESIZE = "resize"

# Categories with a configured, validated model today (a snapshot of SERVING_CONFIGS at
# import time; get_supported_categories() is the live view). predict_image itself is
# generic: a category is served if and only if it has a serving configuration.
SUPPORTED_CATEGORIES = get_supported_categories()


def _image_to_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Model-ready CHW tensor for one image, via the existing preprocessing pipeline."""
    result = process_image(path, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def _patchcore_tensor(path: Path, mode: str) -> torch.Tensor:
    """ImageNet-normalized CHW tensor in a WRN-50 model's locked input mode - the same decode -> resize/crop ->
    normalize composition the WRN-50 final test was scored with (scripts/experiments/patchcore_wrn50/
    run_wrn50_final.py)."""
    image = decode_image(path)
    if mode == INPUT_MODE_FULL320:
        return normalize(full320_module().resize_full320(image))
    return normalize(resize_and_crop(image, mode))


def predict_image(
    image_path: Path,
    category: str,
    model_name: str | None = None,
    image_size: tuple[int, int] | None = None,
    threshold_k: float | None = None,
    return_patch_scores: bool = False,
) -> PredictionResult:
    """Run AI anomaly-detection inference for one inspection image.

    Looks up `category`'s serving configuration, scores the image with its configured model
    (ConvAE: reconstruction error; patch-anomaly: frozen-ResNet-18 patch nearest-neighbour
    score; WRN-50 PatchCore: frozen-WRN-50-2 patch nearest-neighbour score, aggregated as locked),
    and compares the score against its configured locked threshold
    (score <= threshold -> "good", otherwise "defective"). The score is reported in
    PredictionResult.reconstruction_error (the existing field/column name). Takes no
    ground-truth label and produces none - this is an AI prediction, not an evaluation
    against MVTec ground truth.

    `model_name` and `threshold_k` exist only so pre-serving-registry callers keep working
    (same names and positions as before). Neither can select a different model or threshold -
    the serving configuration is authoritative:
      - `model_name`, if given, must equal the category's configured model name
        ("autoencoder" for ConvAE categories). Any other value is refused, never used to
        pick an artifact. None (the default) means the configured model.
      - `threshold_k` must be None. The old K-sigma threshold (recomputed from all train/good
        images on every call) no longer exists; silently ignoring a K would hand the caller a
        different threshold than the one it asked for, so a non-None value is rejected.

    `image_size` defaults to the category's configured input size. A patch-anomaly or WRN-50
    model is only valid at its locked input size, so any other value is refused for it.

    `return_patch_scores=True` additionally fills PredictionResult.patch_scores (the (h, w) grid of
    per-patch anomaly scores the image score is aggregated from) and .input_mode, for anomaly-map
    localization (app.ai.inference.localization). The score, threshold and prediction are the same
    with or without it: for WRN-50 the grid is the tensor already computed; for ResNet-18 the
    detector's score_images is unrolled into its own three steps (features -> scorer ->
    aggregate_scores), which is exactly what score_images does. ConvAE models have no patch grid.

    Raises:
        ValueError: `threshold_k` was given, or a patch / WRN-50 category got a different
            `image_size` (see above).
        ModelArtifactNotFoundError: `category` has no configured model, `model_name` is not
            the configured one, or the configured artifact is missing on disk. Never falls
            back to any other model.
        ModelIntegrityError: the configured artifact is not the locked model (hash, backbone,
            input mode or aggregation mismatch).
        app.ai.preprocessing.errors.ImageProcessingError: image missing/unsupported/undecodable.
    """
    if threshold_k is not None:
        raise ValueError(
            "threshold_k is no longer supported: the decision threshold comes from the category's "
            "serving configuration (app.ai.inference.serving) and is never recomputed per call."
        )

    start = time.perf_counter()

    config = get_serving_config(category)
    if model_name is not None and model_name != config.model_name:
        raise ModelArtifactNotFoundError(
            f"No model named '{model_name}' is configured for category '{category}' "
            f"(configured model: '{config.model_name}')."
        )
    locked_size = config.model_family != MODEL_FAMILY_CONVAE
    if image_size is None:
        image_size = config.input_size
    elif locked_size and tuple(image_size) != tuple(config.input_size):
        raise ValueError(
            f"Category '{category}' is served at its validated input size {config.input_size}; "
            f"image_size {tuple(image_size)} is not allowed."
        )
    model = load_serving_model(config)
    patch_grid = None
    input_mode = None

    if config.model_family == MODEL_FAMILY_PATCHCORE_WRN50:
        # ImageNet-normalized at the locked mode; image score = the locked aggregation of the k=1 patch distances.
        image_tensor = _patchcore_tensor(Path(image_path), config.input_mode)
        with torch.no_grad():
            patch_scores = model.patch_scores_from_features(model.extract_features(image_tensor.unsqueeze(0)))
            reconstruction_error = float(model.image_scores(patch_scores, config.aggregation)[0])
        if return_patch_scores:
            side = config.input_size[0] // FEATURE_STRIDE
            patch_grid, input_mode = patch_scores[0].reshape(side, side), config.input_mode
    elif locked_size and return_patch_scores:
        # PatchAnomalyDetector.score_images, unrolled so the patch grid is kept - same calls, same order.
        image_tensor = _image_to_tensor(Path(image_path), image_size)
        with torch.no_grad():
            grid = model.scorer.patch_scores(extract_patch_features(model.extractor, image_tensor.unsqueeze(0), model.layers))
            reconstruction_error = [float(s) for s in aggregate_scores(grid, model.aggregation)][0]
        patch_grid, input_mode = grid[0], INPUT_MODE_RESIZE
    elif locked_size:
        # [0,1] RGB at the locked size; ImageNet normalization happens inside the detector.
        image_tensor = _image_to_tensor(Path(image_path), image_size)
        reconstruction_error = model.score_images(image_tensor.unsqueeze(0))[0]
    else:
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
        patch_scores=patch_grid,
        input_mode=input_mode,
    )
