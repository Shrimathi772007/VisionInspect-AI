"""Category-specific AI serving configuration and cached model loading.

    category -> CategoryServingConfig (model artifact + validated threshold)
             -> cached, hash-verified model

Answers exactly two questions for inference, both in O(1) - no dataset access:
which trained model artifact serves a category, and which validated threshold
its predictions are compared against. Until this module existed, inference
always loaded ai_models/<category>/autoencoder.pt and recomputed a K=3 threshold
from every train/good image on every call; that served the original Phase 1
Bottle model rather than the validated Phase 3 one, and reprocessed the whole
training set per inspection.

Only categories with a genuinely validated model appear in SERVING_CONFIGS.
A category with no entry has no AI support - predict_image raises
ModelArtifactNotFoundError for it (which app.inspections.service already turns
into "ai_* stay NULL"). There is deliberately no default entry and no fallback to
another category's model, or from a configured artifact to a different one.

Two model families are served: the ConvAutoencoder (reconstruction error) and the
frozen-ResNet-18 PatchAnomalyDetector from the alternative model-family study (patch
nearest-neighbour score). The family, input size and artifact hashes are part of each
category's configuration, so a category can only be served by its own locked model.

This module never trains, evaluates, tunes or selects anything: thresholds are
copied in from an already-locked validation report, not derived here.
"""

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE
from app.ai.inference.errors import ModelArtifactNotFoundError, ModelIntegrityError
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.resnet18 import load_pretrained_resnet18
from app.ai.training.artifacts import get_model_path, load_model_for_category

# How a category's artifact is loaded and scored.
MODEL_FAMILY_CONVAE = "convae"  # ConvAutoencoder state_dict -> per-image reconstruction MSE
MODEL_FAMILY_PATCH_ANOMALY = "patch_anomaly"  # frozen ResNet-18 patch features -> PatchAnomalyDetector score


@dataclass(frozen=True)
class CategoryServingConfig:
    """Everything inference needs to serve one category, and where each value came from."""

    category: str
    # Artifact name relative to ai_models/<category>/, without ".pt" - the same convention
    # as app.ai.training.artifacts.get_model_path's `model_name` (e.g. "phase3_validation/autoencoder").
    artifact_name: str
    # Value reported as PredictionResult.model_name and stored in Inspection.ai_model_name.
    # "autoencoder" for ConvAE entries (the existing database contract); a patch-anomaly entry
    # uses its locked candidate id (e.g. "knn_l23_256"). Which artifact actually served a
    # prediction is identified by `artifact_name`, not by this label.
    model_name: str
    # Validated decision threshold: anomaly score (reconstruction error for ConvAE) <= threshold -> "good".
    threshold: float
    threshold_method: str
    threshold_parameter: float
    # MD5 of the expected artifact. When set, the artifact is verified on first load so a
    # replaced or retrained file is refused instead of silently served. None skips the check.
    expected_md5: str | None
    # Human-readable pointer to the authoritative record of the threshold and model hash.
    provenance: str
    # --- Fields below default to the original ConvAE behaviour, so existing entries are unchanged. ---
    model_family: str = MODEL_FAMILY_CONVAE
    # (width, height) the image is resized to - part of the validated model, not a caller choice
    # for patch-anomaly models (see app.ai.inference.predict.predict_image).
    input_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    # Patch-anomaly only: SHA-256 of the model state and of its model_config.json (layers, input size,
    # scorer, aggregation). Both are verified on first load, like expected_md5.
    expected_sha256: str | None = None
    expected_config_sha256: str | None = None
    # Identifier of the locked experiment the model and threshold come from, when there is one.
    experiment_id: str | None = None


SERVING_CONFIGS: dict[str, CategoryServingConfig] = {
    # Milestone 4 Phase 3 validated Bottle model. Every value below is copied verbatim from
    # backend/ai_models/bottle/evaluation_reports/phase3_validated_model_report.json:
    #   threshold   <- selection.selected_final_test_result.threshold (mean + 1.5*std of the
    #                  42 genuine validation errors; selected from validation only, BEFORE the
    #                  final test set was scored)
    #   expected_md5 <- phase3_model.md5
    # The Phase 1 model (ai_models/bottle/autoencoder.pt, MD5 2f470c30...) and its K=3 threshold
    # are intentionally NOT served any more. Phase 3 final-test metrics at this threshold:
    # accuracy 72.29%, precision 88.46%, recall 73.02%, F1 80.00%.
    "bottle": CategoryServingConfig(
        category="bottle",
        artifact_name="phase3_validation/autoencoder",
        model_name="autoencoder",
        threshold=0.0028031117030001018,
        threshold_method="mean_std",
        threshold_parameter=1.5,
        expected_md5="76478dd6996feb50fadaf5e5e5be1ae4",
        provenance=(
            "backend/ai_models/bottle/evaluation_reports/phase3_validated_model_report.json "
            "(selection.selected_final_test_result)"
        ),
    ),
    # Tile: the locked winner of the normal-only alternative model-family study (the Phase 1
    # ConvAE, ai_models/tile/phase1_baseline/, failed with AUROC 0.481 and is NOT served).
    # Every value is copied verbatim from backend/ai_models/tile/model_family_study/selection_lock.json:
    #   threshold        <- selected_threshold (policy mean_std_3: mean + 3*std of the leave-one-image-out
    #                       scores of all 230 train/good images; selected from normal data only)
    #   expected_md5/sha <- selected_artifact.md5 / .sha256
    # Model: frozen ImageNet ResNet-18, layers 2+3 (384-d, 3x3 avg pooling), 256x256 input, k=1
    # Euclidean nearest neighbour to a 10% per-image coreset bank, image score = mean of the top 1%
    # of patch scores. Final test (scored once): TN 33 FP 0 FN 7 TP 77, recall 91.67%, F1 95.65%,
    # FPR 0%, AUROC 0.9971 (EXCELLENT); gray_stroke recall is 62.5% (6 of the 7 misses).
    "tile": CategoryServingConfig(
        category="tile",
        artifact_name="model_family_study/selected_candidate/model_state",
        model_name="knn_l23_256",
        threshold=1.7393077017650718,
        threshold_method="mean_std",
        threshold_parameter=3.0,
        expected_md5="cb9530d76709319ec0eed58f55717356",
        provenance=(
            "backend/ai_models/tile/model_family_study/selection_lock.json (selected_threshold, "
            "selected_artifact; lock_digest d3302166d3b7088302286fdc46243f8316d39e848677ce045cd22597e51d472b) "
            "and reports/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCH_ANOMALY,
        input_size=(256, 256),
        expected_sha256="de00e2774daf02d97e1414fe75b10305e2ad392696449b1627a250b1ec0dd4d7",
        expected_config_sha256="6ab1c5e603cf2b63cb9c9f0e17320d83c2a5242c6df7c3cf386dedb63ed3812d",
        experiment_id="tile-model-family-tile-0ed823168064",
    ),
    # Cable: the locked winner of the normal-only alternative model-family study (the Phase 1
    # ConvAE, ai_models/cable/phase1_baseline/, failed with recall 0% / AUROC 0.574 and is NOT served).
    # Every value is copied verbatim from backend/ai_models/cable/model_family_study/selection_lock.json:
    #   threshold        <- selected_threshold (policy mean_std_3: mean + 3*std of the leave-one-image-out
    #                       scores of all 224 train/good images; selected from normal data only)
    #   expected_md5/sha <- selected_artifact.md5 / .sha256
    # Model: frozen ImageNet ResNet-18, layers 2+3 (384-d, 3x3 avg pooling), 128x128 input, k=1
    # Euclidean nearest neighbour to a 10% per-image coreset bank, image score = mean of the top 1%
    # of patch scores. Final test (scored once): TN 57 FP 1 FN 13 TP 79, recall 85.87%, F1 91.86%,
    # FPR 1.72%, AUROC 0.9880 (GOOD on point estimates; recall Wilson 95% lower bound 77.3%);
    # poke_insulation recall is 40% (6 of the 13 misses).
    "cable": CategoryServingConfig(
        category="cable",
        artifact_name="model_family_study/selected_candidate/model_state",
        model_name="knn_l23_128",
        threshold=2.244786389430004,
        threshold_method="mean_std",
        threshold_parameter=3.0,
        expected_md5="c878207964fc22865743f77c43ed06ea",
        provenance=(
            "backend/ai_models/cable/model_family_study/selection_lock.json (selected_threshold, "
            "selected_artifact; lock_digest 31d1aed63843aa0e316c3343b5a82309b2b349fde803d93219e22d012bcbc675) "
            "and reports/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCH_ANOMALY,
        input_size=(128, 128),
        expected_sha256="432859c5e39aaa85f8adbbd952f82ff0be1e6a905e360cfcca978a5638de7f5a",
        expected_config_sha256="f6696e7fbb3af5f275d8678c66339210ec7230195c212cad70dd22777bcf3230",
        experiment_id="cable-model-family-cable-ce2a8f015143",
    ),
}


def get_serving_config(category: str) -> CategoryServingConfig:
    """The serving configuration for `category`.

    Raises ModelArtifactNotFoundError if the category has no validated model configured.
    """
    config = SERVING_CONFIGS.get(category)
    if config is None:
        raise ModelArtifactNotFoundError(f"No AI model is configured for category '{category}'.")
    return config


def get_supported_categories() -> tuple[str, ...]:
    """Categories that currently have a configured (validated) AI model."""
    return tuple(SERVING_CONFIGS)


def _file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_md5(path: Path) -> str:
    return _file_digest(path, "md5")


def _missing(config: CategoryServingConfig) -> ModelArtifactNotFoundError:
    return ModelArtifactNotFoundError(
        f"No trained model artifact available for category '{config.category}' "
        f"(artifact '{config.artifact_name}')."
    )


def _verify(config: CategoryServingConfig, path: Path, algorithm: str, expected: str | None, what: str) -> None:
    if expected is None:
        return
    actual = _file_digest(path, algorithm)
    if actual != expected:
        raise ModelIntegrityError(
            f"{what} for category '{config.category}' (artifact '{config.artifact_name}') does not "
            f"match the validated model: expected {algorithm.upper()} {expected}, found {actual}."
        )


# One loaded model per resolved artifact path, invalidated if the file on disk changes.
# value: (file signature, model). Guarded by a lock because request handlers run in a
# thread pool; the models themselves are only ever used in eval()/no_grad mode.
_MODEL_CACHE: dict[str, tuple[tuple, nn.Module | PatchAnomalyDetector]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def load_serving_model(config: CategoryServingConfig) -> nn.Module | PatchAnomalyDetector:
    """The model for `config`, loaded from its configured artifact (cached, hash-verified).

    ConvAE configs return the ConvAutoencoder; patch-anomaly configs return a PatchAnomalyDetector
    built from the artifact's directory (model_state.pt + model_config.json) and the hash-verified
    frozen ImageNet ResNet-18. Loading is read-only: nothing is fitted and no file is written.

    Raises ModelArtifactNotFoundError if that exact artifact (or, for patch-anomaly models, its
    model_config.json) is missing - never falls back to any other file - and ModelIntegrityError
    if a configured MD5/SHA-256 differs.
    """
    path = get_model_path(config.category, config.artifact_name)
    patch = config.model_family == MODEL_FAMILY_PATCH_ANOMALY
    if not patch and config.model_family != MODEL_FAMILY_CONVAE:
        raise ModelArtifactNotFoundError(
            f"Unknown model family '{config.model_family}' configured for category '{config.category}'."
        )
    config_path = path.parent / "model_config.json"
    try:
        stat = path.stat()
        signature: tuple = (stat.st_mtime_ns, stat.st_size)
        if patch:
            config_stat = config_path.stat()
            signature += (config_stat.st_mtime_ns, config_stat.st_size)
    except OSError as exc:
        raise _missing(config) from exc
    key = str(path)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        _verify(config, path, "md5", config.expected_md5, "Model artifact")
        _verify(config, path, "sha256", config.expected_sha256, "Model artifact")
        if patch:
            _verify(config, config_path, "sha256", config.expected_config_sha256, "Model configuration")
            try:
                model = PatchAnomalyDetector.load(path.parent, load_pretrained_resnet18())
            except FileNotFoundError as exc:
                raise _missing(config) from exc
            if (model.image_size, model.image_size) != tuple(config.input_size):
                raise ModelIntegrityError(
                    f"Model configuration for category '{config.category}' expects {model.image_size}x"
                    f"{model.image_size} input, but the serving configuration says {config.input_size}."
                )
        else:
            try:
                model = load_model_for_category(config.category, model_name=config.artifact_name)
            except FileNotFoundError as exc:
                raise _missing(config) from exc

        _MODEL_CACHE[key] = (signature, model)
        return model
