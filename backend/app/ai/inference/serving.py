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

This module never trains, evaluates, tunes or selects anything: thresholds are
copied in from an already-locked validation report, not derived here.
"""

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from app.ai.inference.errors import ModelArtifactNotFoundError, ModelIntegrityError
from app.ai.training.artifacts import get_model_path, load_model_for_category


@dataclass(frozen=True)
class CategoryServingConfig:
    """Everything inference needs to serve one category, and where each value came from."""

    category: str
    # Artifact name relative to ai_models/<category>/, without ".pt" - the same convention
    # as app.ai.training.artifacts.get_model_path's `model_name` (e.g. "phase3_validation/autoencoder").
    artifact_name: str
    # Value reported as PredictionResult.model_name and stored in Inspection.ai_model_name.
    # Kept as "autoencoder" (the existing database contract); which artifact actually served
    # a prediction is identified by `artifact_name`, not by this label.
    model_name: str
    # Validated decision threshold: reconstruction_error <= threshold -> "good".
    threshold: float
    threshold_method: str
    threshold_parameter: float
    # MD5 of the expected artifact. When set, the artifact is verified on first load so a
    # replaced or retrained file is refused instead of silently served. None skips the check.
    expected_md5: str | None
    # Human-readable pointer to the authoritative record of the threshold and model hash.
    provenance: str


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


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# One loaded model per resolved artifact path, invalidated if the file on disk changes.
# value: ((mtime_ns, size), model). Guarded by a lock because request handlers run in a
# thread pool; the models themselves are only ever used in eval()/no_grad mode.
_MODEL_CACHE: dict[str, tuple[tuple[int, int], nn.Module]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def load_serving_model(config: CategoryServingConfig) -> nn.Module:
    """The model for `config`, loaded from its configured artifact (cached, hash-verified).

    Raises ModelArtifactNotFoundError if that exact artifact is missing - never falls back
    to any other file - and ModelIntegrityError if its MD5 differs from `expected_md5`.
    """
    path = get_model_path(config.category, config.artifact_name)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelArtifactNotFoundError(
            f"No trained model artifact available for category '{config.category}' "
            f"(artifact '{config.artifact_name}')."
        ) from exc
    signature = (stat.st_mtime_ns, stat.st_size)
    key = str(path)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        if config.expected_md5 is not None:
            actual_md5 = _file_md5(path)
            if actual_md5 != config.expected_md5:
                raise ModelIntegrityError(
                    f"Model artifact for category '{config.category}' (artifact "
                    f"'{config.artifact_name}') does not match the validated model: "
                    f"expected MD5 {config.expected_md5}, found {actual_md5}."
                )

        try:
            model = load_model_for_category(config.category, model_name=config.artifact_name)
        except FileNotFoundError as exc:
            raise ModelArtifactNotFoundError(
                f"No trained model artifact available for category '{config.category}' "
                f"(artifact '{config.artifact_name}')."
            ) from exc

        _MODEL_CACHE[key] = (signature, model)
        return model
