"""Milestone 4 Phase 4: run one candidate (train-or-reuse + validate + select
threshold) WITHOUT ever touching final-test data.

Structurally guarantees no final-test access: this module never imports
discover_test_samples or anything from the final-test discovery path, and
CandidateResult carries no field derived from it. The final test set is
evaluated only later, by scripts/train_and_evaluate_bottle_phase4.py, after
a candidate has already been selected using nothing but the objects this
module returns.

Reuses Phase 1-3 code unmodified: app.ai.training.phase3_train for training,
app.ai.evaluation.threshold_experiments for reconstruction errors and
threshold candidates, app.ai.evaluation.phase3_threshold_selection for
validation-only threshold selection, app.ai.evaluation.model_metadata for
artifact identity/hashing.
"""

import statistics as stats_mod
import time
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from app.ai.evaluation.model_metadata import ModelMetadata, compute_model_metadata
from app.ai.evaluation.phase3_threshold_selection import Phase3SelectionResult, select_threshold_from_validation
from app.ai.evaluation.threshold_experiments import ThresholdCandidate, compute_errors_for_samples, generate_candidates
from app.ai.training.artifacts import load_model
from app.ai.training.model import build_model
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.phase4_candidates import CandidateConfig
from app.ai.training.schemas import DatasetSample


@dataclass
class CandidateResult:
    config: CandidateConfig
    model_path: Path
    model: nn.Module
    model_metadata: ModelMetadata
    training_duration_s: float
    final_training_loss: float
    validation_errors: list[float]
    validation_mean: float
    validation_std: float
    validation_min: float
    validation_max: float
    threshold_candidates: list[ThresholdCandidate]
    selection: Phase3SelectionResult
    validation_inference_ms: float


def run_candidate(
    config: CandidateConfig,
    training_samples: list[DatasetSample],
    validation_samples: list[DatasetSample],
    model_path: Path,
    reused_model_path: Path | None = None,
    reused_training_duration_s: float | None = None,
    reused_final_loss: float | None = None,
) -> CandidateResult:
    """Train `config` (or reuse an existing artifact if `config.reuse_phase3_artifact`
    is set) and evaluate it on VALIDATION ONLY - no final-test access is possible here.
    """
    if config.reuse_phase3_artifact:
        if reused_model_path is None or not reused_model_path.is_file():
            raise ValueError(
                f"{config.candidate_id} is marked reuse_phase3_artifact but no valid "
                "reused_model_path was given."
            )
        model = load_model(
            reused_model_path, build_model(latent_channels=config.training_config.model.latent_channels)
        )
        training_duration_s = reused_training_duration_s if reused_training_duration_s is not None else 0.0
        final_training_loss = reused_final_loss if reused_final_loss is not None else float("nan")
        actual_model_path = reused_model_path
    else:
        result, model = train_anomaly_model_on_samples(training_samples, config.training_config, model_path)
        training_duration_s = result.duration_seconds
        final_training_loss = result.final_loss
        actual_model_path = result.model_path

    image_size = config.training_config.image_size

    val_start = time.perf_counter()
    validation_errors = compute_errors_for_samples(model, validation_samples, image_size)
    validation_inference_ms = (time.perf_counter() - val_start) * 1000

    validation_mean = stats_mod.fmean(validation_errors)
    validation_std = (
        (sum((e - validation_mean) ** 2 for e in validation_errors) / len(validation_errors)) ** 0.5
    )

    threshold_candidates = generate_candidates(validation_errors)
    selection = select_threshold_from_validation(threshold_candidates, validation_errors)

    model_metadata = compute_model_metadata(
        actual_model_path,
        category=config.training_config.category,
        model_name=config.candidate_id,
        latent_channels=config.training_config.model.latent_channels,
    )

    return CandidateResult(
        config=config,
        model_path=actual_model_path,
        model=model,
        model_metadata=model_metadata,
        training_duration_s=training_duration_s,
        final_training_loss=final_training_loss,
        validation_errors=validation_errors,
        validation_mean=validation_mean,
        validation_std=validation_std,
        validation_min=min(validation_errors),
        validation_max=max(validation_errors),
        threshold_candidates=threshold_candidates,
        selection=selection,
        validation_inference_ms=validation_inference_ms,
    )
