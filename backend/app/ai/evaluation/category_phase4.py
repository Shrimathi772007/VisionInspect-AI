"""Milestone 4 Phase 4 optimization experiment for one category - VALIDATION ONLY.

Trains/evaluates the staged candidates (app.ai.training.category_phase4_candidates) on the
Phase 3 train/validation split, picks a winner per stage, and freezes ONE candidate + ONE
threshold as an immutable `LockedSelection`. It never touches the final test set:

  * This module does not import `discover_test_samples` (or anything that lists test images),
    and the only things it consumes are the 313 training samples and the 78 validation samples
    handed to `run_optimization_experiment`. A test asserts this by parsing the source.
  * The final test is scored elsewhere (app.ai.evaluation.category_phase4_final), and only from
    a `LockedSelection` - the frozen result of this module.

Selection rules (declared here, before any final-test result exists, and never changed after):

  Candidate eligibility  a candidate whose every threshold option exceeds the inherited 10%
                         validation false-positive ceiling is disqualified
                         (app.ai.evaluation.phase3_threshold_selection).
  Ranking                ascending validation coefficient of variation, CoV = std / mean of the
                         candidate's reconstruction errors on the (good-only) validation images.
  Ties                   CoVs within TIE_TOLERANCE (2%, relative) of the best are tied; the tie is
                         broken by, in order: fewer epochs, smaller input size, smaller model
                         file, learning rate closer to the baseline's, candidate id. Inference
                         speed is recorded and reported but wall-clock time is deliberately NOT a
                         tie-breaker, so a selection can never flip because of timing noise
                         (speed is already implied by input size, which is a tie-breaker).
  Threshold              for the winner, the existing Phase 3 rule on its own validation errors:
                         lowest threshold candidate with validation FP rate <= 10%.

LIMITATION, stated plainly: the validation set holds only good images, so it cannot measure recall
or F1. CoV is an engineering signal for "how consistently does this model reconstruct normal
images", not a proven predictor of defect-detection performance, and lower CoV does not
mathematically guarantee better anomaly detection. Training loss is never used to choose.
"""

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from app.ai.evaluation.evaluate import compute_reconstruction_error
from app.ai.evaluation.model_metadata import ModelMetadata, compute_model_metadata
from app.ai.evaluation.phase3_threshold_selection import (
    STATUS_NO_DEFENSIBLE_WINNER,
    Phase3SelectionResult,
    select_threshold_from_validation,
)
from app.ai.evaluation.phase4_selection import TIE_TOLERANCE
from app.ai.evaluation.threshold_experiments import ThresholdCandidate, generate_candidates
from app.ai.training.artifacts import load_model
from app.ai.training.category_phase4_candidates import (
    BASELINE_LEARNING_RATE,
    stage1_candidates,
    stage2_candidates,
    stage3_candidates,
)
from app.ai.training.dataset import load_sample
from app.ai.training.model import build_model
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.phase4_candidates import CandidateConfig
from app.ai.training.schemas import DatasetSample
from app.ai.training.validation_split import TrainValidationSplit

EVENT_CANDIDATE_READY = "candidate_ready:"  # + candidate_id
EVENT_STAGE_SELECTED = "stage_selected:"  # + stage name
EVENT_SELECTION_FROZEN = "selection_frozen"


@dataclass
class Phase3Reference:
    """The protected Phase 3 baseline artifact, reusable ONLY as the exact baseline candidate."""

    model_path: Path
    training_duration_s: float
    final_loss: float
    num_training_images: int
    loss_history: list[float] = field(default_factory=list)


@dataclass
class CandidateOutcome:
    config: CandidateConfig
    model_path: Path
    model_metadata: ModelMetadata
    reused_phase3_artifact: bool
    training_duration_s: float
    final_training_loss: float
    loss_history: list[float]
    num_training_images: int
    model_load_ms: float
    validation_errors: list[float]
    validation_mean: float
    validation_std: float
    validation_min: float
    validation_max: float
    coefficient_of_variation: float
    threshold_candidates: list[ThresholdCandidate]
    selection: Phase3SelectionResult
    validation_evaluation_ms: float
    validation_preprocess_ms_per_image: float
    validation_inference_ms_per_image: float
    model: nn.Module = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def candidate_id(self) -> str:
        return self.config.candidate_id


def evaluate_on_samples(
    model: nn.Module, samples: list[DatasetSample], image_size: tuple[int, int]
) -> tuple[list[float], float, float, float]:
    """(errors, total_ms, mean preprocess ms/image, mean forward-pass ms/image).

    Same operations as threshold_experiments.compute_errors_for_samples - load_sample ->
    CHW tensor -> compute_reconstruction_error - with the two costs timed separately. Never
    looks at a label.
    """
    model.eval()
    errors, prep_ms, inf_ms = [], [], []
    start = time.perf_counter()
    for sample in samples:
        t0 = time.perf_counter()
        result = load_sample(sample, target_size=image_size)
        tensor = torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()
        t1 = time.perf_counter()
        errors.append(compute_reconstruction_error(model, tensor))
        t2 = time.perf_counter()
        prep_ms.append((t1 - t0) * 1000)
        inf_ms.append((t2 - t1) * 1000)
    total_ms = (time.perf_counter() - start) * 1000
    return errors, total_ms, statistics.fmean(prep_ms), statistics.fmean(inf_ms)


def run_candidate(
    config: CandidateConfig,
    split: TrainValidationSplit,
    model_path: Path,
    phase3_reference: Phase3Reference | None = None,
) -> CandidateOutcome:
    """Train (or, for the exact baseline configuration only, load) one candidate and evaluate it
    on VALIDATION images only. There is no parameter through which test data could enter."""
    training_config = config.training_config
    if config.reuse_phase3_artifact:
        if phase3_reference is None or not phase3_reference.model_path.is_file():
            raise ValueError(f"{config.candidate_id} reuses the Phase 3 artifact but none was provided.")
        model = load_model(phase3_reference.model_path, build_model(training_config.model.latent_channels))
        actual_path = phase3_reference.model_path
        duration, final_loss = phase3_reference.training_duration_s, phase3_reference.final_loss
        loss_history, trained_on = phase3_reference.loss_history, phase3_reference.num_training_images
    else:
        result, model = train_anomaly_model_on_samples(split.training, training_config, model_path)
        actual_path = result.model_path
        duration, final_loss = result.duration_seconds, result.final_loss
        loss_history, trained_on = result.loss_history, result.num_training_images

    # Cold load of the saved artifact, for the model-loading timing.
    t0 = time.perf_counter()
    load_model(actual_path, build_model(training_config.model.latent_channels))
    model_load_ms = (time.perf_counter() - t0) * 1000

    errors, total_ms, prep_ms, inf_ms = evaluate_on_samples(model, split.validation, training_config.image_size)
    mean = statistics.fmean(errors)
    std = (sum((e - mean) ** 2 for e in errors) / len(errors)) ** 0.5

    candidates = generate_candidates(errors)
    selection = select_threshold_from_validation(candidates, errors)  # validation errors only

    return CandidateOutcome(
        config=config,
        model_path=actual_path,
        model_metadata=compute_model_metadata(
            actual_path,
            category=training_config.category,
            model_name=config.candidate_id,
            latent_channels=training_config.model.latent_channels,
            device=training_config.device,
        ),
        reused_phase3_artifact=config.reuse_phase3_artifact,
        training_duration_s=duration,
        final_training_loss=final_loss,
        loss_history=list(loss_history),
        num_training_images=trained_on,
        model_load_ms=model_load_ms,
        validation_errors=errors,
        validation_mean=mean,
        validation_std=std,
        validation_min=min(errors),
        validation_max=max(errors),
        coefficient_of_variation=std / mean if mean else float("inf"),
        threshold_candidates=candidates,
        selection=selection,
        validation_evaluation_ms=total_ms,
        validation_preprocess_ms_per_image=prep_ms,
        validation_inference_ms_per_image=inf_ms,
        model=model,
    )


# ---------------------------------------------------------------------------
# Stage selection (validation-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageSelection:
    stage: str
    winner_id: str
    reasoning: str
    ranked: list[tuple[str, float]]  # (candidate_id, CoV), best first, eligible candidates only
    disqualified: list[str]
    tied: list[str]


def _tie_break_key(outcome: CandidateOutcome) -> tuple:
    tc = outcome.config.training_config
    return (
        tc.epochs,
        tc.image_size[0] * tc.image_size[1],
        outcome.model_metadata.artifact_size_bytes,
        abs(tc.learning_rate - BASELINE_LEARNING_RATE),
        outcome.candidate_id,
    )


def select_stage_winner(stage: str, outcomes: list[CandidateOutcome]) -> tuple[CandidateOutcome, StageSelection]:
    """Winner among `outcomes`, using validation-derived fields ONLY (CoV, threshold status)."""
    if not outcomes:
        raise ValueError("select_stage_winner requires at least one candidate.")
    eligible = [o for o in outcomes if o.selection.status != STATUS_NO_DEFENSIBLE_WINNER]
    disqualified = [o.candidate_id for o in outcomes if o.selection.status == STATUS_NO_DEFENSIBLE_WINNER]
    if not eligible:
        raise ValueError(f"No candidate in {stage} produced a defensible validation threshold.")

    ranked = sorted(eligible, key=lambda o: (o.coefficient_of_variation, o.candidate_id))
    best_cov = ranked[0].coefficient_of_variation
    tied = [o for o in ranked if abs(o.coefficient_of_variation - best_cov) <= TIE_TOLERANCE * best_cov]

    if len(tied) > 1:
        winner = min(tied, key=_tie_break_key)
        reasoning = (
            f"{len(tied)} candidates tied within {TIE_TOLERANCE:.0%} of the best CoV ({best_cov:.4f}): "
            f"{[o.candidate_id for o in tied]}. Tie broken by fewer epochs, then smaller input size, then "
            f"smaller model file, then learning rate nearest the baseline -> '{winner.candidate_id}'."
        )
    else:
        winner = ranked[0]
        reasoning = (
            f"'{winner.candidate_id}' had the lowest validation coefficient of variation "
            f"({best_cov:.4f}) among eligible candidates. This is an engineering consistency signal on "
            "good-only validation images, not a recall/F1 measurement."
        )
    return winner, StageSelection(
        stage=stage,
        winner_id=winner.candidate_id,
        reasoning=reasoning,
        ranked=[(o.candidate_id, o.coefficient_of_variation) for o in ranked],
        disqualified=disqualified,
        tied=[o.candidate_id for o in tied] if len(tied) > 1 else [],
    )


# ---------------------------------------------------------------------------
# The frozen result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LockedSelection:
    """The immutable outcome of the validation-only experiment: one candidate, one threshold.

    The final-test scorer accepts nothing else. `digest` fingerprints every candidate's
    validation results at the moment of freezing, so the report can prove what was locked.
    """

    candidate_id: str
    image_size: tuple[int, int]
    epochs: int
    learning_rate: float
    model_path: Path
    model_md5: str
    threshold_method: str
    threshold_parameter: float
    threshold: float
    is_baseline_configuration: bool
    digest: str


@dataclass
class ExperimentResult:
    outcomes: dict[str, CandidateOutcome]  # by candidate_id, in the order they became ready
    stage_selections: list[StageSelection]
    winner: CandidateOutcome
    locked: LockedSelection
    events: list[str]
    total_ms: float


def _digest(outcomes: dict[str, CandidateOutcome], stages: list[StageSelection]) -> str:
    payload = {
        "candidates": {
            cid: {
                "model_md5": o.model_metadata.md5,
                "validation_mean": o.validation_mean,
                "validation_std": o.validation_std,
                "selection_status": o.selection.status,
                "selected_threshold": None if o.selection.selected is None else o.selection.selected.threshold,
            }
            for cid, o in outcomes.items()
        },
        "winners": [(s.stage, s.winner_id) for s in stages],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_optimization_experiment(
    category: str,
    split: TrainValidationSplit,
    candidates_root: Path,
    phase3_reference: Phase3Reference,
    log=print,
) -> ExperimentResult:
    """Stages 1-3 on validation data only, ending in a frozen selection.

    Candidate artifacts are written to `candidates_root/<candidate_id>/autoencoder.pt` (never
    over the Phase 3 artifact). The final test set is not reachable from here.
    """
    started = time.perf_counter()
    outcomes: dict[str, CandidateOutcome] = {}
    events: list[str] = []
    stages: list[StageSelection] = []

    def get(config: CandidateConfig) -> CandidateOutcome:
        if config.candidate_id not in outcomes:  # one configuration = one candidate, trained once
            path = candidates_root / config.candidate_id / "autoencoder.pt"
            log(f"  [{config.stage}] {config.candidate_id}: "
                f"{'loading protected Phase 3 baseline artifact' if config.reuse_phase3_artifact else 'training'} ...")
            outcome = run_candidate(config, split, path, phase3_reference)
            outcomes[config.candidate_id] = outcome
            events.append(EVENT_CANDIDATE_READY + config.candidate_id)
            log(f"      done: train {outcome.training_duration_s:.1f}s, val CoV {outcome.coefficient_of_variation:.4f}, "
                f"selection {outcome.selection.status}")
        return outcomes[config.candidate_id]

    def stage(name: str, configs: list[CandidateConfig]) -> CandidateOutcome:
        pool = [get(c) for c in configs]
        winner, selection = select_stage_winner(name, pool)
        stages.append(selection)
        events.append(EVENT_STAGE_SELECTED + name)
        log(f"  => {name} winner: {winner.candidate_id}. {selection.reasoning}")
        return winner

    log("=== STAGE 1: input size ===")
    w1 = stage("stage1_input_size", stage1_candidates(category))
    size = w1.config.training_config.image_size

    log(f"=== STAGE 2: epochs at input size {size} ===")
    w2 = stage("stage2_epochs", stage2_candidates(category, size))
    epochs = w2.config.training_config.epochs

    log(f"=== STAGE 3: learning rate at input size {size}, epochs {epochs} ===")
    w3 = stage("stage3_learning_rate", stage3_candidates(category, size, epochs))

    selected = w3.selection.selected
    assert selected is not None  # eligibility guarantees a validation-selected threshold
    tc = w3.config.training_config
    events.append(EVENT_SELECTION_FROZEN)
    locked = LockedSelection(
        candidate_id=w3.candidate_id,
        image_size=tuple(tc.image_size),
        epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        model_path=w3.model_path,
        model_md5=w3.model_metadata.md5,
        threshold_method=selected.method,
        threshold_parameter=selected.parameter,
        threshold=selected.threshold,
        is_baseline_configuration=w3.config.reuse_phase3_artifact,
        digest=_digest(outcomes, stages),
    )
    return ExperimentResult(
        outcomes=outcomes,
        stage_selections=stages,
        winner=w3,
        locked=locked,
        events=events,
        total_ms=(time.perf_counter() - started) * 1000,
    )


# ---------------------------------------------------------------------------
# Reproducibility comparison of two complete experiments
# ---------------------------------------------------------------------------

def compare_experiments(a: ExperimentResult, b: ExperimentResult) -> dict:
    """Field-by-field comparison of two complete, independent experiment runs."""

    def per(fn):
        return {cid: fn(o) for cid, o in a.outcomes.items()} == {cid: fn(o) for cid, o in b.outcomes.items()}

    result = {
        "candidate_configurations_identical": [o.config.training_config for o in a.outcomes.values()]
        == [o.config.training_config for o in b.outcomes.values()]
        and list(a.outcomes) == list(b.outcomes),
        "model_hashes_identical": per(lambda o: o.model_metadata.md5),
        "training_losses_identical": per(lambda o: (o.final_training_loss, tuple(o.loss_history))),
        "validation_errors_identical": per(lambda o: tuple(o.validation_errors)),
        "threshold_candidates_identical": per(lambda o: tuple(c.threshold for c in o.threshold_candidates)),
        "per_candidate_selected_thresholds_identical": per(
            lambda o: None if o.selection.selected is None else o.selection.selected.threshold
        ),
        "stage_winners_identical": [(s.stage, s.winner_id) for s in a.stage_selections]
        == [(s.stage, s.winner_id) for s in b.stage_selections],
        "selected_candidate_identical": a.locked.candidate_id == b.locked.candidate_id,
        "locked_threshold_identical": a.locked.threshold == b.locked.threshold,
        "locked_digest_identical": a.locked.digest == b.locked.digest,
    }
    result["all_identical"] = all(result.values())
    return result


__all__ = [
    "CandidateOutcome",
    "ExperimentResult",
    "LockedSelection",
    "Phase3Reference",
    "StageSelection",
    "compare_experiments",
    "evaluate_on_samples",
    "run_candidate",
    "run_optimization_experiment",
    "select_stage_winner",
]
