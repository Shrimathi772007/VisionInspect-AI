"""Milestone 4 Phase 2: additive threshold-CANDIDATE generation and final-test scoring.

Generates and scores threshold CANDIDATES for the existing, unchanged
autoencoder. Does not change the production threshold rule
(app.ai.evaluation.threshold.compute_threshold, still mean + 3*std, still
used unmodified by app.ai.inference.predict.predict_image) and does not
retrain or otherwise touch the model.

Reuses, rather than reimplements:
- app.ai.evaluation.evaluate.compute_reconstruction_error for reconstruction error
- app.ai.evaluation.threshold.compute_threshold for the mean+K*std formula
- app.ai.evaluation.metrics.compute_classification_metrics for every metric

Two threshold methods, named explicitly (never inferred implicitly from
context) so a report/test can always say exactly which rule produced a value:

    METHOD_MEAN_STD    threshold = mean(calibration_errors) + K * std(calibration_errors)
    METHOD_PERCENTILE  threshold = percentile(calibration_errors, P)

Both operate ONLY on calibration-subset reconstruction errors (the same
167-of-209 train/good split established in Phase 1 -
app.ai.evaluation.calibration.split_for_calibration, fraction=0.8, seed=42).
Neither this module nor its candidates are ever given final-test images,
errors, or labels - see app.ai.evaluation.threshold_selection for how the
final-test set stays untouched until AFTER a candidate is chosen.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from app.ai.evaluation.evaluate import compute_reconstruction_error
from app.ai.evaluation.metrics import DEFECTIVE_LABEL, GOOD_LABEL, compute_classification_metrics
from app.ai.evaluation.threshold import compute_threshold
from app.ai.training.dataset import load_sample
from app.ai.training.schemas import DatasetSample

METHOD_MEAN_STD = "mean_std"
METHOD_PERCENTILE = "percentile"

DEFAULT_K_VALUES: tuple[float, ...] = (3.0, 2.5, 2.0, 1.5, 1.0)
DEFAULT_PERCENTILES: tuple[float, ...] = (95.0, 97.0, 98.0, 99.0)


def _sample_to_tensor(sample: DatasetSample, image_size: tuple[int, int]) -> torch.Tensor:
    result = load_sample(sample, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def compute_errors_for_samples(
    model: nn.Module, samples: Sequence[DatasetSample], image_size: tuple[int, int]
) -> list[float]:
    """Reconstruction error for each sample, via the existing, unmodified
    compute_reconstruction_error. Never looks at a label."""
    model.eval()
    return [compute_reconstruction_error(model, _sample_to_tensor(s, image_size)) for s in samples]


@dataclass(frozen=True)
class ThresholdCandidate:
    """One threshold candidate, computed ONLY from calibration-subset errors."""

    method: str  # METHOD_MEAN_STD | METHOD_PERCENTILE
    parameter: float  # K for mean_std, percentile (0-100) for percentile
    threshold: float
    calibration_sample_count: int
    calibration_mean: float
    calibration_std: float
    calibration_percentile: float | None  # set only for METHOD_PERCENTILE


def generate_candidates(
    calibration_errors: Sequence[float],
    k_values: Sequence[float] = DEFAULT_K_VALUES,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> list[ThresholdCandidate]:
    """All threshold candidates for one set of calibration errors.

    Deterministic: a pure function of `calibration_errors`, `k_values`, and
    `percentiles` - no randomness, no I/O, no test-set access of any kind.
    Raises ValueError on empty `calibration_errors`.
    """
    if not calibration_errors:
        raise ValueError("Cannot generate threshold candidates from empty calibration errors.")

    errors = np.asarray(list(calibration_errors), dtype=np.float64)
    mean = float(errors.mean())
    std = float(errors.std())
    count = len(errors)

    candidates: list[ThresholdCandidate] = []
    for k in k_values:
        candidates.append(
            ThresholdCandidate(
                method=METHOD_MEAN_STD,
                parameter=float(k),
                threshold=compute_threshold(calibration_errors, k=float(k)),
                calibration_sample_count=count,
                calibration_mean=mean,
                calibration_std=std,
                calibration_percentile=None,
            )
        )
    for p in percentiles:
        candidates.append(
            ThresholdCandidate(
                method=METHOD_PERCENTILE,
                parameter=float(p),
                threshold=float(np.percentile(errors, p)),
                calibration_sample_count=count,
                calibration_mean=mean,
                calibration_std=std,
                calibration_percentile=float(p),
            )
        )
    return candidates


@dataclass(frozen=True)
class CandidateTestResult:
    """One candidate's performance on the final, untouched test set.

    Computing this does not influence which candidate was selected - see
    app.ai.evaluation.threshold_selection.select_threshold, which must be
    called (and its result locked in) before this function is ever used.
    """

    candidate: ThresholdCandidate
    total_test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    false_positive_rate: float  # FP / (FP + TN) - fraction of actually-good images wrongly flagged
    false_negative_rate: float  # FN / (FN + TP) - fraction of actually-defective images missed


def evaluate_candidate_on_final_test(
    candidate: ThresholdCandidate, test_labels: Sequence[int], test_errors: Sequence[float]
) -> CandidateTestResult:
    """Score one candidate against already-computed final-test errors/labels.

    Uses compute_classification_metrics unmodified - identical metric
    definitions to every other evaluation in this project.
    """
    predictions = [DEFECTIVE_LABEL if error > candidate.threshold else GOOD_LABEL for error in test_errors]
    metrics = compute_classification_metrics(list(test_labels), predictions)

    negatives = metrics.false_positives + metrics.true_negatives
    positives = metrics.false_negatives + metrics.true_positives

    return CandidateTestResult(
        candidate=candidate,
        total_test_samples=len(test_labels),
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        true_negatives=metrics.true_negatives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        true_positives=metrics.true_positives,
        false_positive_rate=metrics.false_positives / negatives if negatives else 0.0,
        false_negative_rate=metrics.false_negatives / positives if positives else 0.0,
    )
