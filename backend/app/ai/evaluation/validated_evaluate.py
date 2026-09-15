"""Milestone 4 Phase 1: leakage-free, timed, reproducible evaluation.

    trained model (unchanged) -> train/good errors -> deterministic
    calibration/held-out split -> threshold from calibration subset only
    -> untouched final test set -> real metrics + timing

Differs from app.ai.evaluation.evaluate.evaluate_model (which this module
does not modify or replace - that function still reproduces the original,
documented baseline methodology exactly) only in how the threshold is
calibrated: here it comes from a deterministic subset of train/good errors
(see app.ai.evaluation.calibration) rather than all of them. The final test
set - the same test/good + test/<defect_type>/* images evaluate_model uses -
is never touched by calibration and is scored exactly once.

Does not train, retrain, or modify the model. Ground-truth labels are used
only to score predictions after the fact.
"""

import statistics
import time
from collections.abc import Sequence

import torch
from torch import nn

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_CALIBRATION_SEED,
    split_for_calibration,
)
from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE, compute_reconstruction_error
from app.ai.evaluation.metrics import DEFECTIVE_LABEL, GOOD_LABEL, compute_classification_metrics
from app.ai.evaluation.schemas import CalibrationInfo, EvaluationSample, TimingSummary, ValidatedEvaluationResult
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold
from app.ai.training.dataset import discover_test_samples, discover_train_samples, load_sample
from app.ai.training.schemas import DatasetSample


def _sample_to_tensor(sample: DatasetSample, image_size: tuple[int, int]) -> torch.Tensor:
    result = load_sample(sample, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def _errors_for(model: nn.Module, samples: Sequence[DatasetSample], image_size: tuple[int, int]) -> list[float]:
    return [compute_reconstruction_error(model, _sample_to_tensor(s, image_size)) for s in samples]


def evaluate_model_validated(
    category: str,
    model: nn.Module,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    calibration_seed: int = DEFAULT_CALIBRATION_SEED,
    model_load_ms: float = 0.0,
) -> ValidatedEvaluationResult:
    """Evaluate `model` on `category` using a calibration-split threshold.

    Raises ValueError if there are too few train/good samples to split, or
    no test images exist for `category` (same failure mode as evaluate_model).
    """
    model.eval()

    train_samples = discover_train_samples(category)
    split = split_for_calibration(train_samples, fraction=calibration_fraction, seed=calibration_seed)

    calibration_errors = _errors_for(model, split.calibration, image_size)
    threshold = compute_threshold(calibration_errors, k=threshold_k)

    held_out_errors = _errors_for(model, split.held_out, image_size)
    held_out_false_positives = sum(1 for error in held_out_errors if error > threshold)

    calibration_info = CalibrationInfo(
        method="mean_plus_k_std_on_calibration_subset",
        threshold_k=threshold_k,
        calibration_fraction=split.fraction,
        calibration_seed=split.seed,
        calibration_sample_count=len(split.calibration),
        held_out_sample_count=len(split.held_out),
        held_out_mean_error=statistics.fmean(held_out_errors) if held_out_errors else float("nan"),
        held_out_max_error=max(held_out_errors) if held_out_errors else float("nan"),
        held_out_false_positive_count=held_out_false_positives,
    )

    test_samples = discover_test_samples(category)
    if not test_samples:
        raise ValueError(f"No test images found for category {category!r}; cannot evaluate.")

    eval_samples: list[EvaluationSample] = []
    total_preprocessing_ms = 0.0
    total_inference_ms = 0.0

    for sample in test_samples:
        t0 = time.perf_counter()
        tensor = _sample_to_tensor(sample, image_size)
        t1 = time.perf_counter()
        error = compute_reconstruction_error(model, tensor)
        t2 = time.perf_counter()

        total_preprocessing_ms += (t1 - t0) * 1000
        total_inference_ms += (t2 - t1) * 1000

        predicted_label = DEFECTIVE_LABEL if error > threshold else GOOD_LABEL
        eval_samples.append(
            EvaluationSample(
                path=sample.path,
                defect_type=sample.defect_type,
                label=sample.label,
                reconstruction_error=error,
                predicted_label=predicted_label,
            )
        )

    ground_truth = [s.label for s in eval_samples]
    predictions = [s.predicted_label for s in eval_samples]
    errors = [s.reconstruction_error for s in eval_samples]

    metrics = compute_classification_metrics(ground_truth, predictions)

    good_errors = [s.reconstruction_error for s in eval_samples if s.label == GOOD_LABEL]
    defective_errors = [s.reconstruction_error for s in eval_samples if s.label == DEFECTIVE_LABEL]
    num_correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)

    sample_count = len(eval_samples)
    timing = TimingSummary(
        model_load_ms=model_load_ms,
        final_test_sample_count=sample_count,
        total_preprocessing_ms=total_preprocessing_ms,
        total_inference_ms=total_inference_ms,
        mean_preprocessing_ms_per_image=total_preprocessing_ms / sample_count if sample_count else float("nan"),
        mean_inference_ms_per_image=total_inference_ms / sample_count if sample_count else float("nan"),
        mean_total_ms_per_image=(total_preprocessing_ms + total_inference_ms) / sample_count
        if sample_count
        else float("nan"),
    )

    return ValidatedEvaluationResult(
        category=category,
        num_training_normal_samples=len(train_samples),
        total_test_samples=sample_count,
        good_test_count=len(good_errors),
        defective_test_count=len(defective_errors),
        threshold=threshold,
        calibration=calibration_info,
        samples=eval_samples,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        true_negatives=metrics.true_negatives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        true_positives=metrics.true_positives,
        min_error=min(errors),
        max_error=max(errors),
        mean_error=statistics.fmean(errors),
        mean_good_error=statistics.fmean(good_errors) if good_errors else float("nan"),
        mean_defective_error=statistics.fmean(defective_errors) if defective_errors else float("nan"),
        num_correct=num_correct,
        num_incorrect=sample_count - num_correct,
        timing=timing,
    )
