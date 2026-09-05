"""Evaluate an already-trained anomaly-detection model on a category's held-out test set.

    trained model -> test images -> reconstruction error -> threshold
    (from training-normal errors) -> good/defective prediction -> real metrics

Reuses, rather than duplicates:
- app.ai.training.dataset (discover_train_samples / discover_test_samples / load_sample)
  for dataset discovery and the existing Phase 2 preprocessing pipeline.
- app.ai.training.artifacts.load_model_for_category for loading the trained model.
- app.ai.evaluation.threshold / metrics for threshold selection and scikit-learn metrics.

Does not retrain, modify, or otherwise touch the model architecture or weights.
"""

import statistics
from collections.abc import Sequence

import torch
from torch import nn

from app.ai.evaluation.metrics import DEFECTIVE_LABEL, GOOD_LABEL, compute_classification_metrics
from app.ai.evaluation.schemas import EvaluationResult, EvaluationSample
from app.ai.evaluation.threshold import DEFAULT_THRESHOLD_K, compute_threshold
from app.ai.training.dataset import discover_test_samples, discover_train_samples, load_sample
from app.ai.training.schemas import DatasetSample

DEFAULT_IMAGE_SIZE = (128, 128)  # matches the size used during training


def _sample_to_tensor(sample: DatasetSample, image_size: tuple[int, int]) -> torch.Tensor:
    """Model-ready tensor for one sample, via the existing preprocessing pipeline."""
    result = load_sample(sample, target_size=image_size)
    return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()


def compute_reconstruction_error(model: nn.Module, image_tensor: torch.Tensor) -> float:
    """Mean-squared reconstruction error for one CHW image tensor. Never sees any label."""
    model.eval()
    with torch.no_grad():
        batch = image_tensor.unsqueeze(0)
        reconstruction = model(batch)
        return float(torch.mean((reconstruction - batch) ** 2).item())


def _reconstruction_errors(
    model: nn.Module, samples: Sequence[DatasetSample], image_size: tuple[int, int]
) -> list[float]:
    return [compute_reconstruction_error(model, _sample_to_tensor(sample, image_size)) for sample in samples]


def evaluate_model(
    category: str,
    model: nn.Module,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    threshold_k: float = DEFAULT_THRESHOLD_K,
) -> EvaluationResult:
    """Evaluate `model` on `category`'s full test set, using a threshold derived
    only from training-normal (train/good) reconstruction error."""
    model.eval()

    train_samples = discover_train_samples(category)
    train_errors = _reconstruction_errors(model, train_samples, image_size)
    threshold = compute_threshold(train_errors, k=threshold_k)

    test_samples = discover_test_samples(category)
    if not test_samples:
        raise ValueError(f"No test images found for category {category!r}; cannot evaluate.")

    eval_samples: list[EvaluationSample] = []
    for sample in test_samples:
        error = compute_reconstruction_error(model, _sample_to_tensor(sample, image_size))
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

    return EvaluationResult(
        category=category,
        num_training_normal_samples=len(train_samples),
        total_test_samples=len(eval_samples),
        good_test_count=len(good_errors),
        defective_test_count=len(defective_errors),
        threshold=threshold,
        threshold_k=threshold_k,
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
        num_incorrect=len(ground_truth) - num_correct,
    )
