"""Data structures produced by evaluating a trained anomaly-detection model."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvaluationSample:
    """One evaluated test image - ground truth kept strictly separate from the prediction."""

    path: Path
    defect_type: str
    label: int  # ground truth: 0 good, 1 defective (MVTec folder taxonomy)
    reconstruction_error: float  # computed WITHOUT ever looking at `label`
    predicted_label: int  # 0 or 1, derived only from reconstruction_error vs. threshold


@dataclass
class EvaluationResult:
    """Real evaluation outcome for one category - every field is a measured/derived value."""

    category: str

    # Dataset composition (real counts)
    num_training_normal_samples: int
    total_test_samples: int
    good_test_count: int
    defective_test_count: int

    # Threshold (derived only from training-normal reconstruction errors)
    threshold: float
    threshold_k: float

    # Per-sample results
    samples: list[EvaluationSample] = field(default_factory=list)

    # Classification metrics (scikit-learn)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Confusion matrix, in [[TN, FP], [FN, TP]] form (rows=actual, cols=predicted; 0=good, 1=defective)
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0

    # Reconstruction-error statistics
    min_error: float = 0.0
    max_error: float = 0.0
    mean_error: float = 0.0
    mean_good_error: float = 0.0
    mean_defective_error: float = 0.0

    num_correct: int = 0
    num_incorrect: int = 0

    @property
    def confusion_matrix(self) -> list[list[int]]:
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]
