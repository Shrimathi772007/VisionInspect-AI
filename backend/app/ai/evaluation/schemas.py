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


@dataclass
class CalibrationInfo:
    """How the threshold used by a ValidatedEvaluationResult was calibrated."""

    method: str  # e.g. "mean_plus_k_std_on_calibration_subset"
    threshold_k: float
    calibration_fraction: float
    calibration_seed: int
    calibration_sample_count: int
    held_out_sample_count: int
    held_out_mean_error: float
    held_out_max_error: float
    held_out_false_positive_count: int  # held-out train/good samples whose error exceeds the threshold
    note: str = (
        "Held-out samples were still used to train the model (all train/good "
        "images contributed to every training epoch); this split only "
        "isolates them from the threshold's mean/std calculation, it is not "
        "a genuinely unseen validation set. It is a calibration-stability "
        "check, not evidence of generalization to novel normal images."
    )


@dataclass
class TimingSummary:
    """Measured wall-clock timing, in milliseconds, from one evaluation run."""

    model_load_ms: float
    final_test_sample_count: int
    total_preprocessing_ms: float
    total_inference_ms: float
    mean_preprocessing_ms_per_image: float
    mean_inference_ms_per_image: float
    mean_total_ms_per_image: float


@dataclass
class ValidatedEvaluationResult:
    """Phase 1 validated evaluation: leakage-free threshold calibration + full metrics.

    Distinct from EvaluationResult (which reproduces the original,
    train-good-based threshold methodology unchanged) - this result uses a
    threshold calibrated on a deterministic subset of train/good errors,
    documented in `calibration`, and evaluated once against the untouched
    final test set (identical test images to EvaluationResult - nothing is
    removed from the test set).
    """

    category: str

    num_training_normal_samples: int
    total_test_samples: int
    good_test_count: int
    defective_test_count: int

    threshold: float
    calibration: CalibrationInfo

    samples: list[EvaluationSample] = field(default_factory=list)

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0

    min_error: float = 0.0
    max_error: float = 0.0
    mean_error: float = 0.0
    mean_good_error: float = 0.0
    mean_defective_error: float = 0.0

    num_correct: int = 0
    num_incorrect: int = 0

    timing: TimingSummary | None = None

    @property
    def confusion_matrix(self) -> list[list[int]]:
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]
