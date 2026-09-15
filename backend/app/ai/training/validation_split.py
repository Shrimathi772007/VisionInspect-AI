"""Milestone 4 Phase 3: a TRUE train/validation split.

Unlike Phase 1/2's calibration split (app.ai.evaluation.calibration -
which computes threshold *statistics* from a subset of reconstruction
errors but still trains the model on all 209 train/good images), THIS split
changes what the model itself is trained on: only the `training` subset is
ever passed through the training DataLoader (see
app.ai.training.phase3_dataset.GoodImageSubsetDataset and
app.ai.training.phase3_train.train_anomaly_model_on_samples). The
`validation` subset is genuinely unseen by the resulting model - it never
contributes a gradient, unlike Phase 1/2's "held-out-42".

Reuses the exact same deterministic partitioning algorithm as
app.ai.evaluation.calibration.split_for_calibration (same
numpy.random.default_rng(seed).permutation logic, same default
fraction=0.8/seed=42) so that, given the same samples/fraction/seed, this
produces the identical 167/42 image partition Phase 1/2 already used for
calibration/held-out - the difference is entirely in how the two subsets
are *used* afterward (Phase 3 excludes `validation` from training; Phase 1/2
never did), not in how they are computed.
"""

from dataclasses import dataclass

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_CALIBRATION_SEED,
    split_for_calibration,
)
from app.ai.training.schemas import DatasetSample

DEFAULT_TRAINING_FRACTION = DEFAULT_CALIBRATION_FRACTION  # 0.8
DEFAULT_SPLIT_SEED = DEFAULT_CALIBRATION_SEED  # 42


@dataclass(frozen=True)
class TrainValidationSplit:
    """A true train/validation partition of train/good samples.

    `training` is what the Phase 3 model is fit on; `validation` is
    excluded from training entirely and used only afterward, for threshold
    selection (see app.ai.evaluation.phase3_threshold_selection).
    """

    training: list[DatasetSample]
    validation: list[DatasetSample]
    seed: int
    training_fraction: float


def split_train_validation(
    samples: list[DatasetSample],
    training_fraction: float = DEFAULT_TRAINING_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> TrainValidationSplit:
    """Deterministically split `samples` into a training subset and a validation subset.

    Raises ValueError under the same conditions as split_for_calibration
    (fewer than 2 samples, or `training_fraction` outside (0, 1)).
    """
    split = split_for_calibration(samples, fraction=training_fraction, seed=seed)
    return TrainValidationSplit(
        training=split.calibration,
        validation=split.held_out,
        seed=split.seed,
        training_fraction=split.fraction,
    )
