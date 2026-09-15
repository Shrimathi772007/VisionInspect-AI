"""Deterministic calibration/held-out split over train/good samples.

MVTec AD ships no official validation split, and the bottle training set is
good-only, so there is no separate "normal but unused-in-training" pool to
draw a classic validation set from. Rather than retrain the model on a
subset (which Milestone 4 Phase 1 explicitly avoids - the shipped model
artifact and its training data stay exactly as they are), this module
splits the *reconstruction-error computation used for threshold
calibration*, not the training data itself:

    train/good (209 images, unchanged - the model was fit on all of them)
        -> deterministic split (fixed seed) of the SAMPLE LIST only
        -> calibration subset  -> its reconstruction errors feed the threshold
        -> held-out subset     -> used only as a calibration-stability check

Honesty about what this does and doesn't prove: the held-out subset is still
part of the images the model was trained on (all 209 contributed to every
training epoch), so it is NOT a genuinely unseen validation set the way a
held-out split would be for a supervised classifier. It only isolates the
*threshold statistic's* input sample from the rest of the pool, which is
useful for checking that the calibration subset isn't producing a threshold
wildly different from the full pool - it is not evidence the threshold
generalizes to truly novel normal images. The final test set (test/good +
test/<defect_type>/*) is completely separate from this split and is never
touched by it.
"""

from dataclasses import dataclass

import numpy as np

from app.ai.training.schemas import DatasetSample

DEFAULT_CALIBRATION_FRACTION = 0.8
DEFAULT_CALIBRATION_SEED = 42


@dataclass(frozen=True)
class CalibrationSplit:
    """Result of deterministically splitting train/good samples for calibration."""

    calibration: list[DatasetSample]
    held_out: list[DatasetSample]
    fraction: float
    seed: int


def split_for_calibration(
    samples: list[DatasetSample],
    fraction: float = DEFAULT_CALIBRATION_FRACTION,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> CalibrationSplit:
    """Deterministically split `samples` into a calibration subset and a held-out subset.

    Reproducible: the same `samples` list, `fraction`, and `seed` always
    produce the same split, since the split is driven by `numpy.random.default_rng(seed)`
    rather than any process-global RNG state. `samples` should already be in a
    deterministic order (discover_train_samples sorts by filename), so the
    permutation below is the only source of shuffling.

    Raises ValueError if there are too few samples to produce a non-empty
    calibration and held-out subset, or if `fraction` is out of (0, 1).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples to split for calibration, got {len(samples)}")

    split_index = round(len(samples) * fraction)
    split_index = min(max(split_index, 1), len(samples) - 1)  # keep both subsets non-empty

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))

    calibration_indices = sorted(order[:split_index].tolist())
    held_out_indices = sorted(order[split_index:].tolist())

    return CalibrationSplit(
        calibration=[samples[i] for i in calibration_indices],
        held_out=[samples[i] for i in held_out_indices],
        fraction=fraction,
        seed=seed,
    )
