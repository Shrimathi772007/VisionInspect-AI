"""Anomaly threshold derived ONLY from normal (training) reconstruction error.

    threshold = mean(normal_errors) + K * std(normal_errors)

This is the standard "K-sigma" outlier rule: since the autoencoder was
trained exclusively to reconstruct normal images well, its reconstruction
error on normal data should cluster tightly around a mean; a genuinely
anomalous image is expected to reconstruct poorly and produce an unusually
high error relative to that normal distribution.

K=3 is used by default (three-sigma rule): for an approximately normal
error distribution, ~99.7% of normal samples fall within 3 standard
deviations of the mean, so this threshold is deliberately conservative -
it should rarely flag a genuinely normal image, while still catching
reconstruction errors that are statistical outliers relative to normal
appearance.

Critically: this threshold is computed ONLY from train/good reconstruction
errors. Defective test images and their labels are never used here - doing
so would leak the test set into threshold selection and invalidate the
evaluation as a genuine, unseen test.
"""

from collections.abc import Sequence

import numpy as np

DEFAULT_THRESHOLD_K = 3.0


def compute_threshold(normal_errors: Sequence[float], k: float = DEFAULT_THRESHOLD_K) -> float:
    if not normal_errors:
        raise ValueError("Cannot compute a threshold from an empty set of normal reconstruction errors.")

    errors = np.asarray(list(normal_errors), dtype=np.float64)
    return float(errors.mean() + k * errors.std())
