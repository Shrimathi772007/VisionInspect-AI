"""Milestone 4 Phase 4: summary statistics for a repeated timing benchmark.

Small, standalone helper so benchmark-summary math (mean/median/min/max/
count) is unit-testable independent of the long-running Phase 4 script that
produces the raw timing samples.
"""

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingSummary:
    mean: float
    median: float
    min: float
    max: float
    count: int


def summarize_timings(values: list[float]) -> TimingSummary:
    if not values:
        raise ValueError("Cannot summarize an empty list of timing samples.")
    return TimingSummary(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        count=len(values),
    )
