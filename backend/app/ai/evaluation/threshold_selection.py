"""Milestone 4 Phase 2: pre-registered, calibration-only threshold selection.

The rule below is fixed BEFORE any final-test evaluation happens - see
scripts/evaluate_bottle_model_phase2.py, which calls select_threshold() and
locks in its result before the final 83-image test set is touched at all.
select_threshold() itself has no way to see final-test data - it is only
ever given ThresholdCandidate objects (calibration statistics) and the
held-out-42 calibration errors, never test errors or test labels.

IMPORTANT LIMITATION (documented, not hidden): the held-out-42 subset is NOT
an independent/unseen validation set. It is 42 of the same 209 train/good
images the model was trained on (see app.ai.evaluation.calibration - the
split only isolates a subset of *reconstruction-error statistics*, it does
not exclude any image from training). Its false-positive rate at a given
threshold is used below only as a coarse calibration-stability proxy - does
this threshold flag an implausible fraction of images the model has already
learned to reconstruct well? - not as evidence of generalization to novel
normal images. Because of this, a candidate chosen by this rule is always
reported as PROVISIONAL, never as a fully validated production threshold:
genuinely validating a threshold would require retraining without the
validation images, which Milestone 4 Phase 2 explicitly does not do.

Fixed rule, in order:
    1. Reject any candidate whose held-out-42 false-positive rate exceeds
       MAX_HELD_OUT_FALSE_POSITIVE_RATE. Flagging more than this fraction of
       images the model reconstructs well (because it was trained on them)
       would be an unstable, impractically noisy operating point.
    2. Among candidates that pass, prefer the LOWEST threshold value: a
       lower reconstruction-error cutoff flags more outliers, which is the
       direction expected to increase recall - the stated Phase 2 objective
       - without abandoning the stability constraint from step 1.
    3. If no candidate passes step 1, nothing is selected.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.evaluation.threshold_experiments import ThresholdCandidate

MAX_HELD_OUT_FALSE_POSITIVE_RATE = 0.10  # at most 1 in 10 held-out calibration images

STATUS_PROVISIONAL = "PROVISIONAL - NOT VALIDATED"
STATUS_NO_DEFENSIBLE_WINNER = "NO DEFENSIBLE WINNER"


@dataclass(frozen=True)
class CandidateStability:
    """One candidate's held-out-42 calibration-stability check (not a test-set result)."""

    candidate: ThresholdCandidate
    held_out_sample_count: int
    held_out_false_positive_count: int
    held_out_false_positive_rate: float
    passes_stability_check: bool


@dataclass(frozen=True)
class SelectionResult:
    status: str  # STATUS_PROVISIONAL | STATUS_NO_DEFENSIBLE_WINNER
    selected: ThresholdCandidate | None
    reasoning: str
    stability_by_candidate: list[CandidateStability]


def _evaluate_stability(
    candidate: ThresholdCandidate, held_out_errors: Sequence[float], max_rate: float
) -> CandidateStability:
    count = len(held_out_errors)
    fp_count = sum(1 for error in held_out_errors if error > candidate.threshold)
    rate = fp_count / count if count else 0.0
    return CandidateStability(
        candidate=candidate,
        held_out_sample_count=count,
        held_out_false_positive_count=fp_count,
        held_out_false_positive_rate=rate,
        passes_stability_check=rate <= max_rate,
    )


def select_threshold(
    candidates: Sequence[ThresholdCandidate],
    held_out_errors: Sequence[float],
    max_held_out_false_positive_rate: float = MAX_HELD_OUT_FALSE_POSITIVE_RATE,
) -> SelectionResult:
    """Choose a threshold candidate using ONLY calibration/held-out data.

    Deterministic: a pure function of its arguments. Never accepts or looks
    at final-test errors or labels - there is no parameter for them.
    """
    stability = [_evaluate_stability(c, held_out_errors, max_held_out_false_positive_rate) for c in candidates]

    passing = [s for s in stability if s.passes_stability_check]
    if not passing:
        return SelectionResult(
            status=STATUS_NO_DEFENSIBLE_WINNER,
            selected=None,
            reasoning=(
                "No threshold candidate kept the held-out-42 calibration false-positive rate "
                f"at or below {max_held_out_false_positive_rate:.0%}; no defensible operating "
                "point could be selected from calibration data alone."
            ),
            stability_by_candidate=stability,
        )

    chosen = min(passing, key=lambda s: s.candidate.threshold)
    reasoning = (
        f"Selected the lowest threshold ({chosen.candidate.method}, parameter="
        f"{chosen.candidate.parameter}) among candidates whose held-out-42 false-positive rate "
        f"stayed at or below {max_held_out_false_positive_rate:.0%} "
        f"({chosen.held_out_false_positive_count}/{chosen.held_out_sample_count} = "
        f"{chosen.held_out_false_positive_rate:.1%}). A lower threshold flags more "
        "reconstruction-error outliers, which is the direction expected to increase recall. "
        "Labeled PROVISIONAL rather than SELECTED because the held-out-42 subset is not an "
        "independent/unseen validation set (see module docstring) - no threshold choice from "
        "this data can be called fully validated without retraining on a genuine "
        "train/validation split."
    )
    return SelectionResult(
        status=STATUS_PROVISIONAL,
        selected=chosen.candidate,
        reasoning=reasoning,
        stability_by_candidate=stability,
    )
