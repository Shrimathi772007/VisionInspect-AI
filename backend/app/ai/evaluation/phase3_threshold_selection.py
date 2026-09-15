"""Milestone 4 Phase 3: threshold selection from a GENUINE, unseen validation set.

Unlike Phase 2's selection (app.ai.evaluation.threshold_selection), which
could only ever be labeled PROVISIONAL because its "held-out-42" data was
not independent of training (all 209 train/good images, including those 42,
were used to fit the Phase 1/2 model), this module operates on validation
reconstruction errors from images that were STRUCTURALLY EXCLUDED from the
Phase 3 model's training DataLoader (see
app.ai.training.validation_split.split_train_validation and
app.ai.training.phase3_train.train_anomaly_model_on_samples - the
validation subset never appears in the training loop). A threshold chosen
here is deliberately labeled SELECTED, not PROVISIONAL.

This is still limited, and the limits are not hidden: the validation set
contains only good/normal images (MVTec provides no labeled-defective
validation split), so it can only establish FALSE-POSITIVE / stability
behavior on unseen normal images - it cannot measure recall or F1, which
require labeled defective examples. Those come only from the final 83-image
test set, evaluated once, after selection (see
app.ai.evaluation.threshold_experiments.evaluate_candidate_on_final_test).

Selection rule (fixed BEFORE the final-test set is touched):
    1. Reject any candidate whose validation false-positive rate exceeds
       MAX_VALIDATION_FALSE_POSITIVE_RATE.
    2. Among candidates that pass, prefer the LOWEST threshold - a lower
       reconstruction-error cutoff flags more outliers, the direction
       expected to increase recall, without abandoning the stability
       constraint from step 1.
    3. If no candidate passes step 1, nothing is selected.

MAX_VALIDATION_FALSE_POSITIVE_RATE reuses the same 10% ceiling Phase 2 used.
This is carried forward as an ENGINEERING OPERATING CONSTRAINT for
comparability across phases, NOT a value defined by the original project
specification - no such specification value exists in this repository (see
the Milestone 4 audit, which found no accessible spec document). A later
phase could legitimately revisit this ceiling against real production
tolerance for false alarms.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.evaluation.threshold_experiments import ThresholdCandidate

MAX_VALIDATION_FALSE_POSITIVE_RATE = 0.10  # inherited from Phase 2 as an operating constraint, not a spec value

STATUS_SELECTED = "SELECTED"
STATUS_NO_DEFENSIBLE_WINNER = "NO DEFENSIBLE WINNER"


@dataclass(frozen=True)
class ValidationStability:
    """One candidate's false-positive behavior on the genuine, unseen validation set."""

    candidate: ThresholdCandidate
    validation_sample_count: int
    validation_false_positive_count: int
    validation_false_positive_rate: float
    passes_stability_check: bool


@dataclass(frozen=True)
class Phase3SelectionResult:
    status: str  # STATUS_SELECTED | STATUS_NO_DEFENSIBLE_WINNER
    selected: ThresholdCandidate | None
    reasoning: str
    stability_by_candidate: list[ValidationStability]


def _evaluate_stability(
    candidate: ThresholdCandidate, validation_errors: Sequence[float], max_rate: float
) -> ValidationStability:
    count = len(validation_errors)
    fp_count = sum(1 for error in validation_errors if error > candidate.threshold)
    rate = fp_count / count if count else 0.0
    return ValidationStability(
        candidate=candidate,
        validation_sample_count=count,
        validation_false_positive_count=fp_count,
        validation_false_positive_rate=rate,
        passes_stability_check=rate <= max_rate,
    )


def select_threshold_from_validation(
    candidates: Sequence[ThresholdCandidate],
    validation_errors: Sequence[float],
    max_validation_false_positive_rate: float = MAX_VALIDATION_FALSE_POSITIVE_RATE,
) -> Phase3SelectionResult:
    """Choose a threshold candidate using ONLY genuine validation reconstruction errors.

    Deterministic: a pure function of its arguments. Never accepts or looks
    at final-test errors or labels - there is no parameter for them.
    """
    stability = [
        _evaluate_stability(c, validation_errors, max_validation_false_positive_rate) for c in candidates
    ]

    passing = [s for s in stability if s.passes_stability_check]
    if not passing:
        return Phase3SelectionResult(
            status=STATUS_NO_DEFENSIBLE_WINNER,
            selected=None,
            reasoning=(
                "No threshold candidate kept the validation false-positive rate at or "
                f"below {max_validation_false_positive_rate:.0%}; no defensible operating "
                "point could be selected from the genuine validation set alone."
            ),
            stability_by_candidate=stability,
        )

    chosen = min(passing, key=lambda s: s.candidate.threshold)
    reasoning = (
        f"Selected the lowest threshold ({chosen.candidate.method}, parameter="
        f"{chosen.candidate.parameter}) among candidates whose false-positive rate on the "
        f"genuine, unseen validation set stayed at or below "
        f"{max_validation_false_positive_rate:.0%} "
        f"({chosen.validation_false_positive_count}/{chosen.validation_sample_count} = "
        f"{chosen.validation_false_positive_rate:.1%}). A lower threshold flags more "
        "reconstruction-error outliers, which is the direction expected to increase recall. "
        "Labeled SELECTED (not PROVISIONAL) because, unlike Phase 2's held-out-42, this "
        "validation subset was structurally excluded from model training - it is genuinely "
        "unseen data. This still does not measure recall/F1 (validation contains only good "
        "images); see the Phase 3 report's Limitations section."
    )
    return Phase3SelectionResult(
        status=STATUS_SELECTED,
        selected=chosen.candidate,
        reasoning=reasoning,
        stability_by_candidate=stability,
    )
