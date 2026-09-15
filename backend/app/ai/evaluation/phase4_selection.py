"""Milestone 4 Phase 4: pick a stage winner using validation-only signals.

The validation set contains only good/normal images, so it cannot measure
defect recall, precision, F1, or true class separation directly. The
primary signal used here is the coefficient of variation (std/mean) of each
candidate's validation reconstruction errors: a tighter, more consistent
distribution of reconstruction error on known-normal images is the standard
justification, in reconstruction-based anomaly detection, for a model that
has learned a more confident representation of "normal" - lower CoV is
preferred. This is NOT a proxy for recall/F1 and is never claimed to be one.

Composes with the other validation-only signals Phase 4 allows:
    - validation false-positive rate at each candidate's own selected
      threshold (already enforced at <=10% by
      app.ai.evaluation.phase3_threshold_selection - a candidate with no
      threshold passing that ceiling is disqualified outright)
    - training stability (a finite, non-NaN final loss)
    - inference/validation performance and model size (used only as a
      tie-breaker, per the "prefer simpler/cheaper when tied" rule below)

Selection rule:
    1. Disqualify any candidate whose selection status is
       NO_DEFENSIBLE_WINNER (every one of ITS OWN threshold options failed
       the validation false-positive ceiling).
    2. Rank remaining candidates by coefficient of variation, ascending.
    3. If the best and next-best differ by less than TIE_TOLERANCE
       (relative), treat them as tied and break the tie by preferring fewer
       training epochs, then a smaller model file, then faster validation
       inference - i.e., prefer the simpler/cheaper candidate rather than
       manufacturing a difference the data doesn't support.
"""

from dataclasses import dataclass

from app.ai.evaluation.phase3_threshold_selection import STATUS_NO_DEFENSIBLE_WINNER
from app.ai.training.phase4_experiment import CandidateResult

TIE_TOLERANCE = 0.02  # 2% relative difference in coefficient of variation counts as a tie


@dataclass(frozen=True)
class StageSelectionResult:
    winner: CandidateResult
    reasoning: str
    ranked: list[tuple[str, float]]  # (candidate_id, coefficient_of_variation), best first


def _coefficient_of_variation(result: CandidateResult) -> float:
    return result.validation_std / result.validation_mean if result.validation_mean else float("inf")


def select_stage_winner(results: list[CandidateResult]) -> StageSelectionResult:
    if not results:
        raise ValueError("select_stage_winner requires at least one candidate result.")

    eligible = [r for r in results if r.selection.status != STATUS_NO_DEFENSIBLE_WINNER]
    if not eligible:
        raise ValueError(
            "No candidate in this stage produced a defensible validation threshold "
            "(every candidate's own threshold options exceeded the false-positive ceiling)."
        )

    ranked = sorted(eligible, key=_coefficient_of_variation)
    ranked_pairs = [(r.config.candidate_id, _coefficient_of_variation(r)) for r in ranked]

    best, best_cov = ranked[0], _coefficient_of_variation(ranked[0])
    tied = [r for r in ranked if best_cov > 0 and abs(_coefficient_of_variation(r) - best_cov) / best_cov <= TIE_TOLERANCE]
    if best_cov == 0:
        tied = [r for r in ranked if _coefficient_of_variation(r) == 0]

    if len(tied) > 1:
        winner = min(
            tied,
            key=lambda r: (
                r.config.training_config.epochs,
                r.model_metadata.artifact_size_bytes,
                r.validation_inference_ms,
            ),
        )
        reasoning = (
            f"{len(tied)} candidates tied within {TIE_TOLERANCE:.0%} coefficient-of-variation "
            f"tolerance (best CoV={best_cov:.4f}); selected '{winner.config.candidate_id}' as the "
            "simplest/cheapest tied option (fewest epochs, then smallest model file, then fastest "
            "validation inference) rather than manufacturing a difference the data doesn't support."
        )
    else:
        winner = best
        reasoning = (
            f"'{winner.config.candidate_id}' had the lowest validation reconstruction-error "
            f"coefficient of variation (std/mean = {best_cov:.4f}) among candidates that produced a "
            "defensible validation threshold - interpreted as the most consistent reconstruction of "
            "known-normal images, not as a direct recall/F1 proxy."
        )

    return StageSelectionResult(winner=winner, reasoning=reasoning, ranked=ranked_pairs)
