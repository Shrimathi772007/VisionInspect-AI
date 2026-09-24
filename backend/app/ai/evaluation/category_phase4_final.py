"""Milestone 4 Phase 4: the ONE final-test evaluation, baseline comparison, adoption rule and audit.

Kept apart from app.ai.evaluation.category_phase4 (the validation-only experiment) so that the
module which chooses a candidate has no way to reach test images. The single entry point here,
`evaluate_locked_on_final_test`, accepts only a `LockedSelection` - the frozen candidate +
threshold produced by the experiment - verifies that the locked model file is byte-identical to
what was locked, and scores the test set exactly once.

ADOPTION RULE (declared before any final-test result existed; never altered afterwards)
---------------------------------------------------------------------------------------
The candidate is chosen without the test set. Whether it is worth *retaining over* the protected
Phase 3 baseline is a separate question, answered by comparing the two on the final test with three
criteria that must ALL hold:

  A1  F1 is strictly higher than the baseline's.
  A2  The false-positive rate is at most 10% (the inherited validation ceiling, applied here too).
  A3  The recall gain is not plausibly luck: among defective images on which the two models
      disagree, the candidate wins significantly more often than it loses - one-sided exact sign
      test (binomial, p = 0.5) with p < 0.05.

Otherwise the Phase 3 baseline is retained. These are engineering criteria, not specification
values, and 110 test images is a small sample.
"""

import ast
import inspect
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

from app.ai.evaluation import category_phase4 as experiment_module
from app.ai.evaluation.category_phase3 import (
    defect_type_recall,
    metrics_from_predictions,
    predictions_at_threshold,
    split_leakage_checks,
)
from app.ai.evaluation.category_phase4 import (
    EVENT_SELECTION_FROZEN,
    ExperimentResult,
    LockedSelection,
    evaluate_on_samples,
    run_candidate,
    select_stage_winner,
)
from app.ai.evaluation.phase3_threshold_selection import select_threshold_from_validation
from app.ai.training.artifacts import load_model
from app.ai.training.dataset import discover_test_samples
from app.ai.training.model import build_model
from app.ai.training.schemas import DatasetSample
from app.ai.training.validation_split import TrainValidationSplit

EVENT_FINAL_TEST_SCORED = "final_test_scored"

ADOPTION_MAX_FALSE_POSITIVE_RATE = 0.10
ADOPTION_SIGN_TEST_ALPHA = 0.05


@dataclass
class FinalTestResult:
    locked_digest: str
    test_samples: list[DatasetSample]
    test_errors: list[float]
    predictions: list[dict]
    metrics: dict
    recall_by_defect_type: dict
    total_ms: float
    preprocess_ms_per_image: float
    inference_ms_per_image: float
    events: list[str] = field(default_factory=list)  # experiment events + the scoring event, in order


def _file_md5(path: Path) -> str:
    import hashlib

    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_locked_on_final_test(category: str, locked: LockedSelection, experiment_events: list[str]) -> FinalTestResult:
    """Score the final test set ONCE with the frozen candidate and threshold.

    Refuses to run unless the selection was frozen (its event is recorded) and the locked model
    file still has the hash that was locked.
    """
    if EVENT_SELECTION_FROZEN not in experiment_events:
        raise RuntimeError("Refusing to touch the final test set: the selection has not been frozen.")
    actual_md5 = _file_md5(locked.model_path)
    if actual_md5 != locked.model_md5:
        raise RuntimeError(
            f"Locked model changed after freezing: expected MD5 {locked.model_md5}, found {actual_md5}."
        )

    model = load_model(locked.model_path, build_model())
    test_samples = discover_test_samples(category)  # first (and only) test-set access in Phase 4
    errors, total_ms, prep_ms, inf_ms = evaluate_on_samples(model, test_samples, locked.image_size)

    predictions = predictions_at_threshold(test_samples, errors, locked.threshold)
    return FinalTestResult(
        locked_digest=locked.digest,
        test_samples=test_samples,
        test_errors=errors,
        predictions=predictions,
        metrics=metrics_from_predictions(predictions),
        recall_by_defect_type=defect_type_recall(predictions),
        total_ms=total_ms,
        preprocess_ms_per_image=prep_ms,
        inference_ms_per_image=inf_ms,
        events=list(experiment_events) + [EVENT_FINAL_TEST_SCORED],
    )


# ---------------------------------------------------------------------------
# Baseline comparison + adoption rule
# ---------------------------------------------------------------------------

def sign_test_p_value(wins: int, losses: int) -> float:
    """One-sided exact binomial (p=0.5) probability of >= `wins` wins among wins+losses ties-removed pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2**n


def compare_to_baseline(baseline_predictions: list[dict], baseline_metrics: dict, candidate: FinalTestResult) -> dict:
    """Metric deltas, paired per-image disagreement analysis, and the adoption decision."""
    base = {p["file"]: p for p in baseline_predictions}
    cand = {p["file"]: p for p in candidate.predictions}
    if set(base) != set(cand):
        raise ValueError("Baseline and candidate predictions cover different test images.")

    wins = sum(1 for f, p in cand.items() if p["label"] == 1 and p["predicted_label"] == 1 and base[f]["predicted_label"] == 0)
    losses = sum(1 for f, p in cand.items() if p["label"] == 1 and p["predicted_label"] == 0 and base[f]["predicted_label"] == 1)
    p_value = sign_test_p_value(wins, losses)

    m = candidate.metrics
    deltas = {
        key: {"baseline": baseline_metrics[key], "candidate": m[key], "absolute_change": m[key] - baseline_metrics[key]}
        for key in (
            "accuracy", "precision", "recall", "f1_score",
            "false_positives", "false_negatives", "true_positives", "true_negatives",
        )
    }

    a1 = m["f1_score"] > baseline_metrics["f1_score"]
    a2 = m["false_defect_detection_rate"] <= ADOPTION_MAX_FALSE_POSITIVE_RATE
    a3 = wins > losses and p_value < ADOPTION_SIGN_TEST_ALPHA
    adopt = a1 and a2 and a3
    return {
        "metrics": deltas,
        "paired_defective_images": {
            "newly_detected_by_candidate": wins,
            "newly_missed_by_candidate": losses,
            "sign_test_one_sided_p_value": p_value,
        },
        "adoption_criteria": {
            "A1_f1_strictly_higher": a1,
            "A2_false_positive_rate_at_most_10_percent": a2,
            "A3_recall_gain_significant_sign_test_p_lt_0_05": a3,
        },
        "decision": "OPTIMIZED CANDIDATE RETAINED (pending review)" if adopt else "PHASE 3 BASELINE RETAINED",
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def module_references(module, names: set[str]) -> set[str]:
    """Which of `names` the module's source actually imports or uses (AST-based, so docstrings and
    comments that merely mention a name do not count)."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    found |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    found |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    return found & names


def audit_phase4_leakage(
    category: str,
    split: TrainValidationSplit,
    experiments: list[ExperimentResult],
    final: FinalTestResult | None,
    baseline_validation_stats: dict | None = None,
) -> dict:
    """Independent, data-derived verification of every Phase 4 protection."""
    test_samples = final.test_samples if final is not None else discover_test_samples(category)  # paths only
    checks = split_leakage_checks(category, split, test_samples)

    training_count = len(split.training)
    checks["every_candidate_trained_on_exactly_the_training_subset"] = all(
        o.num_training_images == training_count for e in experiments for o in e.outcomes.values()
    )
    checks["experiment_module_cannot_list_test_images"] = not module_references(
        experiment_module, {"discover_test_samples"}
    )
    checks["experiment_events_contain_no_test_access"] = all(
        not e.startswith("final_test") for x in experiments for e in x.events
    )
    checks["threshold_selection_receives_validation_data_only"] = set(
        inspect.signature(select_threshold_from_validation).parameters
    ) == {"candidates", "validation_errors", "max_validation_false_positive_rate"}
    checks["candidate_run_and_stage_selection_receive_no_test_data"] = (
        set(inspect.signature(run_candidate).parameters) == {"config", "split", "model_path", "phase3_reference"}
        and set(inspect.signature(select_stage_winner).parameters) == {"stage", "outcomes"}
    )
    checks["selection_frozen_in_every_experiment"] = all(EVENT_SELECTION_FROZEN in x.events for x in experiments)
    checks["baseline_reuse_only_for_exact_baseline_configuration"] = all(
        (not o.reused_phase3_artifact)
        or (o.config.training_config.image_size, o.config.training_config.epochs, o.config.training_config.learning_rate)
        == ((128, 128), 15, 1e-3)
        for e in experiments
        for o in e.outcomes.values()
    )
    if baseline_validation_stats is not None:
        base = next(
            (o for o in experiments[0].outcomes.values() if o.reused_phase3_artifact), None
        )
        checks["split_matches_phase3_baseline_validation_statistics"] = (
            base is not None
            and abs(base.validation_mean - baseline_validation_stats["mean_error"]) <= 1e-9 * baseline_validation_stats["mean_error"]
            and abs(base.validation_std - baseline_validation_stats["std_error"]) <= 1e-9 * baseline_validation_stats["std_error"]
        )
    if final is not None:
        order = {e: i for i, e in enumerate(final.events)}
        checks["final_test_scored_after_selection_frozen"] = order.get(EVENT_SELECTION_FROZEN, 10**9) < order.get(
            EVENT_FINAL_TEST_SCORED, -1
        )
        checks["final_test_scored_exactly_once"] = final.events.count(EVENT_FINAL_TEST_SCORED) == 1
        checks["final_test_scored_only_the_frozen_selection"] = final.locked_digest == experiments[0].locked.digest
    checks["all_passed"] = all(checks.values())
    return checks

