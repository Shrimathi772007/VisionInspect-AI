"""Milestone 4 Phase 2: tests for threshold-candidate generation, calibration-only
selection, final-test scoring, and report serialization.

Does not modify or duplicate Phase 1 tests (test_ai_evaluation.py,
test_ai_evaluation_validation.py) - those keep covering evaluate_model and
evaluate_model_validated unchanged; this file covers only the new Phase 2
additions.
"""

import json
import math

import numpy as np
import pytest

from app.ai.evaluation.metrics import DEFECTIVE_LABEL, GOOD_LABEL
from app.ai.evaluation.model_metadata import compute_model_metadata
from app.ai.evaluation.phase2_report import build_phase2_report, save_phase2_report
from app.ai.evaluation.threshold import compute_threshold
from app.ai.evaluation.threshold_experiments import (
    METHOD_MEAN_STD,
    METHOD_PERCENTILE,
    ThresholdCandidate,
    evaluate_candidate_on_final_test,
    generate_candidates,
)
from app.ai.evaluation.threshold_selection import (
    STATUS_NO_DEFENSIBLE_WINNER,
    STATUS_PROVISIONAL,
    select_threshold,
)
from app.ai.training import build_model, save_model
from app.ai.training.schemas import DatasetStatistics

CALIBRATION_ERRORS = [0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0100]


# ---------------------------------------------------------------------------
# Candidate generation - mean+K*std
# ---------------------------------------------------------------------------

def test_mean_std_candidate_matches_compute_threshold():
    candidates = generate_candidates(CALIBRATION_ERRORS, k_values=[2.0], percentiles=[])
    assert len(candidates) == 1
    candidate = candidates[0]

    assert candidate.method == METHOD_MEAN_STD
    assert candidate.parameter == 2.0
    assert candidate.threshold == pytest.approx(compute_threshold(CALIBRATION_ERRORS, k=2.0))
    assert candidate.calibration_percentile is None


def test_mean_std_candidates_decrease_as_k_decreases():
    candidates = generate_candidates(CALIBRATION_ERRORS, k_values=[3.0, 2.0, 1.0], percentiles=[])
    thresholds = [c.threshold for c in candidates]
    assert thresholds == sorted(thresholds, reverse=True)  # K=3 threshold >= K=2 >= K=1


# ---------------------------------------------------------------------------
# Candidate generation - percentile
# ---------------------------------------------------------------------------

def test_percentile_candidate_matches_numpy_percentile():
    candidates = generate_candidates(CALIBRATION_ERRORS, k_values=[], percentiles=[90.0])
    assert len(candidates) == 1
    candidate = candidates[0]

    assert candidate.method == METHOD_PERCENTILE
    assert candidate.parameter == 90.0
    assert candidate.calibration_percentile == 90.0
    assert candidate.threshold == pytest.approx(float(np.percentile(CALIBRATION_ERRORS, 90.0)))


def test_higher_percentile_gives_higher_threshold():
    candidates = generate_candidates(CALIBRATION_ERRORS, k_values=[], percentiles=[50.0, 95.0, 99.0])
    thresholds = [c.threshold for c in candidates]
    assert thresholds == sorted(thresholds)


def test_generate_candidates_raises_on_empty_calibration_errors():
    with pytest.raises(ValueError):
        generate_candidates([], k_values=[3.0], percentiles=[])


def test_generate_candidates_default_produces_nine_candidates():
    candidates = generate_candidates(CALIBRATION_ERRORS)
    assert len(candidates) == 9  # 5 K values + 4 percentiles
    assert sum(1 for c in candidates if c.method == METHOD_MEAN_STD) == 5
    assert sum(1 for c in candidates if c.method == METHOD_PERCENTILE) == 4


# ---------------------------------------------------------------------------
# Determinism - candidate generation is a pure function
# ---------------------------------------------------------------------------

def test_candidate_generation_is_deterministic():
    candidates_a = generate_candidates(CALIBRATION_ERRORS)
    candidates_b = generate_candidates(CALIBRATION_ERRORS)
    assert [c.threshold for c in candidates_a] == [c.threshold for c in candidates_b]


# ---------------------------------------------------------------------------
# Final-test scoring reuses compute_classification_metrics exactly
# ---------------------------------------------------------------------------

def test_evaluate_candidate_on_final_test_known_case():
    # 4 actual good, 4 actual defective; threshold misses two defectives.
    candidate = ThresholdCandidate(
        method=METHOD_MEAN_STD,
        parameter=3.0,
        threshold=0.05,
        calibration_sample_count=10,
        calibration_mean=0.02,
        calibration_std=0.01,
        calibration_percentile=None,
    )
    test_labels = [GOOD_LABEL, GOOD_LABEL, GOOD_LABEL, GOOD_LABEL, DEFECTIVE_LABEL, DEFECTIVE_LABEL, DEFECTIVE_LABEL, DEFECTIVE_LABEL]
    test_errors = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.04, 0.03]  # last two stay below threshold -> missed

    result = evaluate_candidate_on_final_test(candidate, test_labels, test_errors)

    assert result.true_negatives == 4
    assert result.false_positives == 0
    assert result.true_positives == 2
    assert result.false_negatives == 2
    assert result.total_test_samples == 8
    assert result.accuracy == pytest.approx(6 / 8)
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.false_negative_rate == pytest.approx(2 / 4)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1_score <= 1.0


def test_evaluate_candidate_never_accepts_a_pre_touched_threshold_source():
    """Structural guard: ThresholdCandidate/evaluate_candidate_on_final_test have no
    parameter through which calibration could be recomputed from test data - the
    threshold is a plain float already fixed before this function is ever called."""
    candidate = ThresholdCandidate(
        method=METHOD_PERCENTILE,
        parameter=95.0,
        threshold=0.01,
        calibration_sample_count=5,
        calibration_mean=0.005,
        calibration_std=0.001,
        calibration_percentile=95.0,
    )
    import inspect

    params = list(inspect.signature(evaluate_candidate_on_final_test).parameters)
    assert params == ["candidate", "test_labels", "test_errors"]


# ---------------------------------------------------------------------------
# Selection - calibration/held-out only, never given test data
# ---------------------------------------------------------------------------

def test_select_threshold_signature_has_no_test_data_parameter():
    import inspect

    params = list(inspect.signature(select_threshold).parameters)
    assert params == ["candidates", "held_out_errors", "max_held_out_false_positive_rate"]
    assert "test_errors" not in params
    assert "test_labels" not in params


def test_select_threshold_rejects_unstable_candidates():
    low = ThresholdCandidate(METHOD_MEAN_STD, 1.0, threshold=0.01, calibration_sample_count=10,
                              calibration_mean=0.005, calibration_std=0.002, calibration_percentile=None)
    high = ThresholdCandidate(METHOD_MEAN_STD, 3.0, threshold=0.05, calibration_sample_count=10,
                               calibration_mean=0.005, calibration_std=0.002, calibration_percentile=None)
    held_out_errors = [0.02, 0.03, 0.015, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]  # 3/10 exceed `low`

    result = select_threshold([low, high], held_out_errors, max_held_out_false_positive_rate=0.10)

    assert result.status == STATUS_PROVISIONAL
    assert result.selected is high  # `low` fails the 10% stability ceiling (3/10 = 30%), `high` passes (0/10)


def test_select_threshold_prefers_lowest_passing_threshold():
    candidates = [
        ThresholdCandidate(METHOD_MEAN_STD, 3.0, threshold=0.05, calibration_sample_count=10,
                            calibration_mean=0.02, calibration_std=0.01, calibration_percentile=None),
        ThresholdCandidate(METHOD_MEAN_STD, 2.0, threshold=0.03, calibration_sample_count=10,
                            calibration_mean=0.02, calibration_std=0.005, calibration_percentile=None),
    ]
    held_out_errors = [0.001] * 10  # nothing exceeds either threshold -> both pass

    result = select_threshold(candidates, held_out_errors)

    assert result.status == STATUS_PROVISIONAL
    assert result.selected.parameter == 2.0  # lower threshold preferred among passing candidates


def test_select_threshold_returns_no_defensible_winner_when_all_unstable():
    candidates = [
        ThresholdCandidate(METHOD_MEAN_STD, 1.0, threshold=0.001, calibration_sample_count=10,
                            calibration_mean=0.005, calibration_std=0.002, calibration_percentile=None),
    ]
    held_out_errors = [0.01] * 10  # all 10 exceed the threshold -> 100% FP rate

    result = select_threshold(candidates, held_out_errors, max_held_out_false_positive_rate=0.10)

    assert result.status == STATUS_NO_DEFENSIBLE_WINNER
    assert result.selected is None
    assert len(result.stability_by_candidate) == 1
    assert result.stability_by_candidate[0].held_out_false_positive_rate == pytest.approx(1.0)


def test_selection_status_never_claims_fully_selected():
    """The status string must never claim a full, validated selection - only PROVISIONAL
    or NO_DEFENSIBLE_WINNER - because the held-out-42 subset is not independent validation
    data (see threshold_selection.py module docstring)."""
    assert "PROVISIONAL" in STATUS_PROVISIONAL
    assert "NOT VALIDATED" in STATUS_PROVISIONAL
    assert STATUS_PROVISIONAL != "SELECTED"


def test_select_threshold_is_deterministic():
    candidates = generate_candidates(CALIBRATION_ERRORS)
    held_out_errors = [0.0021, 0.0099, 0.0022]

    result_a = select_threshold(candidates, held_out_errors)
    result_b = select_threshold(candidates, held_out_errors)

    assert result_a.status == result_b.status
    assert (result_a.selected is None) == (result_b.selected is None)
    if result_a.selected is not None:
        assert result_a.selected.threshold == result_b.selected.threshold


# ---------------------------------------------------------------------------
# Leakage protection: calibration-only samples never include test-split data
# ---------------------------------------------------------------------------

def test_generate_candidates_uses_only_the_errors_it_is_given():
    """Structural leakage guard: generate_candidates has no dataset/category
    parameter - it cannot reach for test images even if called incorrectly,
    it can only ever see the calibration_errors list explicitly passed in."""
    import inspect

    params = list(inspect.signature(generate_candidates).parameters)
    assert params == ["calibration_errors", "k_values", "percentiles"]
    assert "test_errors" not in params
    assert "category" not in params


# ---------------------------------------------------------------------------
# Report build/save
# ---------------------------------------------------------------------------

def test_build_and_save_phase2_report_round_trips(tmp_path):
    model = build_model()
    model_path = tmp_path / "ai_models" / "widget" / "autoencoder.pt"
    save_model(model, model_path)
    metadata = compute_model_metadata(model_path, category="widget", latent_channels=model.latent_channels)

    stats = DatasetStatistics(
        category="widget", train_good_count=10, test_good_count=2, test_defective_count=2,
        test_total_count=4, test_defect_type_counts={"good": 2, "broken": 2},
    )

    candidates = generate_candidates(CALIBRATION_ERRORS, k_values=[3.0, 1.0], percentiles=[95.0])
    test_labels = [GOOD_LABEL, GOOD_LABEL, DEFECTIVE_LABEL, DEFECTIVE_LABEL]
    test_errors = [0.001, 0.002, 0.02, 0.03]
    candidate_results = [evaluate_candidate_on_final_test(c, test_labels, test_errors) for c in candidates]
    selection = select_threshold(candidates, held_out_errors=[0.001, 0.002, 0.0021])

    report = build_phase2_report(
        model_metadata=metadata,
        dataset_stats=stats,
        git_branch="test-branch",
        git_head="deadbeef",
        calibration_fraction=0.8,
        calibration_seed=42,
        calibration_sample_count=len(CALIBRATION_ERRORS),
        held_out_sample_count=3,
        phase1_baseline={"threshold": 0.0032, "confusion_matrix": [[19, 1], [33, 30]]},
        candidate_results=candidate_results,
        selection=selection,
        reproducibility={"run_count": 2, "overall_reproducible": True},
        timing_ms={"model_load_ms": 1.0},
    )

    report_path = save_phase2_report(report, tmp_path / "reports" / "phase2_threshold_experiment_report.json")
    assert report_path.is_file()

    reloaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert reloaded["report_type"] == "milestone_4_phase_2_threshold_experiment"
    assert reloaded["git"]["branch"] == "test-branch"
    assert reloaded["git"]["head"] == "deadbeef"
    assert len(reloaded["candidates"]) == 3
    assert reloaded["selection"]["status"] in (STATUS_PROVISIONAL, STATUS_NO_DEFENSIBLE_WINNER)
    assert reloaded["leakage_verification"]["final_test_used_for_threshold_selection"] is False
    assert reloaded["leakage_verification"]["held_out_42_described_as_independent_validation"] is False
    assert len(reloaded["limitations"]) >= 1


def test_phase2_report_never_overwrites_phase1_report_path():
    from app.ai.evaluation.phase2_report import default_phase2_report_path
    from app.ai.evaluation.report import default_report_path

    assert default_phase2_report_path("bottle") != default_report_path("bottle")
    assert default_phase2_report_path("bottle").name == "phase2_threshold_experiment_report.json"
    assert default_report_path("bottle").name == "phase1_validation_report.json"


# ---------------------------------------------------------------------------
# Reproducibility across two full experiment "runs" (candidate gen + selection)
# ---------------------------------------------------------------------------

def test_full_candidate_and_selection_pipeline_reproducible_across_two_runs():
    def run():
        candidates = generate_candidates(CALIBRATION_ERRORS)
        held_out_errors = [0.0025, 0.0099, 0.0021, 0.0026]
        selection = select_threshold(candidates, held_out_errors)
        return candidates, selection

    candidates_1, selection_1 = run()
    candidates_2, selection_2 = run()

    assert [c.threshold for c in candidates_1] == [c.threshold for c in candidates_2]
    assert selection_1.status == selection_2.status
    if selection_1.selected is not None:
        assert selection_1.selected.threshold == selection_2.selected.threshold
        assert math.isfinite(selection_1.selected.threshold)
