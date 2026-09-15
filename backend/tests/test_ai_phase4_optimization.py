"""Milestone 4 Phase 4: tests for staged candidate configuration, the
train-or-reuse-and-validate harness, validation-only stage selection, and
leakage protection against the final test set.

Does not modify or duplicate Phase 1-3 tests - those keep covering the
unchanged Phase 1-3 code. Uses small, fast synthetic datasets/configs
throughout (not the real 209/83 bottle dataset) so this suite runs quickly -
the real dataset is exercised once, deliberately, by
scripts/train_and_evaluate_bottle_phase4.py, not by the automated test suite.
"""

import inspect
import math
from pathlib import Path

import pytest

from app.ai.evaluation.benchmark_stats import TimingSummary, summarize_timings
from app.ai.evaluation.phase3_threshold_selection import STATUS_NO_DEFENSIBLE_WINNER
from app.ai.evaluation.phase4_selection import TIE_TOLERANCE, select_stage_winner
from app.ai.training.phase4_candidates import (
    BASELINE_EPOCHS,
    BASELINE_IMAGE_SIZE,
    BASELINE_LEARNING_RATE,
    CandidateConfig,
    stage1_candidates,
    stage2_candidates,
    stage3_candidate,
)
from app.ai.training.phase4_experiment import CandidateResult, run_candidate
from app.ai.training.validation_split import split_train_validation
from tests.conftest import make_image_bytes

CATEGORY = "widget"


def _write_fake_train_good(dataset_root: Path, category: str, count: int) -> None:
    good_dir = dataset_root / category / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(count):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=(i % 250, 40, 90)))


# ---------------------------------------------------------------------------
# 1: Candidate configuration validation
# ---------------------------------------------------------------------------

def test_stage1_candidates_cover_128_160_192_with_baseline_epochs_and_lr():
    candidates = stage1_candidates()
    sizes = {c.training_config.image_size for c in candidates}
    assert sizes == {(128, 128), (160, 160), (192, 192)}
    assert all(c.training_config.epochs == BASELINE_EPOCHS for c in candidates)
    assert all(c.training_config.learning_rate == BASELINE_LEARNING_RATE for c in candidates)
    assert all(c.training_config.seed == 42 for c in candidates)


def test_only_candidate_input128_reuses_the_phase3_artifact():
    candidates = stage1_candidates()
    reused = [c for c in candidates if c.reuse_phase3_artifact]
    assert len(reused) == 1
    assert reused[0].candidate_id == "candidate_input128"
    assert reused[0].training_config.image_size == BASELINE_IMAGE_SIZE


def test_stage2_candidates_vary_only_epochs():
    candidates = stage2_candidates((160, 160))
    assert {c.training_config.epochs for c in candidates} == {25, 35}
    assert all(c.training_config.image_size == (160, 160) for c in candidates)
    assert all(c.training_config.learning_rate == BASELINE_LEARNING_RATE for c in candidates)


def test_stage3_candidate_varies_only_learning_rate():
    candidate = stage3_candidate((160, 160), 25)
    assert candidate.training_config.learning_rate == 5e-4
    assert candidate.training_config.image_size == (160, 160)
    assert candidate.training_config.epochs == 25


def test_all_candidate_ids_are_unique():
    ids = [c.candidate_id for c in stage1_candidates()]
    ids += [c.candidate_id for c in stage2_candidates((160, 160))]
    ids.append(stage3_candidate((160, 160), 25).candidate_id)
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 2: No candidate can access final-test data during selection
# ---------------------------------------------------------------------------

def test_run_candidate_signature_has_no_final_test_parameter():
    params = list(inspect.signature(run_candidate).parameters)
    assert "test_samples" not in params
    assert "test_errors" not in params
    assert "test_labels" not in params


def test_select_stage_winner_signature_has_no_final_test_parameter():
    params = list(inspect.signature(select_stage_winner).parameters)
    assert params == ["results"]


def test_phase4_experiment_module_never_imports_test_discovery():
    import app.ai.training.phase4_experiment as module

    # Checks the module's actual namespace (what it imported), not its docstring prose.
    assert not hasattr(module, "discover_test_samples")


def test_phase4_selection_module_never_imports_test_discovery():
    import app.ai.evaluation.phase4_selection as module

    assert not hasattr(module, "discover_test_samples")


# ---------------------------------------------------------------------------
# 3: Candidate experiments use the exact Phase 3 train/validation split
# ---------------------------------------------------------------------------

def test_run_candidate_trains_only_on_given_training_samples(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 20)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.8, seed=42)
    config = CandidateConfig(
        candidate_id="candidate_test", stage="test",
        training_config=__import__("app.ai.training.schemas", fromlist=["TrainingConfig"]).TrainingConfig(
            category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, learning_rate=1e-3, seed=42
        ),
        notes="test",
    )
    model_path = tmp_path / "ai_models" / CATEGORY / "candidate_test" / "autoencoder.pt"

    result = run_candidate(config, split.training, split.validation, model_path)

    assert isinstance(result, CandidateResult)
    assert len(result.validation_errors) == len(split.validation)
    assert model_path.is_file()


# ---------------------------------------------------------------------------
# 4-5: Threshold selection uses validation only; final-test needs a lock
# ---------------------------------------------------------------------------

def test_candidate_result_selection_never_touches_test_data(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 20)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)
    split = split_train_validation(samples, training_fraction=0.8, seed=42)

    from app.ai.training.schemas import TrainingConfig
    config = CandidateConfig(
        candidate_id="candidate_test", stage="test",
        training_config=TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, seed=42),
        notes="test",
    )
    result = run_candidate(config, split.training, split.validation, tmp_path / "m.pt")

    # The selection's stability entries only ever reference validation sample counts.
    for stability in result.selection.stability_by_candidate:
        assert stability.validation_sample_count == len(split.validation)


def test_evaluate_candidate_on_final_test_requires_an_explicit_locked_candidate():
    from app.ai.evaluation.threshold_experiments import evaluate_candidate_on_final_test

    params = list(inspect.signature(evaluate_candidate_on_final_test).parameters)
    assert params == ["candidate", "test_labels", "test_errors"]
    # No default candidate - a caller MUST pass an already-selected ThresholdCandidate explicitly.
    assert inspect.signature(evaluate_candidate_on_final_test).parameters["candidate"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# 6: Model artifact paths are unique
# ---------------------------------------------------------------------------

def test_model_paths_are_unique_per_candidate():
    from app.ai.training.artifacts import get_model_path

    ids = [c.candidate_id for c in stage1_candidates()] + [c.candidate_id for c in stage2_candidates((160, 160))]
    paths = {get_model_path("bottle", model_name=f"phase4_optimization/{cid}/autoencoder") for cid in ids}
    assert len(paths) == len(ids)


# ---------------------------------------------------------------------------
# 7: Phase 3 model artifact remains unchanged (guard, skipped if not present)
# ---------------------------------------------------------------------------

def test_phase3_artifact_hash_unaffected_by_phase4_test_suite():
    from app.ai.evaluation.model_metadata import compute_model_metadata
    from app.ai.training.artifacts import get_model_path

    phase3_path = get_model_path("bottle", model_name="phase3_validation/autoencoder")
    if not phase3_path.is_file():
        pytest.skip("Phase 3 model artifact not present in this environment")

    metadata = compute_model_metadata(phase3_path, category="bottle")
    assert metadata.md5 == "76478dd6996feb50fadaf5e5e5be1ae4"


# ---------------------------------------------------------------------------
# 8: Candidate results are deterministic
# ---------------------------------------------------------------------------

def test_run_candidate_is_deterministic_across_two_calls(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 16)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)
    split = split_train_validation(samples, training_fraction=0.75, seed=42)

    from app.ai.training.schemas import TrainingConfig
    config = CandidateConfig(
        candidate_id="candidate_test", stage="test",
        training_config=TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=2, seed=42),
        notes="test",
    )

    result_a = run_candidate(config, split.training, split.validation, tmp_path / "a" / "m.pt")
    result_b = run_candidate(config, split.training, split.validation, tmp_path / "b" / "m.pt")

    assert result_a.model_metadata.md5 == result_b.model_metadata.md5
    assert result_a.validation_errors == result_b.validation_errors
    assert result_a.selection.status == result_b.selection.status


# ---------------------------------------------------------------------------
# 9: Benchmark timing calculations are correct
# ---------------------------------------------------------------------------

def test_summarize_timings_known_values():
    summary = summarize_timings([10.0, 20.0, 30.0])
    assert isinstance(summary, TimingSummary)
    assert summary.mean == pytest.approx(20.0)
    assert summary.median == pytest.approx(20.0)
    assert summary.min == 10.0
    assert summary.max == 30.0
    assert summary.count == 3


def test_summarize_timings_raises_on_empty():
    with pytest.raises(ValueError):
        summarize_timings([])


def test_summarize_timings_single_value():
    summary = summarize_timings([5.0])
    assert summary.mean == summary.median == summary.min == summary.max == 5.0
    assert summary.count == 1


# ---------------------------------------------------------------------------
# Stage selection (validation-only ranking)
# ---------------------------------------------------------------------------

def _fake_candidate_result(candidate_id, mean, std, epochs=15, size_bytes=1000, val_ms=10.0, status="SELECTED"):
    from app.ai.evaluation.threshold_experiments import ThresholdCandidate
    from app.ai.evaluation.phase3_threshold_selection import Phase3SelectionResult
    from app.ai.evaluation.model_metadata import ModelMetadata
    from app.ai.training.schemas import TrainingConfig

    threshold_candidate = ThresholdCandidate("mean_std", 1.5, threshold=mean + 1.5 * std,
                                              calibration_sample_count=10, calibration_mean=mean,
                                              calibration_std=std, calibration_percentile=None)
    selection = Phase3SelectionResult(
        status=status, selected=threshold_candidate if status != STATUS_NO_DEFENSIBLE_WINNER else None,
        reasoning="test", stability_by_candidate=[],
    )
    config = CandidateConfig(
        candidate_id=candidate_id, stage="test",
        training_config=TrainingConfig(category="widget", image_size=(32, 32), epochs=epochs, seed=42),
        notes="test",
    )
    metadata = ModelMetadata(category="widget", model_name=candidate_id, architecture="conv_autoencoder",
                              latent_channels=128, artifact_path="x", artifact_size_bytes=size_bytes,
                              sha256="x", md5="x")
    return CandidateResult(
        config=config, model_path=Path("x"), model=None, model_metadata=metadata,
        training_duration_s=1.0, final_training_loss=0.01,
        validation_errors=[mean] * 10, validation_mean=mean, validation_std=std,
        validation_min=mean - std, validation_max=mean + std,
        threshold_candidates=[threshold_candidate], selection=selection, validation_inference_ms=val_ms,
    )


def test_select_stage_winner_picks_lowest_coefficient_of_variation():
    tight = _fake_candidate_result("tight", mean=0.01, std=0.0005)   # CoV = 0.05
    loose = _fake_candidate_result("loose", mean=0.01, std=0.003)    # CoV = 0.3

    result = select_stage_winner([tight, loose])

    assert result.winner.config.candidate_id == "tight"


def test_select_stage_winner_excludes_no_defensible_winner_candidates():
    ok = _fake_candidate_result("ok", mean=0.01, std=0.001)
    bad = _fake_candidate_result("bad", mean=0.01, std=0.0001, status=STATUS_NO_DEFENSIBLE_WINNER)

    result = select_stage_winner([bad, ok])

    assert result.winner.config.candidate_id == "ok"


def test_select_stage_winner_raises_when_all_excluded():
    bad = _fake_candidate_result("bad", mean=0.01, std=0.0001, status=STATUS_NO_DEFENSIBLE_WINNER)
    with pytest.raises(ValueError):
        select_stage_winner([bad])


def test_select_stage_winner_tie_break_prefers_fewer_epochs():
    a = _fake_candidate_result("a_more_epochs", mean=0.01, std=0.001, epochs=35)
    b = _fake_candidate_result("b_fewer_epochs", mean=0.01, std=0.001, epochs=15)  # same CoV -> tie

    result = select_stage_winner([a, b])

    assert result.winner.config.candidate_id == "b_fewer_epochs"
    assert abs(result.ranked[0][1] - result.ranked[1][1]) <= TIE_TOLERANCE * result.ranked[0][1] + 1e-12


# ---------------------------------------------------------------------------
# 10: Existing Phase 1-3 functionality remains unaffected
# ---------------------------------------------------------------------------

def test_phase3_threshold_selection_still_importable_and_unaffected():
    from app.ai.evaluation.phase3_threshold_selection import STATUS_SELECTED, select_threshold_from_validation

    assert STATUS_SELECTED == "SELECTED"
    assert callable(select_threshold_from_validation)


def test_phase4_report_path_distinct_from_all_prior_phase_reports():
    from app.ai.evaluation.phase2_report import default_phase2_report_path
    from app.ai.evaluation.phase3_report import default_phase3_report_path
    from app.ai.evaluation.phase4_report import default_phase4_report_path
    from app.ai.evaluation.report import default_report_path

    paths = {
        default_report_path("bottle"), default_phase2_report_path("bottle"),
        default_phase3_report_path("bottle"), default_phase4_report_path("bottle"),
    }
    assert len(paths) == 4
