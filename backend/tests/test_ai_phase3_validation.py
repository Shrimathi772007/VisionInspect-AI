"""Milestone 4 Phase 3: tests for the true train/validation split, subset
training, genuine-validation threshold selection, and leakage protection.

Does not modify or duplicate Phase 1 (test_ai_evaluation_validation.py) or
Phase 2 (test_ai_threshold_experiments.py) tests - those keep covering the
unchanged Phase 1/2 code; this file covers only the new Phase 3 additions.

Uses small, fast synthetic datasets/models throughout (not the real 209/83
bottle dataset) so this suite runs quickly - the real dataset is exercised
once, deliberately, by scripts/train_and_evaluate_bottle_phase3.py, not by
the automated test suite.
"""

import hashlib
import math
from pathlib import Path

import pytest
import torch

from app.ai.evaluation.model_metadata import compute_model_metadata
from app.ai.evaluation.phase3_threshold_selection import (
    STATUS_NO_DEFENSIBLE_WINNER,
    STATUS_SELECTED,
    select_threshold_from_validation,
)
from app.ai.evaluation.threshold_experiments import (
    ThresholdCandidate,
    compute_errors_for_samples,
    evaluate_candidate_on_final_test,
    generate_candidates,
)
from app.ai.training import build_model
from app.ai.training.phase3_dataset import GoodImageSubsetDataset
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.schemas import DatasetSample, TrainingConfig
from app.ai.training.validation_split import DEFAULT_SPLIT_SEED, split_train_validation
from tests.conftest import make_image_bytes

CATEGORY = "widget"


def _write_fake_train_good(dataset_root: Path, category: str, count: int) -> None:
    good_dir = dataset_root / category / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(count):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=(i % 250, 40, 90)))


def _make_fake_test(dataset_root: Path, category: str, defect_counts: dict[str, int]) -> None:
    for defect_type, count in defect_counts.items():
        d = dataset_root / category / "test" / defect_type
        d.mkdir(parents=True)
        for i in range(count):
            color = (10, 200, 10) if defect_type == "good" else (200, 10, 10)
            (d / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=color))


# ---------------------------------------------------------------------------
# 1-3: Deterministic split, correct sizes, no train/validation overlap
# ---------------------------------------------------------------------------

def test_split_is_deterministic_and_sized_correctly(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 20)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split_a = split_train_validation(samples, training_fraction=0.8, seed=42)
    split_b = split_train_validation(samples, training_fraction=0.8, seed=42)

    assert len(split_a.training) == 16
    assert len(split_a.validation) == 4
    assert [s.path for s in split_a.training] == [s.path for s in split_b.training]
    assert [s.path for s in split_a.validation] == [s.path for s in split_b.validation]
    assert split_a.seed == DEFAULT_SPLIT_SEED == 42
    assert split_a.training_fraction == 0.8


def test_no_overlap_between_training_and_validation(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 30)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.8, seed=42)
    training_paths = {s.path for s in split.training}
    validation_paths = {s.path for s in split.validation}

    assert training_paths & validation_paths == set()
    assert training_paths | validation_paths == {s.path for s in samples}


# ---------------------------------------------------------------------------
# 4: No overlap between train/validation and the final test set
# ---------------------------------------------------------------------------

def test_no_overlap_between_train_validation_and_final_test(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 20)
    _make_fake_test(fake_root, CATEGORY, {"good": 3, "broken": 3})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    from app.ai.training.dataset import discover_test_samples, discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.8, seed=42)
    test_samples = discover_test_samples(CATEGORY)

    train_val_paths = {s.path for s in split.training} | {s.path for s in split.validation}
    test_paths = {s.path for s in test_samples}

    assert train_val_paths & test_paths == set()


# ---------------------------------------------------------------------------
# 5: Validation images are excluded from model training
# ---------------------------------------------------------------------------

def test_validation_samples_never_appear_in_the_training_dataset(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 20)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.8, seed=42)
    dataset = GoodImageSubsetDataset(split.training, image_size=(32, 32))

    dataset_paths = {s.path for s in dataset.samples}
    validation_paths = {s.path for s in split.validation}

    assert dataset_paths == {s.path for s in split.training}
    assert dataset_paths & validation_paths == set()
    assert len(dataset) == len(split.training)


def test_good_image_subset_dataset_raises_on_empty_samples():
    with pytest.raises(ValueError):
        GoodImageSubsetDataset([], image_size=(32, 32))


def test_train_anomaly_model_on_samples_only_uses_given_samples(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 12)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.75, seed=42)
    config = TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, seed=42)
    model_path = tmp_path / "ai_models" / CATEGORY / "phase3" / "autoencoder.pt"

    result, model = train_anomaly_model_on_samples(split.training, config, model_path)

    assert result.num_training_images == len(split.training)
    assert model_path.is_file()
    assert not model.training  # returned in eval() mode
    assert math.isfinite(result.final_loss)


# ---------------------------------------------------------------------------
# 6: Correct validation reconstruction-error calculation
# ---------------------------------------------------------------------------

def test_validation_reconstruction_errors_computed_with_trained_model(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 12)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.75, seed=42)
    config = TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, seed=42)
    model_path = tmp_path / "ai_models" / CATEGORY / "phase3" / "autoencoder.pt"
    _, model = train_anomaly_model_on_samples(split.training, config, model_path)

    validation_errors = compute_errors_for_samples(model, split.validation, image_size=(32, 32))

    assert len(validation_errors) == len(split.validation)
    assert all(math.isfinite(e) and e >= 0 for e in validation_errors)


# ---------------------------------------------------------------------------
# 7-8: Threshold calculation from validation data + candidate generation
# ---------------------------------------------------------------------------

def test_candidates_generated_from_validation_errors_only():
    validation_errors = [0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0100]
    candidates = generate_candidates(validation_errors)
    assert len(candidates) == 9
    assert all(c.calibration_sample_count == len(validation_errors) for c in candidates)


# ---------------------------------------------------------------------------
# 9: Deterministic threshold selection
# ---------------------------------------------------------------------------

def test_selection_from_validation_is_deterministic():
    validation_errors = [0.002, 0.0021, 0.0022, 0.0099, 0.0023]
    candidates = generate_candidates(validation_errors)

    result_a = select_threshold_from_validation(candidates, validation_errors)
    result_b = select_threshold_from_validation(candidates, validation_errors)

    assert result_a.status == result_b.status
    if result_a.selected is not None:
        assert result_a.selected.threshold == result_b.selected.threshold


def test_selection_labeled_selected_not_provisional_when_a_candidate_passes():
    validation_errors = [0.001] * 40 + [0.02] * 2  # 2/42 = ~4.8% FP rate at a low threshold
    candidates = generate_candidates(validation_errors)

    result = select_threshold_from_validation(candidates, validation_errors)

    assert result.status == STATUS_SELECTED
    assert result.status != "PROVISIONAL"
    assert result.selected is not None


def test_selection_no_defensible_winner_when_all_candidates_unstable():
    validation_errors = [0.01] * 20  # every candidate threshold will be exceeded by most of these
    candidate = ThresholdCandidate("mean_std", 100.0, threshold=0.0001, calibration_sample_count=20,
                                    calibration_mean=0.01, calibration_std=0.0, calibration_percentile=None)

    result = select_threshold_from_validation([candidate], validation_errors, max_validation_false_positive_rate=0.10)

    assert result.status == STATUS_NO_DEFENSIBLE_WINNER
    assert result.selected is None


# ---------------------------------------------------------------------------
# 10: Final-test data cannot reach threshold-selection code
# ---------------------------------------------------------------------------

def test_select_threshold_from_validation_has_no_test_data_parameter():
    import inspect

    params = list(inspect.signature(select_threshold_from_validation).parameters)
    assert params == ["candidates", "validation_errors", "max_validation_false_positive_rate"]
    assert "test_errors" not in params
    assert "test_labels" not in params


# ---------------------------------------------------------------------------
# 11: Correct metrics (final-test scoring reuses the Phase 2 function exactly)
# ---------------------------------------------------------------------------

def test_final_test_scoring_known_case():
    candidate = ThresholdCandidate("mean_std", 3.0, threshold=0.05, calibration_sample_count=10,
                                    calibration_mean=0.02, calibration_std=0.01, calibration_percentile=None)
    test_labels = [0, 0, 0, 1, 1, 1]
    test_errors = [0.01, 0.02, 0.06, 0.03, 0.07, 0.08]  # one false positive, one false negative

    result = evaluate_candidate_on_final_test(candidate, test_labels, test_errors)

    assert result.true_negatives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.true_positives == 2
    assert result.accuracy == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# 12: Model artifact integrity (Phase 3 model has its own, different hash)
# ---------------------------------------------------------------------------

def test_phase3_model_hash_differs_from_a_differently_trained_model(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 16)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.75, seed=42)
    config_a = TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, seed=42)
    config_b = TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=1, seed=7)

    path_a = tmp_path / "a" / "autoencoder.pt"
    path_b = tmp_path / "b" / "autoencoder.pt"
    train_anomaly_model_on_samples(split.training, config_a, path_a)
    train_anomaly_model_on_samples(split.training, config_b, path_b)

    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    assert md5(path_a) != md5(path_b)  # different seeds -> different weights -> different hash

    metadata_a = compute_model_metadata(path_a, category=CATEGORY)
    assert metadata_a.md5 == md5(path_a)


def test_same_seed_and_data_produce_identical_model_hash(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _write_fake_train_good(fake_root, CATEGORY, 16)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)
    from app.ai.training.dataset import discover_train_samples
    samples = discover_train_samples(CATEGORY)

    split = split_train_validation(samples, training_fraction=0.75, seed=42)
    config = TrainingConfig(category=CATEGORY, image_size=(32, 32), batch_size=2, epochs=2, seed=42)

    path_a = tmp_path / "run1" / "autoencoder.pt"
    path_b = tmp_path / "run2" / "autoencoder.pt"
    train_anomaly_model_on_samples(split.training, config, path_a)
    train_anomaly_model_on_samples(split.training, config, path_b)

    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    assert md5(path_a) == md5(path_b)  # same seed, same data, same config -> reproducible weights


# ---------------------------------------------------------------------------
# 13-14: Phase 1 and Phase 2 functionality remain unchanged
# ---------------------------------------------------------------------------

def test_phase1_evaluate_model_validated_still_importable_and_unaffected():
    from app.ai.evaluation import DEFAULT_CALIBRATION_FRACTION, DEFAULT_CALIBRATION_SEED, evaluate_model_validated

    assert DEFAULT_CALIBRATION_FRACTION == 0.8
    assert DEFAULT_CALIBRATION_SEED == 42
    assert callable(evaluate_model_validated)


def test_phase2_threshold_selection_still_labeled_provisional():
    from app.ai.evaluation.threshold_selection import STATUS_PROVISIONAL, select_threshold

    assert STATUS_PROVISIONAL == "PROVISIONAL - NOT VALIDATED"
    assert callable(select_threshold)


def test_phase2_and_phase3_report_paths_are_all_distinct():
    from app.ai.evaluation.phase2_report import default_phase2_report_path
    from app.ai.evaluation.phase3_report import default_phase3_report_path
    from app.ai.evaluation.report import default_report_path

    paths = {
        default_report_path("bottle"),
        default_phase2_report_path("bottle"),
        default_phase3_report_path("bottle"),
    }
    assert len(paths) == 3  # all three are distinct files - none overwrite each other
