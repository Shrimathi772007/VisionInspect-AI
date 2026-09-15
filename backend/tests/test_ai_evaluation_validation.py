"""Milestone 4 Phase 1: tests for the leakage-free, reproducible evaluation harness.

Covers: deterministic calibration/held-out splitting, threshold calibration
never touching the final test set, model artifact metadata/hashing,
reproducibility of the validated evaluation, and report build/save.

Does not modify or duplicate the existing evaluate_model tests
(test_ai_evaluation.py) - those keep covering the original, unchanged
methodology; this file covers only the new Phase 1 additions.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from app.ai.evaluation import (
    CalibrationSplit,
    ValidatedEvaluationResult,
    build_phase1_report,
    compute_model_metadata,
    evaluate_model_validated,
    save_report,
    split_for_calibration,
)
from app.ai.training import build_model, get_model_path, save_model
from app.ai.training.schemas import DatasetSample, DatasetStatistics
from tests.conftest import make_image_bytes

CATEGORY = "bottle"


def _fake_samples(n: int, category: str = "widget") -> list[DatasetSample]:
    return [
        DatasetSample(path=Path(f"/fake/{category}/{i:03d}.png"), category=category, split="train", defect_type="good", label=0)
        for i in range(n)
    ]


def _make_fake_category(dataset_root, category, good_count, defect_counts):
    good_dir = dataset_root / category / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(good_count):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=(i * 5, 40, 90)))

    for defect_type, count in defect_counts.items():
        d = dataset_root / category / "test" / defect_type
        d.mkdir(parents=True)
        for i in range(count):
            color = (10, 200, 10) if defect_type == "good" else (200, 10, 10)
            (d / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=color))


# ---------------------------------------------------------------------------
# Calibration split
# ---------------------------------------------------------------------------

def test_split_partitions_every_sample_exactly_once():
    samples = _fake_samples(20)
    split = split_for_calibration(samples, fraction=0.8, seed=42)

    assert set(split.calibration) & set(split.held_out) == set()
    assert set(split.calibration) | set(split.held_out) == set(samples)
    assert len(split.calibration) + len(split.held_out) == len(samples)


def test_split_respects_fraction():
    samples = _fake_samples(20)
    split = split_for_calibration(samples, fraction=0.8, seed=42)
    assert len(split.calibration) == 16
    assert len(split.held_out) == 4


def test_split_is_deterministic_for_same_seed():
    samples = _fake_samples(20)
    split_a = split_for_calibration(samples, fraction=0.8, seed=42)
    split_b = split_for_calibration(samples, fraction=0.8, seed=42)
    assert [s.path for s in split_a.calibration] == [s.path for s in split_b.calibration]
    assert [s.path for s in split_a.held_out] == [s.path for s in split_b.held_out]


def test_split_differs_across_seeds():
    samples = _fake_samples(20)
    split_a = split_for_calibration(samples, fraction=0.8, seed=42)
    split_b = split_for_calibration(samples, fraction=0.8, seed=7)
    assert [s.path for s in split_a.calibration] != [s.path for s in split_b.calibration]


def test_split_never_leaves_either_subset_empty_even_with_extreme_fraction():
    samples = _fake_samples(2)
    split = split_for_calibration(samples, fraction=0.99, seed=42)
    assert len(split.calibration) >= 1
    assert len(split.held_out) >= 1


def test_split_raises_on_too_few_samples():
    with pytest.raises(ValueError):
        split_for_calibration(_fake_samples(1), fraction=0.8, seed=42)
    with pytest.raises(ValueError):
        split_for_calibration([], fraction=0.8, seed=42)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_raises_on_invalid_fraction(bad_fraction):
    with pytest.raises(ValueError):
        split_for_calibration(_fake_samples(10), fraction=bad_fraction, seed=42)


def test_split_returns_calibration_split_dataclass():
    split = split_for_calibration(_fake_samples(10), fraction=0.8, seed=42)
    assert isinstance(split, CalibrationSplit)
    assert split.fraction == 0.8
    assert split.seed == 42


# ---------------------------------------------------------------------------
# Model metadata / hashing
# ---------------------------------------------------------------------------

def test_compute_model_metadata_reads_real_hash(tmp_path):
    model = build_model()
    model_path = tmp_path / "widget" / "autoencoder.pt"
    save_model(model, model_path)

    expected_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_md5 = hashlib.md5(model_path.read_bytes()).hexdigest()

    metadata = compute_model_metadata(model_path, category="widget")

    assert metadata.sha256 == expected_sha256
    assert metadata.md5 == expected_md5
    assert metadata.artifact_size_bytes == model_path.stat().st_size
    assert metadata.category == "widget"
    assert metadata.architecture == "conv_autoencoder"
    assert metadata.torch_version  # a real, non-empty version string


def test_compute_model_metadata_raises_for_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_model_metadata(tmp_path / "does_not_exist.pt", category="widget")


def test_real_bottle_artifact_hash_matches_known_reference():
    """Regression guard: the currently-shipped bottle model artifact's MD5 must
    keep matching the historical reference hash recorded when the Milestone 2
    baseline (threshold ~0.003212, recall 46.03%) was produced. If this ever
    fails, the artifact was replaced/retrained and the historical baseline
    numbers must be re-verified, not assumed."""
    model_path = get_model_path(CATEGORY)
    if not model_path.is_file():
        pytest.skip("bottle model artifact not present in this environment")

    metadata = compute_model_metadata(model_path, category=CATEGORY)
    assert metadata.md5 == "2f470c30834baf80252f1d352c7fb37f"


# ---------------------------------------------------------------------------
# Validated evaluation - leakage-free calibration + reproducibility
# ---------------------------------------------------------------------------

def test_validated_evaluation_threshold_uses_only_calibration_subset(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=10, defect_counts={"good": 3, "broken": 3})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    from app.ai.evaluation.threshold import compute_threshold
    from app.ai.evaluation.validated_evaluate import _errors_for
    from app.ai.training.dataset import discover_train_samples

    model = build_model()
    model.eval()

    train_samples = discover_train_samples("widget")
    split = split_for_calibration(train_samples, fraction=0.8, seed=42)
    expected_threshold = compute_threshold(_errors_for(model, split.calibration, (32, 32)), k=3.0)

    result = evaluate_model_validated("widget", model, image_size=(32, 32), calibration_fraction=0.8, calibration_seed=42)

    assert result.threshold == pytest.approx(expected_threshold)
    assert result.calibration.calibration_sample_count == len(split.calibration)
    assert result.calibration.held_out_sample_count == len(split.held_out)


def test_validated_evaluation_final_test_set_is_full_and_untouched_by_calibration(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=10, defect_counts={"good": 4, "broken": 5})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    result = evaluate_model_validated("widget", model, image_size=(32, 32))

    assert result.total_test_samples == 9  # 4 good + 5 broken, none held back for calibration
    assert result.good_test_count == 4
    assert result.defective_test_count == 5
    assert sum(sum(row) for row in result.confusion_matrix) == result.total_test_samples


def test_validated_evaluation_is_deterministic_across_runs(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=10, defect_counts={"good": 2, "broken": 2})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    model.eval()

    result_a = evaluate_model_validated("widget", model, image_size=(32, 32))
    result_b = evaluate_model_validated("widget", model, image_size=(32, 32))

    assert result_a.threshold == result_b.threshold
    assert result_a.calibration.calibration_sample_count == result_b.calibration.calibration_sample_count
    assert [s.reconstruction_error for s in result_a.samples] == [s.reconstruction_error for s in result_b.samples]
    assert [s.predicted_label for s in result_a.samples] == [s.predicted_label for s in result_b.samples]


def test_validated_evaluation_raises_when_no_test_images(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    good_dir = fake_root / "widget" / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(5):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32)))
    (fake_root / "widget" / "test").mkdir(parents=True)
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    with pytest.raises(ValueError):
        evaluate_model_validated("widget", model, image_size=(32, 32))


def test_validated_evaluation_propagates_calibration_split_error_for_tiny_train_set(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=1, defect_counts={"good": 1})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    with pytest.raises(ValueError):
        evaluate_model_validated("widget", model, image_size=(32, 32))


def test_validated_evaluation_timing_is_measured(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=10, defect_counts={"good": 2, "broken": 2})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    result = evaluate_model_validated("widget", model, image_size=(32, 32), model_load_ms=12.5)

    assert result.timing is not None
    assert result.timing.model_load_ms == 12.5
    assert result.timing.final_test_sample_count == 4
    assert result.timing.mean_preprocessing_ms_per_image >= 0
    assert result.timing.mean_inference_ms_per_image >= 0
    assert math.isfinite(result.timing.mean_total_ms_per_image)


def test_validated_evaluation_result_type():
    # Cheap type-contract check without needing a dataset fixture.
    assert hasattr(ValidatedEvaluationResult, "confusion_matrix")


# ---------------------------------------------------------------------------
# Report build/save
# ---------------------------------------------------------------------------

def test_build_and_save_report_round_trips(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=10, defect_counts={"good": 2, "broken": 2})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    model_path = tmp_path / "ai_models" / "widget" / "autoencoder.pt"
    save_model(model, model_path)

    from app.ai.evaluation.evaluate import evaluate_model

    baseline = evaluate_model("widget", model, image_size=(32, 32))
    validated = evaluate_model_validated("widget", model, image_size=(32, 32))
    metadata = compute_model_metadata(model_path, category="widget", latent_channels=model.latent_channels)
    stats = DatasetStatistics(
        category="widget",
        train_good_count=10,
        test_good_count=2,
        test_defective_count=2,
        test_total_count=4,
        test_defect_type_counts={"good": 2, "broken": 2},
    )

    report = build_phase1_report(
        model_metadata=metadata,
        dataset_stats=stats,
        training_seed=42,
        baseline=baseline,
        validated=validated,
    )

    report_path = save_report(report, tmp_path / "reports" / "phase1_validation_report.json")
    assert report_path.is_file()

    reloaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert reloaded["report_type"] == "milestone_4_phase_1_ai_evaluation"
    assert reloaded["model"]["sha256"] == metadata.sha256
    assert reloaded["dataset"]["train_good_count"] == 10
    assert reloaded["baseline_reproduction"]["confusion_matrix"] == baseline.confusion_matrix
    assert reloaded["validated_phase1_evaluation"]["confusion_matrix"] == validated.confusion_matrix
    assert reloaded["validated_phase1_evaluation"]["calibration"]["calibration_seed"] == 42
    assert "timing_ms" in reloaded["validated_phase1_evaluation"]


def test_save_report_creates_parent_directories(tmp_path):
    report = {"report_type": "milestone_4_phase_1_ai_evaluation"}
    nested_path = tmp_path / "a" / "b" / "c" / "report.json"

    saved_path = save_report(report, nested_path)

    assert saved_path == nested_path
    assert json.loads(nested_path.read_text(encoding="utf-8")) == report
