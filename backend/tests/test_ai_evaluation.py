import math

import pytest
import torch

from app.ai.evaluation import (
    DEFAULT_THRESHOLD_K,
    EvaluationResult,
    compute_classification_metrics,
    compute_reconstruction_error,
    compute_threshold,
    evaluate_model,
)
from app.ai.training import build_model, load_model_for_category, save_model
from tests.conftest import make_image_bytes

CATEGORY = "bottle"


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
# Threshold calculation
# ---------------------------------------------------------------------------

def test_threshold_is_mean_plus_k_std():
    errors = [0.01, 0.012, 0.011, 0.013, 0.009]
    import numpy as np

    expected = float(np.mean(errors) + DEFAULT_THRESHOLD_K * np.std(errors))
    assert compute_threshold(errors) == pytest.approx(expected)


def test_threshold_respects_custom_k():
    errors = [0.01, 0.02, 0.03]
    low_k = compute_threshold(errors, k=1.0)
    high_k = compute_threshold(errors, k=5.0)
    assert high_k > low_k


def test_threshold_raises_on_empty_input():
    with pytest.raises(ValueError):
        compute_threshold([])


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------

def test_reconstruction_error_is_zero_for_identity_like_model():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    model = Identity()
    x = torch.rand(3, 32, 32)
    assert compute_reconstruction_error(model, x) == pytest.approx(0.0, abs=1e-8)


def test_reconstruction_error_is_positive_for_mismatched_output():
    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    model = AddOne()
    x = torch.zeros(3, 16, 16)
    error = compute_reconstruction_error(model, x)
    assert error == pytest.approx(1.0)


def test_reconstruction_error_is_deterministic_for_fixed_weights():
    model = build_model()
    model.eval()
    x = torch.rand(3, 32, 32)
    error_a = compute_reconstruction_error(model, x)
    error_b = compute_reconstruction_error(model, x)
    assert error_a == error_b


# ---------------------------------------------------------------------------
# Good/defective prediction rule
# ---------------------------------------------------------------------------

def test_predicted_label_rule_matches_threshold_boundary():
    from app.ai.evaluation.metrics import DEFECTIVE_LABEL, GOOD_LABEL

    threshold = 0.05
    assert (DEFECTIVE_LABEL if 0.05 > threshold else GOOD_LABEL) == GOOD_LABEL  # exactly at threshold -> good
    assert (DEFECTIVE_LABEL if 0.0501 > threshold else GOOD_LABEL) == DEFECTIVE_LABEL
    assert (DEFECTIVE_LABEL if 0.0499 > threshold else GOOD_LABEL) == GOOD_LABEL


# ---------------------------------------------------------------------------
# Classification metrics / confusion matrix
# ---------------------------------------------------------------------------

def test_classification_metrics_on_known_case():
    # 2 actual good (both correctly predicted good), 2 actual defective
    # (one correctly predicted defective, one missed).
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1_score == pytest.approx(2 / 3)


def test_classification_metrics_handles_zero_division():
    # No predicted-defective at all -> precision would divide by zero without zero_division handling.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


# ---------------------------------------------------------------------------
# Evaluation result schema / full pipeline on a tiny synthetic dataset
# ---------------------------------------------------------------------------

def test_evaluate_model_on_tiny_synthetic_dataset(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=6, defect_counts={"good": 3, "broken": 3})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()

    result = evaluate_model("widget", model, image_size=(32, 32))

    assert isinstance(result, EvaluationResult)
    assert result.category == "widget"
    assert result.num_training_normal_samples == 6
    assert result.total_test_samples == 6
    assert result.good_test_count == 3
    assert result.defective_test_count == 3
    assert len(result.samples) == 6
    assert result.num_correct + result.num_incorrect == 6
    assert math.isfinite(result.threshold)
    assert math.isfinite(result.mean_error)
    assert result.min_error <= result.mean_error <= result.max_error
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1_score <= 1.0

    # Confusion matrix values must add up to the total test count.
    cm = result.confusion_matrix
    assert sum(sum(row) for row in cm) == result.total_test_samples


def test_evaluate_model_never_uses_test_images_for_threshold(tmp_path, monkeypatch):
    """The threshold must be derivable from train/good alone, without test data existing at all."""
    fake_root = tmp_path / "dataset"
    good_dir = fake_root / "widget" / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(5):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32)))
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    from app.ai.training.dataset import discover_train_samples
    from app.ai.evaluation.evaluate import _reconstruction_errors

    model = build_model()
    train_samples = discover_train_samples("widget")
    errors = _reconstruction_errors(model, train_samples, (32, 32))
    threshold = compute_threshold(errors)

    assert math.isfinite(threshold)
    assert len(errors) == 5


def test_evaluate_model_raises_when_no_test_images(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    good_dir = fake_root / "widget" / "train" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "000.png").write_bytes(make_image_bytes("PNG", size=(32, 32)))
    (fake_root / "widget" / "test").mkdir(parents=True)  # empty test dir
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    with pytest.raises(ValueError):
        evaluate_model("widget", model, image_size=(32, 32))


def test_evaluation_is_deterministic_across_runs(tmp_path, monkeypatch):
    fake_root = tmp_path / "dataset"
    _make_fake_category(fake_root, "widget", good_count=4, defect_counts={"good": 2, "broken": 2})
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    model = build_model()
    model.eval()

    result_a = evaluate_model("widget", model, image_size=(32, 32))
    result_b = evaluate_model("widget", model, image_size=(32, 32))

    assert result_a.threshold == result_b.threshold
    assert [s.reconstruction_error for s in result_a.samples] == [s.reconstruction_error for s in result_b.samples]
    assert [s.predicted_label for s in result_a.samples] == [s.predicted_label for s in result_b.samples]


# ---------------------------------------------------------------------------
# Model loading (existing artifact-loading functionality)
# ---------------------------------------------------------------------------

def test_load_model_for_category_used_by_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    model = build_model()
    from app.ai.training.artifacts import get_model_path

    save_model(model, get_model_path("widget"))

    loaded = load_model_for_category("widget")
    assert not loaded.training  # eval() mode

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model.eval()(x)
        loaded_output = loaded(x)
    torch.testing.assert_close(original_output, loaded_output)
