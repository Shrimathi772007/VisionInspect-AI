"""The category-parameterized Phase 1 baseline runner (app.ai.evaluation.category_phase1).

Uses a tiny synthetic category (32x32 images, 1 epoch) wired in by monkeypatching the dataset/artifact roots,
like the other AI tests. The real Leather experiment is run by scripts/run_category_phase1.py, not by the suite.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from app.ai.evaluation import category_phase1 as phase1
from app.ai.evaluation.category_phase1 import (
    compare_runs,
    default_model_path,
    describe,
    phase1_config,
    preprocessing_description,
    run_phase1,
    verify_dataset_counts,
)
from app.ai.training.schemas import TrainingConfig
from tests.conftest import make_image_bytes

CATEGORY = "widget"
IMAGE_SIZE = (32, 32)
EXPECTED = {"train_good": 12, "test_good": 4, "test_defective": 6, "test_total": 10}


def _write(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_image_bytes("PNG", size=IMAGE_SIZE, color=color))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for i in range(12):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", (10 + i * 7, 60, 90))
    for i in range(4):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", (12 + i * 11, 61, 91))
    for defect, base in (("scratch", 200), ("dent", 150)):
        for i in range(3):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", (base, 20 + i * 9, 30))
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    monkeypatch.setattr(phase1, "phase1_config", lambda category: TrainingConfig(
        category=category, image_size=IMAGE_SIZE, batch_size=4, epochs=1, device="cpu"))
    return root


def test_established_baseline_configuration():
    config = phase1_config("leather")
    assert (config.image_size, config.batch_size, config.epochs, config.learning_rate, config.seed, config.device) == (
        (128, 128), 8, 15, 1e-3, 42, "cpu")
    assert config.model.latent_channels == 128


def test_default_artifact_path_is_phase1_baseline(synthetic):
    assert default_model_path("leather").as_posix().endswith("ai_models/leather/phase1_baseline/autoencoder.pt")


def test_count_mismatch_raises_before_any_training(synthetic, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("training must not start on a count mismatch")

    monkeypatch.setattr(phase1, "train_anomaly_model_on_samples", boom)
    with pytest.raises(ValueError, match="train_good"):
        run_phase1(CATEGORY, tmp_path / "m" / "autoencoder.pt", {**EXPECTED, "train_good": 245})
    assert not (tmp_path / "m").exists()


def test_verify_counts_reports_actual_numbers(synthetic):
    counts = verify_dataset_counts(CATEGORY, EXPECTED)
    assert counts["train_good"] == 12 and counts["test_by_type"] == {"dent": 3, "good": 4, "scratch": 3}


def test_refuses_to_overwrite_existing_artifact(synthetic, tmp_path):
    target = tmp_path / "out" / "autoencoder.pt"
    target.parent.mkdir()
    target.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        run_phase1(CATEGORY, target, EXPECTED)
    assert target.read_bytes() == b"precious"


def test_full_run_threshold_lock_and_metrics(synthetic, tmp_path):
    lock_path = tmp_path / "out" / "lock.json"
    run = run_phase1(CATEGORY, tmp_path / "out" / "autoencoder.pt", EXPECTED, lock_path=lock_path)

    errors = np.asarray(run["train_errors"])
    assert len(errors) == 12
    assert run["threshold"]["value"] == pytest.approx(float(errors.mean() + 3 * errors.std()), abs=1e-15)
    assert run["threshold"]["k"] == 3.0
    lock = json.loads(lock_path.read_text())
    assert lock["threshold"] == run["threshold"]["value"] and lock["model_sha256"] == run["artifact"]["sha256"]

    m = run["metrics"]
    assert m["true_positives"] + m["false_negatives"] == 6 and m["true_negatives"] + m["false_positives"] == 4
    assert m["defect_identification_accuracy"] == m["recall"]
    for p in run["predictions"]:
        assert p["predicted_label"] == int(p["reconstruction_error"] > run["threshold"]["value"])
    assert sum(v["total"] for v in run["per_defect"].values()) == 6
    assert set(run["per_defect"]) == {"dent", "scratch"}
    assert run["distributions"]["train_good"]["count"] == 12
    assert all(run["leakage_partial"].values()), run["leakage_partial"]
    assert 0.0 <= run["separation"]["auroc_descriptive_only"] <= 1.0


def test_threshold_is_locked_on_disk_before_test_images_are_listed(synthetic, tmp_path, monkeypatch):
    lock_path = tmp_path / "out" / "lock.json"
    seen = {}
    real = phase1.discover_test_samples

    def spy(category):
        seen["lock_existed"] = lock_path.is_file()
        seen["lock"] = json.loads(lock_path.read_text()) if lock_path.is_file() else None
        return real(category)

    monkeypatch.setattr(phase1, "discover_test_samples", spy)
    run = run_phase1(CATEGORY, tmp_path / "out" / "autoencoder.pt", EXPECTED, lock_path=lock_path)
    assert seen["lock_existed"] and seen["lock"]["threshold"] == run["threshold"]["value"]


def test_the_saved_artifact_is_the_one_that_was_scored(synthetic, tmp_path):
    run = run_phase1(CATEGORY, tmp_path / "out" / "autoencoder.pt", EXPECTED)
    assert run["leakage_partial"]["saved_artifact_is_what_was_scored"]
    assert Path(run["artifact"]["path"]).is_file() and run["artifact"]["size_bytes"] > 0


def test_two_identical_runs_are_reproducible(synthetic, tmp_path):
    a = run_phase1(CATEGORY, tmp_path / "a" / "autoencoder.pt", EXPECTED)
    b = run_phase1(CATEGORY, tmp_path / "b" / "autoencoder.pt", EXPECTED)
    comparison = compare_runs(a, b)
    assert comparison["model_sha256_identical"] and comparison["threshold_identical"], comparison
    assert comparison["predicted_labels_identical"] and comparison["confusion_matrix_identical"]
    assert comparison["loss_history_identical"] and comparison["test_errors_identical"]


def test_compare_runs_detects_a_difference(synthetic, tmp_path):
    a = run_phase1(CATEGORY, tmp_path / "a" / "autoencoder.pt", EXPECTED)
    b = json.loads(json.dumps(a))
    b["threshold"]["value"] += 1e-6
    b["predictions"][0]["reconstruction_error"] += 1e-6
    assert not compare_runs(a, b)["threshold_identical"]
    assert not compare_runs(a, b)["test_errors_identical"]


def test_describe_reports_requested_percentiles():
    d = describe(range(101))
    assert d["count"] == 101 and d["median"] == 50 and d["p5"] == 5 and d["p99"] == 99 and d["max"] == 100


def test_preprocessing_description_matches_the_pipeline_source():
    source = (Path(phase1.__file__).parent.parent / "preprocessing" / "pipeline.py").read_text(encoding="utf-8")
    for token in ("COLOR_BGR2RGB", "INTER_AREA", "astype(np.float32) / 255.0", "IMREAD_COLOR"):
        assert token in source
    desc = preprocessing_description((128, 128))
    assert "INTER_AREA" in desc["resize"] and "255" in desc["normalization"] and desc["augmentation"] == "none"


def test_phase1_module_uses_no_other_phase_machinery():
    tree = ast.parse(Path(phase1.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    forbidden = ("threshold_experiments", "phase3_threshold_selection", "calibration", "patch_anomaly", "resnet18",
                 "validation_split", "phase4")
    assert not [m for m in imported if any(f in m for f in forbidden)], imported


def test_serving_registry_has_no_leather():
    serving = Path(phase1.__file__).parent.parent / "inference" / "serving.py"
    assert "leather" not in serving.read_text(encoding="utf-8").lower()
