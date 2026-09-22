"""Pill Phase 1 ConvAutoencoder baseline (app.ai.evaluation.category_phase1 applied to `pill`).

Real-dataset tests (skipped when the Git-ignored MVTec files are absent) verify discovery, counts, ordering and
preprocessing on the actual dataset without training. Everything that trains uses a tiny synthetic `pill` world
(32x32 images, 1 epoch) wired in by monkeypatching the dataset/artifact roots; the real experiment is run by
scripts/run_category_phase1.py, not by the suite.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from app.ai.evaluation import category_phase1 as phase1
from app.ai.evaluation.category_phase1 import (
    compare_runs,
    dataset_fingerprint,
    default_model_path,
    describe,
    gate_classification,
    phase1_config,
    preprocessing_description,
    run_phase1,
    verify_dataset_counts,
)
from app.ai.evaluation.category_phase3 import defect_type_recall, metrics_from_predictions
from app.ai.evaluation.evaluate import _sample_to_tensor, compute_reconstruction_error
from app.ai.evaluation.threshold import compute_threshold
from app.ai.training import artifacts, discover_test_samples, discover_train_samples
from app.ai.training.dataset import DATASET_ROOT
from app.ai.training.model import ConvAutoencoder, build_model
from app.ai.training.schemas import TrainingConfig
from tests.conftest import make_image_bytes

CATEGORY = "pill"
REAL_EXPECTED = {"train_good": 267, "test_good": 26, "test_defective": 141, "test_total": 167}
REAL_TYPES = {"color": 25, "combined": 17, "contamination": 21, "crack": 26, "faulty_imprint": 19, "good": 26, "pill_type": 9, "scratch": 24}
DEFECT_NAMES = ("color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch")
needs_dataset = pytest.mark.skipif(not (DATASET_ROOT / CATEGORY / "train" / "good").is_dir(), reason="MVTec pill not present")

TINY = (32, 32)
TINY_EXPECTED = {"train_good": 12, "test_good": 4, "test_defective": 14, "test_total": 18}
TINY_TYPES = {"good": 4, **{name: 2 for name in DEFECT_NAMES}}


# ---------------------------------------------------------------------------
# Real dataset: discovery, counts, ordering, preprocessing (no training, no test decoding)
# ---------------------------------------------------------------------------

@needs_dataset
def test_real_dataset_counts_match_the_expected_267_26_141_167():
    counts = verify_dataset_counts(CATEGORY, REAL_EXPECTED)
    assert (counts["train_good"], counts["test_good"], counts["test_defective"], counts["test_total"]) == (267, 26, 141, 167)


@needs_dataset
def test_count_mismatch_is_rejected_loudly():
    with pytest.raises(ValueError, match="train_good"):
        verify_dataset_counts(CATEGORY, {**REAL_EXPECTED, "train_good": 268})
    with pytest.raises(ValueError, match="test_defective"):
        verify_dataset_counts(CATEGORY, {**REAL_EXPECTED, "test_defective": 140})


@needs_dataset
def test_defect_types_are_discovered_from_the_dataset_and_none_is_missing_or_unexpected():
    counts = verify_dataset_counts(CATEGORY, REAL_EXPECTED)
    on_disk = {p.name for p in (DATASET_ROOT / CATEGORY / "test").iterdir() if p.is_dir()}
    assert set(counts["test_by_type"]) == on_disk  # discovered, not hard-coded
    assert on_disk - {"good"} == set(DEFECT_NAMES)
    assert counts["test_by_type"] == REAL_TYPES and sum(v for k, v in REAL_TYPES.items() if k != "good") == 141
    samples = discover_test_samples(CATEGORY)
    assert {s.defect_type for s in samples if s.label == 1} == set(DEFECT_NAMES)
    assert all(s.label == (0 if s.defect_type == "good" else 1) for s in samples)


@needs_dataset
def test_training_discovery_is_train_good_only_sorted_and_deterministic():
    a, b = discover_train_samples(CATEGORY), discover_train_samples(CATEGORY)
    assert len(a) == 267 and [s.path for s in a] == [s.path for s in b]
    assert [s.path.name for s in a] == sorted(s.path.name for s in a)
    assert all(s.split == "train" and s.defect_type == "good" and s.label == 0 and s.path.parent == DATASET_ROOT / CATEGORY / "train" / "good" for s in a)
    order = hashlib.sha256("\n".join(s.path.name for s in a).encode()).hexdigest()
    assert order == hashlib.sha256("\n".join(s.path.name for s in b).encode()).hexdigest()


@needs_dataset
def test_real_train_and_test_filenames_and_contents_do_not_overlap_by_path():
    train = {s.path for s in discover_train_samples(CATEGORY)}
    test = {s.path for s in discover_test_samples(CATEGORY)}
    assert not (train & test) and all("train" in p.parts and "test" not in p.parts for p in train)
    assert not any("ground_truth" in str(p) for p in train | test)  # masks are never part of any sample list


@needs_dataset
def test_real_image_preprocessing_matches_the_documented_pipeline_and_leaves_the_file_untouched():
    sample = discover_train_samples(CATEGORY)[0]  # a TRAIN image: no protected test image is decoded here
    before = hashlib.sha256(sample.path.read_bytes()).hexdigest()
    tensor = _sample_to_tensor(sample, (128, 128))
    expected = cv2.resize(cv2.cvtColor(cv2.imread(str(sample.path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA)
    assert tensor.shape == (3, 128, 128) and tensor.dtype == torch.float32 and tensor.is_contiguous()
    assert torch.equal(tensor, torch.from_numpy(expected.astype(np.float32) / 255.0).permute(2, 0, 1))
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert hashlib.sha256(sample.path.read_bytes()).hexdigest() == before


def test_preprocessing_description_matches_the_pipeline_source():
    source = (Path(phase1.__file__).parent.parent / "preprocessing" / "pipeline.py").read_text(encoding="utf-8")
    for token in ("COLOR_BGR2RGB", "INTER_AREA", "astype(np.float32) / 255.0", "IMREAD_COLOR"):
        assert token in source
    d = preprocessing_description((128, 128))
    assert "128x128" in d["resize"] and d["augmentation"] == "none" and "255" in d["normalization"] and "INTER_AREA" in d["resize"]


# ---------------------------------------------------------------------------
# Architecture and configuration
# ---------------------------------------------------------------------------

def test_architecture_is_the_established_convautoencoder():
    model = build_model()
    assert isinstance(model, ConvAutoencoder)
    enc = [m for m in model.encoder if isinstance(m, nn.Conv2d)]
    dec = [m for m in model.decoder if isinstance(m, nn.ConvTranspose2d)]
    assert [(c.in_channels, c.out_channels) for c in enc] == [(3, 16), (16, 32), (32, 64), (64, 128)]
    assert [(c.in_channels, c.out_channels) for c in dec] == [(128, 64), (64, 32), (32, 16), (16, 3)]
    assert all(c.kernel_size == (4, 4) and c.stride == (2, 2) and c.padding == (1, 1) for c in enc + dec)
    assert sum(isinstance(m, nn.ReLU) for m in model.encoder) == 4 and sum(isinstance(m, nn.ReLU) for m in model.decoder) == 3
    assert isinstance(model.decoder[-1], nn.Sigmoid)
    assert not any(isinstance(m, nn.Linear) for m in model.modules())  # fully convolutional
    assert sum(p.numel() for p in model.parameters()) == 345_955
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out.shape == (2, 3, 128, 128) and 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_training_configuration_is_the_established_phase1_baseline():
    c = phase1_config(CATEGORY)
    assert c.category == CATEGORY and c.image_size == (128, 128) and c.batch_size == 8 and c.epochs == 15
    assert c.learning_rate == 1e-3 and c.seed == 42 and c.device == "cpu" and c.model.latent_channels == 128


# ---------------------------------------------------------------------------
# Threshold, scoring, metrics
# ---------------------------------------------------------------------------

def test_threshold_is_mean_plus_three_population_std_of_the_given_errors_only():
    errors = list(np.random.default_rng(0).lognormal(-7, 0.3, 267))
    arr = np.asarray(errors)
    assert compute_threshold(errors, k=3.0) == pytest.approx(float(arr.mean() + 3 * arr.std(ddof=0)), abs=1e-15)
    assert compute_threshold(errors, k=3.0) != pytest.approx(float(arr.mean() + 3 * arr.std(ddof=1)), abs=1e-15)  # population, not sample
    import inspect

    assert list(inspect.signature(compute_threshold).parameters) == ["normal_errors", "k"]  # no parameter for test data


def test_reconstruction_error_is_the_mean_squared_error():
    torch.manual_seed(0)
    model, x = build_model().eval(), torch.rand(3, 32, 32)
    with torch.no_grad():
        expected = float(((model(x[None]) - x[None]) ** 2).mean())
    assert compute_reconstruction_error(model, x) == pytest.approx(expected, rel=1e-6)


def test_metric_calculation_and_per_defect_recall():
    preds = [{"file": f"{t}/{i}", "defect_type": t, "label": l, "reconstruction_error": e, "predicted_label": p}
             for i, (t, l, e, p) in enumerate([("good", 0, .1, 0), ("good", 0, .9, 1), ("crack", 1, .8, 1), ("crack", 1, .2, 0), ("pill_type", 1, .7, 1)])]
    m = metrics_from_predictions(preds)
    assert (m["true_negatives"], m["false_positives"], m["false_negatives"], m["true_positives"]) == (1, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(3 / 5) and m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1_score"] == pytest.approx(2 / 3) and m["false_defect_detection_rate"] == pytest.approx(0.5)
    assert m["defect_identification_accuracy"] == m["recall"] and m["confusion_matrix"] == [[1, 1], [1, 2]]
    r = defect_type_recall(preds)
    assert r["crack"] == {"total": 2, "detected": 1, "recall": 0.5} and r["pill_type"]["recall"] == 1.0 and "good" not in r


def test_describe_reports_the_requested_statistics():
    d = describe(range(101))
    assert d["count"] == 101 and d["median"] == 50 and d["p5"] == 5 and d["p25"] == 25 and d["p75"] == 75 and d["p90"] == 90 and d["p99"] == 99
    assert describe([1.0, 3.0])["std"] == 1.0  # population std


def test_gate_classification_follows_the_project_gates():
    assert gate_classification(0.90, 0.85, 0.10) == "EXCELLENT"
    assert gate_classification(0.89, 0.85, 0.10) == "GOOD" and gate_classification(0.85, 0.80, 0.10) == "GOOD"
    assert gate_classification(0.75, 0.70, 0.15) == "ACCEPTABLE" and gate_classification(0.95, 0.95, 0.16) == "NOT PRODUCTION READY"
    assert gate_classification(0.74, 0.90, 0.05) == "NOT PRODUCTION READY"


# ---------------------------------------------------------------------------
# Tiny synthetic pill world: training, artifact, leakage, reproducibility, report
# ---------------------------------------------------------------------------

def _write(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_image_bytes("PNG", size=TINY, color=color))


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for i in range(12):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", (10 + i * 7, 60, 90))
    for i in range(4):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", (12 + i * 11, 61, 91))
    for k, name in enumerate(DEFECT_NAMES):
        for i in range(2):
            _write(root / CATEGORY / "test" / name / f"{i:03d}.png", (240 - k * 15, 20 + i * 9, 30))
    mask = root / CATEGORY / "ground_truth" / "crack" / "000_mask.png"  # a mask that must never be read
    _write(mask, (255, 255, 255))
    art = tmp_path / "ai_models"
    for other in ("bottle", "hazelnut", "carpet", "leather", "metal_nut"):  # sibling artifacts that must never be touched
        (art / other / "phase1_baseline").mkdir(parents=True)
        (art / other / "phase1_baseline" / "autoencoder.pt").write_bytes(other.encode())
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", art)
    monkeypatch.setattr(phase1, "phase1_config", lambda category: TrainingConfig(category=category, image_size=TINY, batch_size=4, epochs=1, device="cpu"))
    return {"root": root, "art": art}


def test_artifact_path_and_isolation_from_other_categories(world):
    path = default_model_path(CATEGORY)
    assert path.as_posix().endswith("ai_models/pill/phase1_baseline/autoencoder.pt")
    siblings = {o: (world["art"] / o / "phase1_baseline" / "autoencoder.pt").read_bytes() for o in ("bottle", "hazelnut", "carpet", "leather", "metal_nut")}
    run = run_phase1(CATEGORY, path, TINY_EXPECTED, lock_path=phase1.phase1_dir(CATEGORY) / "threshold_lock.json")
    assert path.is_file() and run["artifact"]["size_bytes"] == path.stat().st_size > 0
    assert run["artifact"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and run["artifact"]["md5"] == hashlib.md5(path.read_bytes()).hexdigest()
    assert siblings == {o: (world["art"] / o / "phase1_baseline" / "autoencoder.pt").read_bytes() for o in siblings}
    assert run["parameter_count"] == 345_955 and run["leakage_partial"]["saved_artifact_is_what_was_scored"]


def test_run_records_the_required_environment_config_and_fingerprint(world, tmp_path):
    run = run_phase1(CATEGORY, tmp_path / "m" / "autoencoder.pt", TINY_EXPECTED)
    env = run["environment"]
    for key in ("python", "torch", "numpy", "opencv", "torch_threads", "cpu_count"):
        assert env[key]
    assert run["config"]["seed"] == 42 and run["config"]["device"] == "cpu" and run["config"]["loss"] == "MSELoss" and run["config"]["optimizer"] == "Adam"
    assert len(run["training"]["loss_history"]) == 1 and run["training"]["final_loss"] == run["training"]["loss_history"][-1]
    assert run["training_files"] == sorted(run["training_files"]) and len(run["training_files"]) == 12
    assert run["training_files_sha256"] == hashlib.sha256("\n".join(run["training_files"]).encode()).hexdigest()
    samples = discover_train_samples(CATEGORY)
    assert run["dataset_fingerprint_sha256"] == dataset_fingerprint(samples, run["counts"])
    assert run["counts"]["test_by_type"] == TINY_TYPES
    assert set(run["per_defect"]) == set(DEFECT_NAMES) and sum(v["total"] for v in run["per_defect"].values()) == 14  # none omitted or merged
    assert run["gate_classification"] in {"EXCELLENT", "GOOD", "ACCEPTABLE", "NOT PRODUCTION READY"}
    for key in ("model_load_ms", "preprocessing_ms_per_image", "inference_ms_per_image", "total_ms_per_image", "final_test_evaluation_total_ms"):
        assert run["timing"][key] > 0
    assert 0.0 <= run["separation"]["auroc_descriptive_only"] <= 1.0


def test_only_train_good_images_are_trained_on_and_the_seed_is_recorded_and_deterministic(world, tmp_path, monkeypatch):
    seen = {}
    real_train = phase1.train_anomaly_model_on_samples
    monkeypatch.setattr(phase1, "train_anomaly_model_on_samples",
                        lambda samples, cfg, path: (seen.update(samples=list(samples), seed=cfg.seed), real_train(samples, cfg, path))[1])
    a = run_phase1(CATEGORY, tmp_path / "a" / "autoencoder.pt", TINY_EXPECTED)
    assert seen["seed"] == 42 and len(seen["samples"]) == 12
    assert all(s.split == "train" and s.defect_type == "good" and s.label == 0 and "test" not in s.path.parts for s in seen["samples"])
    b = run_phase1(CATEGORY, tmp_path / "b" / "autoencoder.pt", TINY_EXPECTED)
    assert a["artifact"]["sha256"] == b["artifact"]["sha256"]  # same seed -> same weights


def test_threshold_is_from_train_scores_only_and_unaffected_by_the_test_set(world, tmp_path):
    a = run_phase1(CATEGORY, tmp_path / "a" / "autoencoder.pt", TINY_EXPECTED)
    arr = np.asarray(a["train_errors"])
    assert len(arr) == 12 and a["threshold"]["value"] == pytest.approx(float(arr.mean() + 3 * arr.std()), abs=1e-15) and a["threshold"]["k"] == 3.0
    assert a["threshold"]["value"] != pytest.approx(float(arr.mean() + 3 * arr.std(ddof=1)), abs=1e-15)
    for path in (world["root"] / CATEGORY / "test").rglob("*.png"):  # change every test image
        _write(path, (250, 5, 5))
    b = run_phase1(CATEGORY, tmp_path / "b" / "autoencoder.pt", TINY_EXPECTED)
    assert b["threshold"]["value"] == a["threshold"]["value"] and b["artifact"]["sha256"] == a["artifact"]["sha256"]
    assert a["predictions"] != b["predictions"]  # ...while the test predictions do change


def test_no_test_image_is_decoded_before_the_lock_and_masks_are_never_read(world, tmp_path, monkeypatch):
    import app.ai.preprocessing.pipeline as pipeline

    decoded, trained = [], {}
    real_load = pipeline._load_image
    monkeypatch.setattr(pipeline, "_load_image", lambda p: (decoded.append(Path(p)), real_load(p))[1])
    real_train = phase1.train_anomaly_model_on_samples
    monkeypatch.setattr(phase1, "train_anomaly_model_on_samples",
                        lambda samples, cfg, path: (trained.update(paths=[s.path for s in samples]), real_train(samples, cfg, path))[1])
    seen = {}
    real_discover = phase1.discover_test_samples
    lock_file = phase1.phase1_dir(CATEGORY) / "threshold_lock.json"

    def spy(category):
        seen["lock_on_disk"] = lock_file.is_file()
        seen["test_decoded_before_lock"] = [p for p in decoded if "test" in p.parts]
        return real_discover(category)

    monkeypatch.setattr(phase1, "discover_test_samples", spy)
    # verify_dataset_counts also lists the test tree by name (metadata only); it uses its own reference to discovery
    run = run_phase1(CATEGORY, tmp_path / "m" / "autoencoder.pt", TINY_EXPECTED, lock_path=lock_file)
    assert seen == {"lock_on_disk": True, "test_decoded_before_lock": []}
    assert all(p.parent.name == "good" and "train" in p.parts for p in trained["paths"]) and len(trained["paths"]) == 12
    assert not any("ground_truth" in str(p) for p in decoded)
    test_positions = [i for i, p in enumerate(decoded) if "test" in p.parts]
    train_positions = [i for i, p in enumerate(decoded) if "train" in p.parts]
    assert len(test_positions) == 18 and train_positions and min(test_positions) > max(train_positions)  # test decoded only after all training/threshold work
    assert all(run["leakage_partial"].values()), run["leakage_partial"]
    assert json.loads(lock_file.read_text())["threshold"] == run["threshold"]["value"]
    assert json.loads(lock_file.read_text())["model_sha256"] == run["artifact"]["sha256"]


def test_two_independent_runs_reproduce_everything(world, tmp_path):
    a = run_phase1(CATEGORY, tmp_path / "a" / "autoencoder.pt", TINY_EXPECTED)
    b = run_phase1(CATEGORY, tmp_path / "b" / "autoencoder.pt", TINY_EXPECTED)
    cmp = compare_runs(a, b)
    assert all(v for k, v in cmp.items() if isinstance(v, bool)), cmp
    assert cmp["dataset_fingerprint_identical"] and cmp["training_order_identical"] and cmp["model_configuration_identical"]
    assert cmp["loss_history_max_abs_difference"] == 0.0 and cmp["test_errors_max_abs_difference"] == 0.0
    tampered = json.loads(json.dumps(a))
    tampered["dataset_fingerprint_sha256"] = "0" * 64
    tampered["threshold"]["value"] += 1e-9
    result = compare_runs(a, tampered)
    assert not result["dataset_fingerprint_identical"] and not result["threshold_identical"]


def test_existing_artifact_is_never_overwritten_and_count_mismatch_stops_before_training(world, tmp_path, monkeypatch):
    target = tmp_path / "out" / "autoencoder.pt"
    target.parent.mkdir()
    target.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        run_phase1(CATEGORY, target, TINY_EXPECTED)
    assert target.read_bytes() == b"precious"
    monkeypatch.setattr(phase1, "train_anomaly_model_on_samples", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained")))
    with pytest.raises(ValueError, match="test_defective"):
        run_phase1(CATEGORY, tmp_path / "x" / "autoencoder.pt", {**TINY_EXPECTED, "test_defective": 141})


def _load_script():
    path = Path(phase1.__file__).parents[3] / "scripts" / "run_category_phase1.py"
    spec = importlib.util.spec_from_file_location("run_category_phase1_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv(monkeypatch, types: str):
    monkeypatch.setattr(sys, "argv", ["run_category_phase1.py", "--category", CATEGORY, "--train-good", "12", "--test-good", "4",
                                       "--test-defective", "14", "--defect-types", types])


def test_script_writes_the_phase1_report_with_all_sections_and_flags(world, monkeypatch, capsys):
    script = _load_script()
    _argv(monkeypatch, ",".join(f"{k}={v}" for k, v in TINY_TYPES.items()))
    script.main()
    report = json.loads((phase1.phase1_dir(CATEGORY) / "reports" / "phase1_report.json").read_text(encoding="utf-8"))
    assert {"git_head", "run1", "run2_summary", "reproducibility", "preexisting_artifact_count"} <= set(report)
    assert all(v for v in report["reproducibility"].values() if isinstance(v, bool)), report["reproducibility"]
    leak = report["run1"]["leakage_partial"]
    assert leak["9_existing_artifacts_not_overwritten"] and leak["9b_serving_registry_unchanged"] and all(leak.values())
    assert report["run1"]["counts"]["test_by_type"] == TINY_TYPES and report["preexisting_artifact_count"] == 5
    assert (phase1.phase1_dir(CATEGORY) / "autoencoder.pt").is_file() and (phase1.phase1_dir(CATEGORY) / "threshold_lock.json").is_file()
    assert "RUN 1" in capsys.readouterr().out


def test_script_stops_when_the_enumerated_defect_types_differ_from_the_declared_ones(world, monkeypatch):
    script = _load_script()
    _argv(monkeypatch, "good=4,crack=14")
    with pytest.raises(SystemExit, match="STOP"):
        script.main()
    assert not default_model_path(CATEGORY).exists()  # stopped before any training


def test_phase1_module_uses_no_other_phase_machinery_and_serving_does_not_register_pill():
    tree = ast.parse(Path(phase1.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    forbidden = ("threshold_experiments", "phase3_threshold_selection", "calibration", "patch_anomaly", "resnet18", "validation_split", "phase4")
    assert not [m for m in imported if any(f in m for f in forbidden)]
    serving = (Path(phase1.__file__).parent.parent / "inference" / "serving.py").read_text(encoding="utf-8").lower()
    assert "pill" not in serving.replace("pillow", "")
