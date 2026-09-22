"""Calibration optimization pipeline end-to-end on a tiny synthetic category: study -> selection -> freeze ->
single final-test evaluation, plus the FPR-aware gates and every refusal path.

Uses an UNTRAINED seeded ResNet-18 (no pretrained weights needed) and 20 small good images; mechanics only.
"""

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ai.evaluation import calibration_final_test as final_module
from app.ai.evaluation import calibration_pipeline as pipeline
from app.ai.evaluation import calibration_study as cs
from app.ai.evaluation.calibration_final_test import (
    ACCEPTABLE, EXCELLENT, GOOD, NOT_PRODUCTION_READY, classify_with_fpr, compare_with_baseline,
    evaluate_optimized_on_final_test, result_path,
)
from app.ai.evaluation.calibration_pipeline import (
    compare_studies, fit_final_detector, freeze_optimized, load_inputs, reports_dir, run_study, selected_dir,
)
from app.ai.evaluation.category_phase3 import metrics_from_predictions, predictions_at_threshold
from app.ai.inference.serving import SERVING_CONFIGS
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.resnet18 import RESNET18_SHA256, ResNet18
from app.ai.training import discover_test_samples

CATEGORY = "widget"


def _extractor() -> ResNet18:
    torch.manual_seed(0)
    model = ResNet18().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# FPR-aware gates
# ---------------------------------------------------------------------------

def test_gates_require_a_low_false_positive_rate():
    assert classify_with_fpr(0.99, 0.91, 0.5714) == NOT_PRODUCTION_READY  # the original Carpet result
    assert classify_with_fpr(0.90, 0.85, 0.10) == EXCELLENT and classify_with_fpr(0.90, 0.85, 0.11) == ACCEPTABLE
    assert classify_with_fpr(0.85, 0.80, 0.10) == GOOD and classify_with_fpr(0.85, 0.80, 0.101) == ACCEPTABLE
    assert classify_with_fpr(0.75, 0.70, 0.15) == ACCEPTABLE and classify_with_fpr(0.75, 0.70, 0.151) == NOT_PRODUCTION_READY
    assert classify_with_fpr(0.74, 0.99, 0.0) == NOT_PRODUCTION_READY and classify_with_fpr(0.99, 0.69, 0.0) == NOT_PRODUCTION_READY


def _result(preds):
    return {"metrics": metrics_from_predictions(preds), "predictions": preds}


def _pred(name, label, predicted):
    return {"file": name, "defect_type": "good" if label == 0 else "d", "label": label,
            "reconstruction_error": 0.0, "predicted_label": predicted}


def test_comparison_reports_deltas_and_paired_changes():
    base = _result([_pred("g1", 0, 1), _pred("g2", 0, 1), _pred("g3", 0, 0), _pred("d1", 1, 1), _pred("d2", 1, 1), _pred("d3", 1, 0)])
    opt = _result([_pred("g1", 0, 0), _pred("g2", 0, 1), _pred("g3", 0, 1), _pred("d1", 1, 1), _pred("d2", 1, 0), _pred("d3", 1, 1)])
    c = compare_with_baseline(base, opt)
    assert c["paired"] == {"same_test_images": True, "false_positives_removed": 1, "false_positives_remaining": 1,
                           "new_false_positives": 1, "defects_lost": 1, "defects_gained": 1}
    assert c["metrics"]["false_positives"]["absolute_change"] == 0
    assert c["metrics"]["false_defect_detection_rate"]["baseline"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# The tiny world
# ---------------------------------------------------------------------------

def _write_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[..., ::-1])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calibration")
    root = tmp / "dataset"
    rng = np.random.default_rng(3)
    base = rng.normal(110, 20, (128, 128, 3))
    for i in range(20):
        _write_png(root / CATEGORY / "train" / "good" / f"{i:03d}.png", np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8))
    for i in range(4):
        _write_png(root / CATEGORY / "test" / "good" / f"{i:03d}.png", np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8))
    for defect, value in (("scratch", 5), ("stain", 250)):
        for i in range(3):
            img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
            img[40 + 10 * i:70 + 10 * i, 40:90] = value
            _write_png(root / CATEGORY / "test" / defect / f"{i:03d}.png", img)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.ai.training.dataset.DATASET_ROOT", root)
        mp.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp / "ai_models")
        extractor = _extractor()
        inp = load_inputs(CATEGORY, extractor, log=lambda *_: None)
        study1 = run_study(inp, log=lambda *_: None)
        study2 = run_study(inp, log=lambda *_: None)
        yield SimpleNamespace(tmp=tmp, inp=inp, study1=study1, study2=study2, extractor=extractor, mp=mp)


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def test_study_covers_the_declared_schemes_aggregations_and_policies(world):
    s = world.study1.summary
    assert len(s["configurations"]) == 5 * 9 and s["n_good"] == 20
    assert {c["aggregation"] for c in s["configurations"]} == set(cs.STUDY_AGGREGATIONS)
    assert s["split_design"]["folds_per_scheme"] == {
        "random_5fold": 15, "contiguous_blocks": 5, "holdout80_20": 10, "holdout75_25": 10, "holdout70_30": 10}
    assert s["declared_gates"] == {"primary_fpr": 0.10, "secondary_fpr": 0.05, "synthetic_recall_floor": 0.75, "recall_tie": 0.03}
    assert s["consistency_with_original_stage_a"] is None  # no stage-A file in the tiny world
    assert s["synthetic_diagnostic"]["note"].startswith("SYNTHETIC")


def test_study_selection_is_the_pre_declared_policy_applied_to_the_rows(world):
    again = cs.select_configuration(world.study1.rows)
    assert again.winner == world.study1.selection.winner
    assert world.study1.summary["selection"]["winner"] == world.study1.selection.winner.config_id
    for cid, t in world.study1.thresholds.items():
        assert t["final"] == max(t["random"], t["block"])  # the more conservative calibration scheme wins


def test_two_complete_study_runs_are_identical(world):
    result = compare_studies(world.study1, world.study2)
    assert result["all_identical"], result


def test_pipeline_and_study_modules_cannot_list_test_images(world):
    def refs(module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        found = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        found |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        found |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        return found & {"discover_test_samples"}

    assert refs(pipeline) == set() and refs(cs) == set() and refs(final_module) == {"discover_test_samples"}


def test_study_never_lists_test_images(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the calibration study must not list test images")

    monkeypatch.setattr("app.ai.training.dataset.discover_test_samples", boom)
    run_study(world.inp, log=lambda *_: None)


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen(world):
    reports = reports_dir(CATEGORY)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "calibration_study.json").write_text(json.dumps(world.study1.summary), encoding="utf-8")
    baseline_dir = world.tmp / "ai_models" / CATEGORY / "selected_model"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "model_state.pt").write_bytes(b"original baseline artifact")
    baseline_ref = {"model_state_md5": hashlib.md5(b"original baseline artifact").hexdigest(),
                    "model_state_sha256": hashlib.sha256(b"original baseline artifact").hexdigest(),
                    "threshold": 1.0, "aggregation": "top1pct_mean"}
    lock = freeze_optimized(world.inp, world.study1, selected_dir(CATEGORY), baseline_ref)
    return SimpleNamespace(lock=lock, dir=selected_dir(CATEGORY), baseline_ref=baseline_ref, baseline_dir=baseline_dir)


def test_freeze_writes_a_complete_lock_record(world, frozen):
    lock, winner = frozen.lock, world.study1.selection.winner
    state = frozen.dir / lock["state_file"]["name"]
    assert lock["state_file"]["sha256"] == hashlib.sha256(state.read_bytes()).hexdigest()
    assert lock["threshold"]["value"] == winner.final_threshold
    assert lock["config"]["aggregation"] == winner.aggregation and lock["config"]["scorer"] == "gaussian"
    assert lock["fitted_on"]["training_images"] == 20 and lock["backbone"]["sha256"] == RESNET18_SHA256
    assert lock["baseline_reference"]["model_state_sha256"] == frozen.baseline_ref["model_state_sha256"]
    assert lock["selection"]["passes_primary_gate"] == world.study1.selection.passes_primary_gate
    assert lock["digest"] and json.loads((frozen.dir / "frozen_model.json").read_text())["digest"] == lock["digest"]


def test_freeze_refuses_to_overwrite_and_a_second_fit_reproduces_the_artifact(world, frozen, tmp_path):
    with pytest.raises(FileExistsError):
        freeze_optimized(world.inp, world.study1, frozen.dir, frozen.baseline_ref)
    again = fit_final_detector(world.inp, world.study1.selection.winner.aggregation)
    state = again.save(tmp_path)
    assert hashlib.sha256(state.read_bytes()).hexdigest() == frozen.lock["state_file"]["sha256"]
    reloaded = PatchAnomalyDetector.load(frozen.dir, world.extractor)
    probe = torch.rand(2, 3, 256, 256)
    assert reloaded.score_images(probe) == again.score_images(probe)


def test_optimized_artifact_lives_beside_not_over_the_baseline(world, frozen):
    assert frozen.dir.name == "selected_candidate" and frozen.dir.parent.name == "optimization"
    assert (frozen.baseline_dir / "model_state.pt").read_bytes() == b"original baseline artifact"


# ---------------------------------------------------------------------------
# Final test
# ---------------------------------------------------------------------------

def _write_baseline_result(world):
    import app.ai.training.dataset as ds

    samples = ds.discover_test_samples(CATEGORY)
    preds = predictions_at_threshold(samples, [1.0] * len(samples), 0.5)  # a fake baseline: flags everything
    path = final_module.baseline_result_path(CATEGORY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics_from_predictions(preds), "predictions": preds}), encoding="utf-8")


def test_final_test_refuses_tampered_artifacts_before_touching_any_test_image(world, frozen, monkeypatch):
    _write_baseline_result(world)

    def boom(*a, **k):
        raise AssertionError("test images were listed before the frozen artifacts were verified")

    monkeypatch.setattr(final_module, "discover_test_samples", boom)
    state = frozen.dir / "model_state.pt"
    original = state.read_bytes()
    try:
        state.write_bytes(b"corrupted after freezing")
        with pytest.raises(RuntimeError, match="changed after freezing"):
            evaluate_optimized_on_final_test(CATEGORY)
    finally:
        state.write_bytes(original)
    baseline_state = frozen.baseline_dir / "model_state.pt"
    try:
        baseline_state.write_bytes(b"baseline overwritten")
        with pytest.raises(RuntimeError, match="baseline artifact no longer matches"):
            evaluate_optimized_on_final_test(CATEGORY)
    finally:
        baseline_state.write_bytes(b"original baseline artifact")
    assert not result_path(CATEGORY).exists()


def test_final_test_runs_once_with_the_locked_threshold_gates_audit_and_baseline_comparison(world, frozen, monkeypatch):
    monkeypatch.setattr(final_module, "load_pretrained_resnet18", lambda: world.extractor)
    result = evaluate_optimized_on_final_test(CATEGORY)

    assert result["counts"] == {"test_total": 10, "test_good": 4, "test_defective": 6}
    m = result["metrics"]
    assert m["true_negatives"] + m["false_positives"] == 4 and m["false_negatives"] + m["true_positives"] == 6
    assert result["threshold"]["value"] == frozen.lock["threshold"]["value"]
    assert all(p["predicted_label"] == (1 if p["reconstruction_error"] > frozen.lock["threshold"]["value"] else 0)
               for p in result["predictions"])
    assert result["classification"] == classify_with_fpr(m["recall"], m["f1_score"], m["false_defect_detection_rate"])
    assert set(result["recall_by_defect_type"]) == {"scratch", "stain"}
    audit = result["leakage_audit"]
    assert audit["all_passed"], {k: v for k, v in audit.items() if not v}
    for key in ("test_disjoint_from_pool_by_content", "locked_threshold_equals_the_good_only_study_threshold",
                "fit_and_calibration_pool_is_exactly_the_train_good_images", "original_baseline_artifact_unchanged"):
        assert audit[key] is True
    cmp = result["comparison_with_original_baseline"]
    assert cmp["paired"]["same_test_images"] and "false_defect_detection_rate" in cmp["metrics"]
    assert "Second scoring" in result["test_set_use_note"]
    assert result_path(CATEGORY).is_file()

    with pytest.raises(FileExistsError, match="scored once"):
        evaluate_optimized_on_final_test(CATEGORY)


def test_serving_registry_is_untouched():
    assert set(SERVING_CONFIGS) == {"bottle"}
    assert dataclasses.is_dataclass(cs.ConfigRow) and discover_test_samples is not None
