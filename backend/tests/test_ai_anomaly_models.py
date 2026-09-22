"""Patch-level anomaly detectors, synthetic defects, model-selection engine, freeze, final-test protocol.

Mechanics tests use a tiny synthetic texture category (64x64 images) and an UNTRAINED, seeded ResNet-18 so
they run in seconds and never need the pretrained weights; tests that need the real weights skip when the
(Git-ignored) file is absent.
"""

import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ai.evaluation import anomaly_final_test as final_module
from app.ai.evaluation import anomaly_model_selection as engine
from app.ai.evaluation.anomaly_final_test import (
    ACCEPTABLE, BELOW_TARGET, EXCELLENT, GOOD, classify_result, evaluate_frozen_on_final_test, final_result_path,
)
from app.ai.evaluation.anomaly_model_selection import (
    CONVAE_GLOBAL, FAMILY_CONVAE, FAMILY_GAUSSIAN, FAMILY_KNN, CandidateSpec, DevResult, Timing, build_dev_arrays,
    dev_metrics, fit_detector, freeze_detector, run_stage, score_uint8, select_best, stage_a_candidates,
)
from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES, make_synthetic_defect
from app.ai.inference.serving import SERVING_CONFIGS
from app.ai.models import patch_anomaly as pa
from app.ai.models.resnet18 import (
    RESNET18_SHA256, PretrainedWeightsError, ResNet18, load_pretrained_resnet18, pretrained_weights_path,
)
from app.ai.training import discover_train_samples
from app.ai.training.validation_split import split_train_validation
from tests.conftest import make_image_bytes  # noqa: F401  (kept for parity with sibling test modules)

CATEGORY = "widget"
SIDE = 64
REAL_WEIGHTS = pretrained_weights_path()


def _extractor() -> ResNet18:
    torch.manual_seed(0)
    model = ResNet18().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# ResNet-18 loader
# ---------------------------------------------------------------------------

def test_resnet18_feature_shapes():
    out = _extractor()(torch.rand(2, 3, 64, 64), (1, 2, 3))
    assert {k: tuple(v.shape) for k, v in out.items()} == {1: (2, 64, 16, 16), 2: (2, 128, 8, 8), 3: (2, 256, 4, 4)}


def test_missing_and_tampered_weights_are_refused(tmp_path):
    with pytest.raises(PretrainedWeightsError, match="not found"):
        load_pretrained_resnet18(tmp_path / "nope.pth")
    bad = tmp_path / "resnet18-f37072fd.pth"
    bad.write_bytes(b"not the real weights")
    with pytest.raises(PretrainedWeightsError, match="hash mismatch"):
        load_pretrained_resnet18(bad)


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="pretrained ResNet-18 weights not present")
def test_real_pretrained_weights_load_strictly_and_are_frozen():
    model = load_pretrained_resnet18()
    assert hashlib.sha256(REAL_WEIGHTS.read_bytes()).hexdigest() == RESNET18_SHA256
    assert sum(p.numel() for p in model.parameters()) == 11_689_512  # torchvision resnet18's parameter count
    assert not model.training and not any(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# Patch features, coreset, scorers, aggregation
# ---------------------------------------------------------------------------

def test_patch_feature_shapes():
    x = torch.rand(3, 3, SIDE, SIDE)
    assert tuple(pa.extract_patch_features(_extractor(), x, (2, 3)).shape) == (3, 8, 8, 384)
    assert tuple(pa.extract_patch_features(_extractor(), x, (2,)).shape) == (3, 8, 8, 128)


def test_coreset_is_deterministic_sized_and_seed_dependent():
    feats = torch.randn(500, 96)
    a = pa.greedy_coreset_indices(feats, 50, seed=1)
    assert torch.equal(a, pa.greedy_coreset_indices(feats, 50, seed=1))
    assert len(a) == 50 and len(set(a.tolist())) == 50
    assert not torch.equal(a, pa.greedy_coreset_indices(feats, 50, seed=2))
    assert len(pa.greedy_coreset_indices(feats, 10_000, seed=1)) == 500  # capped at N


def test_knn_scores_bank_members_zero_and_outliers_high():
    train = torch.randn(4, 6, 6, 32)
    scorer = pa.KNNPatchScorer.fit(train, coreset_fraction=1.0, seed=0)
    # torch.cdist's fast matmul path leaves ~1e-3 rounding noise on self-distances; real distances are >> that
    assert torch.allclose(scorer.patch_scores(train), torch.zeros(4, 6, 6), atol=1e-2)
    outlier = train[:1].clone()
    outlier[0, 2, 2] += 50.0
    scores = scorer.patch_scores(outlier)
    assert scores[0, 2, 2] > 10 and scores[0, 0, 0] < 1e-2


def test_gaussian_scores_are_nonnegative_and_flag_outliers():
    train = torch.randn(8, 6, 6, 16)
    scorer = pa.GaussianPatchScorer.fit(train)
    inlier, outlier = torch.randn(1, 6, 6, 16), torch.randn(1, 6, 6, 16) + 8
    assert (scorer.patch_scores(inlier) >= 0).all()
    assert scorer.patch_scores(outlier).mean() > 3 * scorer.patch_scores(inlier).mean()


def test_aggregation_math():
    maps = torch.zeros(2, 10, 10)
    maps[0, 3, 3] = 5.0
    maps[1] = 1.0
    assert pa.aggregate_scores(maps, pa.AGG_MAX).tolist() == [5.0, 1.0]
    top = pa.aggregate_scores(maps, pa.AGG_TOP1PCT)  # top 1% of 100 patches = 1 patch
    assert top.tolist() == [5.0, 1.0]
    with pytest.raises(ValueError):
        pa.aggregate_scores(maps, "median")


@pytest.mark.parametrize("scorer_kind", ["knn", "gaussian"])
def test_detector_save_load_roundtrip_gives_identical_scores(scorer_kind, tmp_path):
    ext = _extractor()
    train = pa.extract_patch_features(ext, torch.rand(6, 3, SIDE, SIDE), (2, 3))
    scorer = (pa.KNNPatchScorer.fit(train, 0.5, 0) if scorer_kind == "knn" else pa.GaussianPatchScorer.fit(train))
    detector = pa.PatchAnomalyDetector(ext, (2, 3), SIDE, scorer, pa.AGG_MAX)
    images = torch.rand(3, 3, SIDE, SIDE)
    detector.save(tmp_path)
    reloaded = pa.PatchAnomalyDetector.load(tmp_path, ext)
    assert reloaded.score_images(images) == detector.score_images(images)
    assert reloaded.config() == detector.config()


# ---------------------------------------------------------------------------
# Synthetic defects
# ---------------------------------------------------------------------------

def _texture(seed=0, size=128):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 25, (size, size, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_synthetic_defect_is_deterministic_visible_and_non_mutating(defect_type, severity):
    image = _texture()
    original = image.copy()
    a, mask = make_synthetic_defect(image, defect_type, severity, seed=7)
    b, _ = make_synthetic_defect(image, defect_type, severity, seed=7)
    assert np.array_equal(a, b) and np.array_equal(image, original)
    assert mask.any() and not np.array_equal(a, image) and a.shape == image.shape and a.dtype == np.uint8


def test_synthetic_defects_differ_by_seed_and_grow_with_severity():
    image = _texture(size=256)
    assert not np.array_equal(make_synthetic_defect(image, "dark_blob", 1, 1)[0], make_synthetic_defect(image, "dark_blob", 1, 2)[0])
    for blob in ("color_blob", "dark_blob"):
        areas = [make_synthetic_defect(image, blob, s, seed=3)[1].mean() for s in SEVERITIES]
        assert areas[0] < areas[1] < areas[2]


def test_synthetic_defect_rejects_unknown_arguments():
    with pytest.raises(ValueError):
        make_synthetic_defect(_texture(), "scratch", 0)
    with pytest.raises(ValueError):
        make_synthetic_defect(_texture(), "line", 5)


# ---------------------------------------------------------------------------
# A tiny world: dataset + development arrays
# ---------------------------------------------------------------------------

def _write_png(path: Path, array_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array_rgb[..., ::-1])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anomaly")
    root = tmp / "dataset"
    rng = np.random.default_rng(1)
    base = rng.normal(110, 20, (SIDE * 2, SIDE * 2, 3))
    for i in range(20):  # 20 train/good -> 16 training, 4 validation
        img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
        _write_png(root / CATEGORY / "train" / "good" / f"{i:03d}.png", img)
    for i in range(4):
        img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
        _write_png(root / CATEGORY / "test" / "good" / f"{i:03d}.png", img)
    for defect in ("scratch", "stain"):
        for i in range(3):
            img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
            img[40 + 10 * i : 70 + 10 * i, 40:80] = 250 if defect == "stain" else 5
            _write_png(root / CATEGORY / "test" / defect / f"{i:03d}.png", img)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.ai.training.dataset.DATASET_ROOT", root)
        mp.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp / "ai_models")
        split = split_train_validation(discover_train_samples(CATEGORY), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
        data = build_dev_arrays(split, (SIDE,))
        yield SimpleNamespace(tmp=tmp, root=root, split=split, data=data, extractor=_extractor(), mp=mp)


def _spec(family=FAMILY_KNN, agg=pa.AGG_MAX, layers=(2, 3), fraction=0.5, cid="t"):
    return CandidateSpec(cid, family, SIDE, agg, layers, fraction if family == FAMILY_KNN else None)


def test_dev_arrays_shapes_counts_and_provenance(world):
    d = world.data
    assert d.train[SIDE].shape == (16, SIDE, SIDE, 3) and d.validation[SIDE].shape == (4, SIDE, SIDE, 3)
    assert d.synthetic[SIDE].shape == (4 * 15, SIDE, SIDE, 3) and len(d.synthetic_meta) == 60
    assert d.train[SIDE].dtype == np.uint8
    val_names, train_names = set(d.validation_files), set(d.train_files)
    assert {m["source_file"] for m in d.synthetic_meta} == val_names  # synthetic defects derive ONLY from validation images
    assert not (val_names & train_names) and len(d.train_files) == 16


def test_dev_arrays_are_deterministic(world):
    again = build_dev_arrays(world.split, (SIDE,))
    assert again.synthetic_digest == world.data.synthetic_digest
    assert np.array_equal(again.synthetic[SIDE], world.data.synthetic[SIDE])


# ---------------------------------------------------------------------------
# Development metrics + pre-declared selection
# ---------------------------------------------------------------------------

def _meta(n_per_cell=2):
    return [{"source_file": f"{i}.png", "defect_type": t, "severity": s}
            for t in DEFECT_TYPES for s in SEVERITIES for i in range(n_per_cell)]


def _dev(val, syn, meta=None, spec=None):
    meta = meta or _meta(2)
    return dev_metrics(spec or _spec(), val, syn[: len(meta)] if len(syn) >= len(meta) else syn, meta, 1000, Timing())


def test_dev_metrics_recall_and_auroc_at_the_validation_locked_threshold():
    val = [1.0 + 0.01 * i for i in range(20)]
    meta = _meta(2)
    perfect = _dev(val, [10.0] * len(meta), meta)
    assert perfect.synthetic_recall_pooled == 1.0 and perfect.synthetic_auroc == 1.0
    assert perfect.validation_fpr_at_selected <= 0.10 and perfect.selection_status == "SELECTED"
    useless = _dev(val, [1.05] * len(meta), meta)  # synthetic scores inside the normal range
    assert useless.synthetic_recall_pooled < 0.6 and useless.synthetic_auroc < 0.8
    assert set(perfect.synthetic_recall_by_type) == set(DEFECT_TYPES)
    assert set(perfect.synthetic_recall_by_severity) == {"small", "medium", "large"}


def test_threshold_is_derived_from_validation_scores_only():
    val = [1.0 + 0.01 * i for i in range(20)]
    meta = _meta(2)
    a = _dev(val, [10.0] * len(meta), meta)
    b = _dev(val, [3.0] * len(meta), meta)
    assert a.selected_threshold == b.selected_threshold  # synthetic scores never influence the threshold


def _fake_result(cid, recall, auroc, side=256, nbytes=1000, status="SELECTED"):
    return SimpleNamespace(spec=SimpleNamespace(candidate_id=cid, image_size=side), selection_status=status,
                           synthetic_recall_pooled=recall, synthetic_auroc=auroc, state_bytes=nbytes)


def test_select_best_ranks_by_pooled_synthetic_recall():
    sel = select_best([_fake_result("a", 0.80, 0.99), _fake_result("b", 0.95, 0.90), _fake_result("c", 0.60, 1.0)])
    assert sel.winner_id == "b" and [r[0] for r in sel.ranked] == ["b", "a", "c"] and not sel.escalation_needed


def test_recall_ties_go_to_auroc_then_to_the_cheaper_candidate():
    assert select_best([_fake_result("x", 0.95, 0.97), _fake_result("y", 0.94, 0.99)]).winner_id == "y"
    cheap = select_best([_fake_result("big", 0.95, 0.990, side=320), _fake_result("small", 0.95, 0.988, side=224)])
    assert cheap.winner_id == "small"
    same_side = select_best([_fake_result("m", 0.95, 0.99, nbytes=9000), _fake_result("n", 0.95, 0.99, nbytes=100)])
    assert same_side.winner_id == "n"


def test_ineligible_candidates_are_excluded_and_escalation_triggers_below_090():
    sel = select_best([_fake_result("bad", 0.99, 1.0, status="NO DEFENSIBLE WINNER"), _fake_result("ok", 0.70, 0.9)])
    assert sel.winner_id == "ok" and sel.disqualified == ["bad"] and sel.escalation_needed
    assert not select_best([_fake_result("ok", 0.90, 0.9)]).escalation_needed
    with pytest.raises(ValueError):
        select_best([_fake_result("bad", 0.99, 1.0, status="NO DEFENSIBLE WINNER")])


def test_stage_a_pool_is_the_documented_sixteen_and_contains_the_failed_baseline():
    specs = stage_a_candidates()
    ids = [s.candidate_id for s in specs]
    assert len(specs) == len(set(ids)) == 16 and "convae128_global" in ids
    assert {s.family for s in specs} == {FAMILY_CONVAE, FAMILY_KNN, FAMILY_GAUSSIAN}


def test_study_module_cannot_list_test_images_but_the_final_module_can():
    def refs(module, names):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        found = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        found |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        found |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        return found & names

    assert refs(engine, {"discover_test_samples"}) == set()
    assert refs(final_module, {"discover_test_samples"}) == {"discover_test_samples"}


def test_engine_run_uses_training_images_for_fitting_and_never_lists_test_images(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model selection must not list test images")

    monkeypatch.setattr("app.ai.training.dataset.discover_test_samples", boom)
    specs = [_spec(FAMILY_KNN, pa.AGG_MAX, cid="knn_max"), _spec(FAMILY_KNN, pa.AGG_TOP1PCT, cid="knn_top"),
             _spec(FAMILY_GAUSSIAN, pa.AGG_MAX, cid="gauss_max")]
    results = run_stage(specs, CATEGORY, world.extractor, world.data, world.tmp, log=lambda *_: None)
    assert [r.spec.candidate_id for r in results] == ["knn_max", "knn_top", "gauss_max"]
    knn = results[0]
    assert knn.state_bytes == round(16 * 64 * 0.5) * 384 * 4  # bank built from the 16 TRAINING images' patches only
    assert len(knn.validation_scores) == 4 and len(knn.synthetic_scores) == 60


def test_training_image_scores_zero_under_a_full_bank_but_validation_images_do_not(world):
    spec = dataclasses.replace(_spec(FAMILY_KNN), coreset_fraction=1.0)
    detector, _ = fit_detector(spec, world.extractor, world.data)
    train_scores = score_uint8(detector, world.data.train[SIDE][:3])
    val_scores = score_uint8(detector, world.data.validation[SIDE])
    assert max(train_scores) < 1e-2 < min(val_scores)  # validation is NOT in the memory bank


# ---------------------------------------------------------------------------
# Freeze + final test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen(world):
    spec = _spec(FAMILY_KNN, pa.AGG_MAX, cid="knn_max")
    dev = engine.evaluate_patch_group([spec], world.extractor, world.data, log=lambda *_: None)[0]
    out = world.tmp / "ai_models" / CATEGORY / "selected_model"
    model = freeze_detector(spec, world.extractor, world.data, out, RESNET18_SHA256, dev)
    return SimpleNamespace(model=model, dir=out, spec=spec, dev=dev)


def test_freeze_writes_the_lock_record_with_matching_hashes(world, frozen):
    lock = json.loads((frozen.dir / "frozen_model.json").read_text())
    state = frozen.dir / lock["state_file"]["name"]
    assert lock["state_file"]["sha256"] == hashlib.sha256(state.read_bytes()).hexdigest()
    assert lock["state_file"]["md5"] == hashlib.md5(state.read_bytes()).hexdigest()
    assert lock["state_file"]["size_bytes"] == state.stat().st_size
    assert lock["threshold"]["value"] == frozen.model.threshold and lock["backbone"]["sha256"] == RESNET18_SHA256
    assert lock["fitted_on"]["training_images"] == 16
    assert lock["development_evidence_synthetic"]["note"].startswith("SYNTHETIC")
    assert lock["threshold"]["validation_images"] == 4 and lock["threshold"]["validation_fp_ceiling"] == 0.10


def test_frozen_threshold_equals_the_validation_derived_selection(world, frozen):
    from app.ai.evaluation.phase3_threshold_selection import select_threshold_from_validation
    from app.ai.evaluation.threshold_experiments import generate_candidates

    val = score_uint8(pa.PatchAnomalyDetector.load(frozen.dir, world.extractor), world.data.validation[SIDE])
    assert select_threshold_from_validation(generate_candidates(val), val).selected.threshold == frozen.model.threshold


def test_freeze_refuses_to_overwrite_and_is_reproducible(world, frozen):
    with pytest.raises(FileExistsError):
        freeze_detector(frozen.spec, world.extractor, world.data, frozen.dir, RESNET18_SHA256, frozen.dev)
    other = freeze_detector(frozen.spec, world.extractor, world.data, world.tmp / "second_freeze", RESNET18_SHA256, frozen.dev)
    assert (other.state_sha256, other.threshold, other.digest, other.validation_scores) == (
        frozen.model.state_sha256, frozen.model.threshold, frozen.model.digest, frozen.model.validation_scores)


def test_classification_bands():
    assert classify_result(0.90, 0.85) == EXCELLENT and classify_result(0.95, 0.84) == GOOD
    assert classify_result(0.85, 0.80) == GOOD and classify_result(0.85, 0.79) == ACCEPTABLE
    assert classify_result(0.75, 0.70) == ACCEPTABLE and classify_result(0.749, 0.99) == BELOW_TARGET
    assert classify_result(0.99, 0.69) == BELOW_TARGET and classify_result(0.0, 0.0) == BELOW_TARGET


def test_final_test_refuses_tampered_state_before_touching_any_test_image(world, frozen, monkeypatch):
    tampered = world.tmp / "tampered"
    tampered.mkdir()
    for f in frozen.dir.iterdir():
        (tampered / f.name).write_bytes(f.read_bytes())
    (tampered / "model_state.pt").write_bytes(b"corrupted after freezing")

    def boom(*a, **k):
        raise AssertionError("test images were listed before the frozen model was verified")

    monkeypatch.setattr(final_module, "discover_test_samples", boom)
    with pytest.raises(RuntimeError, match="changed after freezing"):
        evaluate_frozen_on_final_test(CATEGORY, tampered)
    assert not final_result_path(CATEGORY).exists()


def test_final_test_runs_once_with_the_locked_threshold_and_a_clean_audit(world, frozen, monkeypatch):
    monkeypatch.setattr(final_module, "load_pretrained_resnet18", lambda: world.extractor)
    result = evaluate_frozen_on_final_test(CATEGORY, frozen.dir)

    assert result["counts"] == {"test_total": 10, "test_good": 4, "test_defective": 6}
    m = result["metrics"]
    assert m["true_negatives"] + m["false_positives"] == 4 and m["false_negatives"] + m["true_positives"] == 6
    assert result["threshold"]["value"] == frozen.model.threshold
    assert all(p["predicted_label"] == (1 if p["reconstruction_error"] > frozen.model.threshold else 0)
               for p in result["predictions"])
    assert result["classification"] == classify_result(m["recall"], m["f1_score"])
    assert set(result["recall_by_defect_type"]) == {"scratch", "stain"}
    assert len(result["detected_files"]) + len(result["missed_files"]) == 6
    audit = result["leakage_audit"]
    assert audit["all_passed"], {k: v for k, v in audit.items() if not v}
    for key in ("test_disjoint_from_training_by_content", "locked_threshold_rederived_from_validation_good_scores_only",
                "model_fitted_on_exactly_the_training_list", "training_and_validation_disjoint"):
        assert audit[key] is True
    assert final_result_path(CATEGORY).is_file()

    with pytest.raises(FileExistsError, match="exactly once"):
        evaluate_frozen_on_final_test(CATEGORY, frozen.dir)


def test_final_test_result_is_not_written_into_serving_and_serving_is_unchanged():
    assert set(SERVING_CONFIGS) == {"bottle"}
    assert math.isclose(SERVING_CONFIGS["bottle"].threshold, 0.0028031117030001018)
