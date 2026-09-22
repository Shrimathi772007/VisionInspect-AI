"""Controlled alternative-model-family study (app.ai.evaluation.model_family_study / _pipeline / _final).

Pure computations use small synthetic tensors/score dicts. The end-to-end tests use ONE module-scoped tiny synthetic
world (20 texture images, 8 candidates x 8 policies on 12 splits) built with the real, hash-verified ResNet-18 weights
copied into a temporary artifact root (skipped when the Git-ignored weights are absent). The real Leather experiment is
run by scripts/run_leather_model_family.py, not by the suite.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.ai.evaluation import model_family_final as final
from app.ai.evaluation import model_family_pipeline as pipe
from app.ai.evaluation import model_family_study as mfs
from app.ai.models.patch_anomaly import GaussianPatchScorer, KNNPatchScorer, aggregate_scores, extract_patch_features
from app.ai.models.resnet18 import RESNET18_FILENAME, RESNET18_SHA256, ResNet18, pretrained_weights_path
from app.ai.training import artifacts

CATEGORY = "widget"
EXPECTED = {"train_good": 20, "test_good": 6, "test_defective": 6, "test_total": 12}
REAL_WEIGHTS = pretrained_weights_path()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_candidate_registry_is_exactly_the_declared_eight():
    specs = mfs.registry()
    assert len(specs) == mfs.EXPECTED_CANDIDATES == 8
    assert [(s.candidate_id, s.family, s.layers, s.image_size) for s in specs] == [
        ("gauss_l2_128", "gaussian", (2,), 128), ("gauss_l3_128", "gaussian", (3,), 128),
        ("gauss_l23_128", "gaussian", (2, 3), 128), ("gauss_l23_256", "gaussian", (2, 3), 256),
        ("knn_l2_128", "knn", (2,), 128), ("knn_l3_128", "knn", (3,), 128),
        ("knn_l23_128", "knn", (2, 3), 128), ("knn_l23_256", "knn", (2, 3), 256)]
    assert all(s.pooling == "avg3x3_stride1_pad1" and s.aggregation == "top1pct_mean" for s in specs)
    assert {s.shrinkage for s in specs if s.family == "gaussian"} == {1e-2} and {s.coreset_fraction for s in specs if s.family == "knn"} == {0.10}
    assert [s.feature_dim for s in specs] == [128, 256, 384, 384, 128, 256, 384, 384]
    assert mfs.registry_fingerprint() == mfs.registry_fingerprint(mfs.registry())
    assert mfs.registry_fingerprint(specs[:7]) != mfs.registry_fingerprint()  # a missing candidate changes the fingerprint


def test_threshold_policies_are_the_declared_eight_and_deterministic():
    assert mfs.policy_ids() == ["mean_std_1.5", "mean_std_2", "mean_std_2.5", "mean_std_3",
                                "percentile_95", "percentile_97", "percentile_98", "percentile_99"]
    x = np.arange(1.0, 101.0)
    assert mfs.policy_threshold("mean_std_2", x) == pytest.approx(x.mean() + 2 * x.std(ddof=0))
    assert mfs.policy_threshold("percentile_98", x) == pytest.approx(np.percentile(x, 98))
    assert len(mfs.registry()) * len(mfs.policy_ids()) == 64


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_normal_splits_are_deterministic_disjoint_and_cover_the_declared_schemes():
    a, b = mfs.scheme_splits(245), mfs.scheme_splits(245)
    assert set(a) == {"holdout_80_20", "random_kfold5_seed42", "random_kfold5_seed43", "random_kfold5_seed44", "contiguous_kfold5"}
    assert len(a["holdout_80_20"]) == 10 and [s["name"] for s in a["holdout_80_20"]] == [f"seed{i}" for i in range(42, 52)]
    assert sum(len(v) for v in a.values()) == 30 and mfs.splits_fingerprint(a) == mfs.splits_fingerprint(b)
    audit = mfs.split_audit(245, a)
    assert audit["calibration_and_held_out_disjoint_in_every_split"] and audit["every_split_covers_the_pool"]
    assert audit["kfold_held_out_folds_partition_the_pool"] and audit["held_out_sizes"] == [49] and audit["calibration_sizes"] == [196]
    blocks = a["contiguous_kfold5"]
    assert [s["held_out"].tolist() for s in blocks] == [list(range(i * 49, (i + 1) * 49)) for i in range(5)]  # sorted, contiguous
    assert (a["random_kfold5_seed42"][0]["held_out"] != a["random_kfold5_seed43"][0]["held_out"]).any()
    assert a["holdout_80_20"][0]["held_out"].tolist() == sorted(np.random.default_rng(42).permutation(245)[:49].tolist())


# ---------------------------------------------------------------------------
# Feature extraction, Gaussian and memory-bank fitting
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_extractor():
    torch.manual_seed(0)
    model = ResNet18().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_feature_extraction_is_deterministic_with_the_documented_shapes(tiny_extractor):
    images = torch.rand(3, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    for layers, shape in (((2,), (3, 16, 16, 128)), ((3,), (3, 8, 8, 256)), ((2, 3), (3, 16, 16, 384))):
        a = extract_patch_features(tiny_extractor, images, layers)
        b = extract_patch_features(tiny_extractor, images, layers)
        assert tuple(a.shape) == shape and torch.equal(a, b)
    assert tuple(extract_patch_features(tiny_extractor, torch.rand(1, 3, 256, 256), (2, 3)).shape) == (1, 32, 32, 384)


def test_gaussian_split_scores_match_a_direct_fit_leave_one_image_out():
    g = torch.Generator().manual_seed(3)
    patches = torch.randn(8, 3, 3, 6, generator=g) + torch.arange(6.0) * 0.1
    sums, outers = mfs.image_statistics(patches)
    cal, held = np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])
    loio, held_scores = mfs.gaussian_split_scores(patches, sums, outers, cal, held)
    full = GaussianPatchScorer.fit(patches[torch.as_tensor(cal)])
    expected_held = aggregate_scores(full.patch_scores(patches[torch.as_tensor(held)]), "top1pct_mean").double().numpy()
    np.testing.assert_allclose(held_scores, expected_held, rtol=1e-3)
    for k, i in enumerate(cal):
        rest = torch.as_tensor([j for j in cal if j != i])
        direct = aggregate_scores(GaussianPatchScorer.fit(patches[rest]).patch_scores(patches[int(i) : int(i) + 1]), "top1pct_mean").item()
        assert loio[k] == pytest.approx(direct, rel=1e-3)


def test_memory_bank_scores_match_a_direct_nearest_neighbour_bank():
    g = torch.Generator().manual_seed(4)
    patches = torch.randn(7, 4, 4, 8, generator=g)
    banks = mfs.per_image_coresets(patches, 0.25, 42)
    assert all(b.shape == (4, 8) for b in banks)  # round(16 * 0.25) coreset patches per image
    assert all(torch.equal(x, y) for x, y in zip(banks, mfs.per_image_coresets(patches, 0.25, 42)))  # deterministic
    minima = mfs.pairwise_minima(patches, banks)
    cal, held = np.array([0, 1, 2, 3, 4]), np.array([5, 6])
    loio, held_scores = mfs.knn_split_scores(minima, 4, 4, cal, held)
    scorer = mfs.knn_bank(banks, cal)
    direct = aggregate_scores(scorer.patch_scores(patches[torch.as_tensor(held)]), "top1pct_mean").double().numpy()
    np.testing.assert_allclose(held_scores, direct, atol=2e-2)
    for k, i in enumerate(cal):  # leave-one-image-out: the image's own coreset is excluded from its bank
        rest = mfs.knn_bank(banks, [j for j in cal if j != i])
        d = aggregate_scores(rest.patch_scores(patches[int(i) : int(i) + 1]), "top1pct_mean").item()
        assert loio[k] == pytest.approx(d, abs=2e-2)
        assert isinstance(KNNPatchScorer(banks[0]).bank, torch.Tensor)


# ---------------------------------------------------------------------------
# The pre-declared selection rule (pure)
# ---------------------------------------------------------------------------

def _entry(worst, recall, cv, thr=1.0):
    return {"worst_fold_fpr": worst, "mean_fpr": worst / 2, "synthetic_recall": recall, "pooled_threshold": {"cv": cv}, "full_pool_threshold": thr}


def _results(mapping):
    """mapping: {(candidate, policy): entry}"""
    out: dict = {}
    for (cid, policy), e in mapping.items():
        out.setdefault(cid, {"policies": {}, "synthetic_auroc": 0.9})["policies"][policy] = e
    return out


SPECS = {s.candidate_id: s for s in mfs.registry()}


def test_gate_rejects_configurations_above_ten_percent_worst_fold_fpr():
    res = _results({("gauss_l23_256", "mean_std_3"): _entry(0.0204, 0.9, 0.02), ("gauss_l23_256", "mean_std_1.5"): _entry(0.10000001, 0.99, 0.01)})
    sel = mfs.select_configuration(res, mfs.registry())
    assert sel["winner"] == "gauss_l23_256|mean_std_3" and sel["gate_passers"] == ["gauss_l23_256|mean_std_3"]
    assert mfs.select_configuration(_results({("gauss_l23_256", "mean_std_3"): _entry(0.10, 0.9, 0.02)}), mfs.registry())["winner"] is not None


def test_no_configuration_passing_the_gate_locks_nothing():
    sel = mfs.select_configuration(_results({("knn_l2_128", "percentile_99"): _entry(0.30, 0.9, 0.02)}), mfs.registry())
    assert sel["winner"] is None and sel["message"] == "No candidate satisfied the predeclared normal-data robustness gate."


def test_preferred_synthetic_floor_filters_when_possible_and_falls_back_otherwise():
    res = _results({("gauss_l2_128", "mean_std_3"): _entry(0.0, 0.50, 0.01), ("gauss_l3_128", "mean_std_3"): _entry(0.05, 0.80, 0.01)})
    sel = mfs.select_configuration(res, mfs.registry())
    assert sel["winner_candidate"] == "gauss_l3_128" and sel["preferred_floor_met"]  # the zero-FPR one misses the 0.75 floor
    res = _results({("gauss_l2_128", "mean_std_3"): _entry(0.0, 0.50, 0.01), ("gauss_l3_128", "mean_std_3"): _entry(0.05, 0.60, 0.01)})
    sel = mfs.select_configuration(res, mfs.registry())
    assert sel["winner_candidate"] == "gauss_l2_128" and not sel["preferred_floor_met"]


def test_ordering_fpr_then_recall_then_cv_then_simplicity_then_id():
    # 1) lower worst-fold FPR wins outright (exact comparison)
    sel = mfs.select_configuration(_results({("gauss_l2_128", "mean_std_3"): _entry(0.0408, 0.99, 0.001), ("gauss_l3_128", "mean_std_3"): _entry(0.0204, 0.80, 0.09)}), mfs.registry())
    assert sel["winner_candidate"] == "gauss_l3_128"
    # 2) equal FPR: higher synthetic recall wins when the gap exceeds the 0.03 tie band
    sel = mfs.select_configuration(_results({("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.80, 0.001), ("gauss_l3_128", "mean_std_3"): _entry(0.02, 0.90, 0.09)}), mfs.registry())
    assert sel["winner_candidate"] == "gauss_l3_128"
    # 3) recalls within 0.03 are tied -> the lower threshold CoV wins (gap above the 0.01 CoV band)
    sel = mfs.select_configuration(_results({("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.90, 0.05), ("gauss_l3_128", "mean_std_3"): _entry(0.02, 0.88, 0.02)}), mfs.registry())
    assert sel["winner_candidate"] == "gauss_l3_128"
    # 4) CoV within 0.01 -> the simpler model (Gaussian before memory bank, then smaller feature dimension)
    sel = mfs.select_configuration(_results({("knn_l2_128", "mean_std_3"): _entry(0.02, 0.90, 0.020), ("gauss_l23_256", "mean_std_3"): _entry(0.02, 0.90, 0.025)}), mfs.registry())
    assert sel["winner_candidate"] == "gauss_l23_256"
    sel = mfs.select_configuration(_results({("gauss_l23_128", "mean_std_3"): _entry(0.02, 0.90, 0.02), ("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.90, 0.02)}), mfs.registry())
    assert sel["winner_candidate"] == "gauss_l2_128"
    # 5) full tie -> candidate id, then policy id (deterministic)
    sel = mfs.select_configuration(_results({("gauss_l2_128", "percentile_99"): _entry(0.02, 0.9, 0.02), ("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.9, 0.02)}), mfs.registry())
    assert sel["winner"] == "gauss_l2_128|mean_std_3"


def test_selection_signature_has_no_place_for_final_test_data():
    import inspect

    assert list(inspect.signature(mfs.select_configuration).parameters) == ["results", "specs"]


# ---------------------------------------------------------------------------
# Static leakage protections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [mfs, pipe])
def test_study_modules_cannot_list_or_score_test_images(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "discover_test_samples" not in source
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not [m for m in imported if m.endswith("model_family_final") or "calibration_final_test" in m]


def test_final_module_only_scores_the_locked_candidate_and_threshold():
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert "select_configuration" in source and "evaluate_candidate_on_final_test" not in source and "candidate_results" not in source


def test_serving_registry_has_no_leather():
    serving = Path(mfs.__file__).parent.parent / "inference" / "serving.py"
    assert "leather" not in serving.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# End to end on one tiny synthetic world
# ---------------------------------------------------------------------------

def _texture(seed: int, patch: str | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(120, 30, (128, 128, 3)).clip(0, 255).astype(np.uint8), (3, 3), 0)
    if patch == "bright":
        img[30:90, 30:90] = 250
    elif patch == "dark":
        img[50:80, :] = 10
    return img


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


def _fake_prior_phases(art: Path) -> None:
    metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5, "false_defect_detection_rate": 0.9,
               "true_negatives": 1, "false_positives": 5, "false_negatives": 4, "true_positives": 4}
    dist = {"mean": 1.0, "std": 0.1, "median": 1.0, "min": 0.5, "max": 2.0}
    per_defect = {"scratch": {"recall": 0.5, "detected": 2, "total": 3}, "dent": {"recall": 0.25, "detected": 1, "total": 3}}
    p1, p2, p3 = (art / CATEGORY / d for d in ("phase1_baseline", "phase2_threshold", "phase3_validation"))
    for p in (p1, p2, p3):
        (p / "reports").mkdir(parents=True)
    (p1 / "autoencoder.pt").write_bytes(b"phase1-weights")
    (p1 / "reports" / "phase1_report.json").write_text(json.dumps({"run1": {
        "threshold": {"value": 0.001}, "metrics": metrics, "separation": {"auroc_descriptive_only": 0.51}, "per_defect": per_defect,
        "distributions": {"train_good": dist, "test_good": dist, "test_defective": dist}}}))
    preds = [{"label": 0, "reconstruction_error": 0.1}, {"label": 1, "reconstruction_error": 0.2}]
    (p2 / "reports" / "final_test_result.json").write_text(json.dumps({"locked_threshold": 0.001, "metrics": metrics, "per_defect": per_defect, "predictions": preds}))
    (p3 / "reports" / "final_test_result.json").write_text(json.dumps({"locked_threshold": 0.001, "metrics": metrics, "per_defect": per_defect, "auroc_descriptive_only": 0.43}))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if not REAL_WEIGHTS.is_file():
        pytest.skip("verified ResNet-18 weights not present")
    tmp = tmp_path_factory.mktemp("mf_world")
    root, art = tmp / "dataset", tmp / "ai_models"
    for i in range(20):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", _texture(i))
    for i in range(6):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", _texture(100 + i))
    for defect, patch in (("scratch", "bright"), ("dent", "dark")):
        for i in range(3):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", _texture(200 + i, patch))
    (art / "_pretrained").mkdir(parents=True)
    shutil.copy2(REAL_WEIGHTS, art / "_pretrained" / RESNET18_FILENAME)
    _fake_prior_phases(art)

    mp = pytest.MonkeyPatch()
    mp.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    mp.setattr(artifacts, "ARTIFACTS_ROOT", art)
    mp.setattr(mfs, "HOLDOUT_SEEDS", (42, 43))
    mp.setattr(mfs, "RANDOM_KFOLD_SEEDS", (42,))
    mp.setattr(mfs, "GATE_FPR", 0.30)  # 4 held-out images per split: one false positive = 25%

    import app.ai.preprocessing.pipeline as preprocessing

    decoded = []
    real_load = preprocessing._load_image
    mp.setattr(pipe, "_load_image", lambda p: (decoded.append(Path(p)), real_load(p))[1])

    inputs = pipe.load_inputs(CATEGORY, EXPECTED)
    study = json.loads(json.dumps(pipe.run_study(inputs, pipe.candidates_dir(CATEGORY), log=lambda m: None), sort_keys=True))
    pipe.reports_dir(CATEGORY).mkdir(parents=True, exist_ok=True)
    pipe.study_report_path(CATEGORY).write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    try:
        yield {"tmp": tmp, "root": root, "art": art, "inputs": inputs, "study": study, "decoded": decoded, "weights": art / "_pretrained" / RESNET18_FILENAME}
    finally:
        mp.undo()


@pytest.fixture
def sandbox(world, tmp_path, monkeypatch):
    """A private copy of the artifact root (study, candidates, weights, fake prior phases) for mutating tests."""
    art = tmp_path / "ai_models"
    shutil.copytree(world["art"], art)
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", art)
    return art


def test_study_covers_all_eight_candidates_by_eight_policies(world):
    study = world["study"]
    assert sorted(study["candidates"]) == sorted(s.candidate_id for s in mfs.registry()) and len(study["selection"]["rows"]) == 64
    assert study["registry_fingerprint"] == mfs.registry_fingerprint() and study["policies"] == mfs.policy_ids()
    assert study["synthetic"]["label"].startswith("DIAGNOSTIC ONLY") and study["synthetic"]["images"] == 4 * 15
    for cid, r in study["candidates"].items():
        s = r["summary"]
        assert set(s["policies"]) == set(mfs.policy_ids()) and 0.0 <= s["synthetic_auroc"] <= 1.0
        for e in s["policies"].values():
            assert 0.0 <= e["worst_fold_fpr"] <= 1.0 and e["pooled_threshold"]["std"] >= 0 and "synthetic_recall" in e
            assert e["full_pool_threshold"] > 0
    assert all(study["split_audit"][k] for k in ("calibration_and_held_out_disjoint_in_every_split", "every_split_covers_the_pool",
                                                 "kfold_held_out_folds_partition_the_pool"))


def test_no_final_test_image_is_decoded_before_the_lock(world):
    assert world["decoded"] and not [p for p in world["decoded"] if "test" in p.parts]
    assert {p.parent.name for p in world["decoded"]} == {"good"}  # only train/good was ever decoded
    assert world["study"]["counts"]["test_total"] == 12 and set(world["study"]["counts"]["test_by_type"]) == {"dent", "good", "scratch"}


def test_dataset_count_mismatch_stops_before_anything_is_fitted(world):
    with pytest.raises(ValueError, match="train_good"):
        pipe.count_dataset_entries(CATEGORY, {**EXPECTED, "train_good": 245})


def test_every_candidate_artifact_has_config_hash_fingerprint_features_and_thresholds(world):
    for cid in (s.candidate_id for s in mfs.registry()):
        d = pipe.candidates_dir(CATEGORY) / cid
        manifest = json.loads((d / "candidate_manifest.json").read_text())
        state = d / "model_state.pt"
        assert manifest["artifact"]["sha256"] == hashlib.sha256(state.read_bytes()).hexdigest() == world["study"]["candidates"][cid]["artifact"]["sha256"]
        assert manifest["training_data_fingerprint"] == world["study"]["dataset_fingerprint_sha256"] and manifest["training_image_count"] == 20
        assert manifest["backbone_sha256"] == RESNET18_SHA256 and set(manifest["thresholds_full_pool"]) == set(mfs.policy_ids())
        assert manifest["feature_config"]["feature_dim"] == manifest["config"]["feature_dim"]
        assert json.loads((d / "model_config.json").read_text())["scorer"] in ("knn", "gaussian")
    assert pipe.backbone_info()["sha256"] == RESNET18_SHA256


def test_study_is_reproducible_across_two_complete_runs(world, tmp_path):
    second = json.loads(json.dumps(pipe.run_study(pipe.load_inputs(CATEGORY, EXPECTED), tmp_path / "candidates", log=lambda m: None), sort_keys=True))
    repro = pipe.compare_studies(world["study"], second)
    assert all(repro.values()), repro
    tampered = json.loads(json.dumps(second))
    first_id = next(iter(tampered["candidates"]))
    tampered["candidates"][first_id]["artifact"]["sha256"] = "0" * 64
    assert not pipe.compare_studies(world["study"], tampered)["candidate_artifact_hashes_identical"]


def _lock(sandbox):
    study = json.loads(pipe.study_report_path(CATEGORY).read_text())
    outcome = pipe.create_selection_lock(CATEGORY, study)
    assert outcome["locked"], study["selection"]["reasoning"]
    return outcome["lock"]


def test_selection_lock_contents_hashes_and_artifact_isolation(world, sandbox):
    before = pipe.prior_phase_reference(CATEGORY)
    lock = _lock(sandbox)
    for key in ("category", "experiment_id", "candidate_list", "candidate_configurations", "backbone", "normal_dataset_fingerprint_sha256",
                "split_fingerprint_sha256", "threshold_candidates", "normal_robustness_results", "synthetic_diagnostic_results", "selection_rule",
                "selected_candidate", "selected_threshold", "selected_artifact", "timestamp_utc", "lock_digest"):
        assert key in lock, key
    assert lock["category"] == CATEGORY and len(lock["candidate_list"]) == 8 and lock["backbone"]["sha256"] == RESNET18_SHA256
    assert lock["synthetic_diagnostic_results"]["label"] == "DIAGNOSTIC ONLY" and lock["final_test_data_used_for_selection"] is False
    selected = pipe.selected_dir(CATEGORY) / "model_state.pt"
    assert lock["selected_artifact"]["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert selected.read_bytes() == (pipe.candidates_dir(CATEGORY) / lock["selected_candidate"] / "model_state.pt").read_bytes()
    assert pipe.lock_digest(lock) == lock["lock_digest"] and pipe.study_root(CATEGORY).name == "model_family_study"
    assert pipe.prior_phase_reference(CATEGORY) == before == lock["prior_phase_artifacts"]  # isolated from Phase 1/2/3
    with pytest.raises(FileExistsError):
        _lock(sandbox)


def test_no_gate_passer_writes_no_lock_and_no_selected_candidate(world, sandbox):
    study = json.loads(pipe.study_report_path(CATEGORY).read_text())
    study["selection"] = {**study["selection"], "winner": None, "message": mfs.NO_CANDIDATE_MESSAGE}
    outcome = pipe.create_selection_lock(CATEGORY, study)
    assert outcome["locked"] is False and not pipe.lock_path(CATEGORY).exists() and not pipe.selected_dir(CATEGORY).exists()


def test_final_test_runs_once_after_the_lock_with_a_clean_leakage_audit(world, sandbox, monkeypatch):
    lock = _lock(sandbox)
    seen = {}
    real = final.discover_test_samples

    def spy(category):
        seen["lock_on_disk"] = pipe.lock_path(category).is_file()
        seen["result_existed"] = final.result_path(category).exists()
        return real(category)

    monkeypatch.setattr(final, "discover_test_samples", spy)
    result = final.evaluate_locked_on_final_test(CATEGORY)
    assert seen == {"lock_on_disk": True, "result_existed": False}
    assert result["selected_candidate"] == lock["selected_candidate"] and result["locked_threshold"] == lock["selected_threshold"]
    m = result["metrics"]
    assert m["true_positives"] + m["false_negatives"] == 6 and m["true_negatives"] + m["false_positives"] == 6
    assert m["defect_identification_accuracy"] == m["recall"] and 0.0 <= result["auroc"] <= 1.0
    for p in result["predictions"]:
        assert p["predicted_label"] == int(p["reconstruction_error"] > lock["selected_threshold"])
    assert result["leakage_audit"]["all_passed"], result["leakage_audit"]
    assert set(result["comparison"]) >= {"phase1_convae", "phase2_convae", "phase3_convae", "alternative", "per_defect"}
    assert result["gate_classification"] in {"EXCELLENT", "GOOD", "ACCEPTABLE", "NOT PRODUCTION READY"}
    assert final.verify_lock_unchanged_since_final_test(CATEGORY)
    with pytest.raises(FileExistsError):  # exactly once
        final.evaluate_locked_on_final_test(CATEGORY)
    # the lock cannot be modified after the final test without detection
    text = pipe.lock_path(CATEGORY).read_text()
    pipe.lock_path(CATEGORY).write_text(text.replace('"category"', '"category" ', 1))
    assert not final.verify_lock_unchanged_since_final_test(CATEGORY)


def _forbid_test_access(monkeypatch):
    monkeypatch.setattr(final, "discover_test_samples", lambda c: (_ for _ in ()).throw(AssertionError("test set touched")))


def test_tampered_candidate_artifact_is_rejected_before_the_test_set_is_touched(world, sandbox, monkeypatch):
    _lock(sandbox)
    with open(pipe.selected_dir(CATEGORY) / "model_state.pt", "ab") as handle:
        handle.write(b"tamper")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="artifact no longer matches"):
        final.evaluate_locked_on_final_test(CATEGORY)
    assert not final.result_path(CATEGORY).exists()


def test_tampered_backbone_is_rejected(world, sandbox, monkeypatch):
    _lock(sandbox)
    with open(sandbox / "_pretrained" / RESNET18_FILENAME, "ab") as handle:
        handle.write(b"tamper")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="backbone"):
        final.evaluate_locked_on_final_test(CATEGORY)


def test_tampered_selection_lock_is_rejected_in_three_ways(world, sandbox, monkeypatch):
    _lock(sandbox)
    _forbid_test_access(monkeypatch)
    path = pipe.lock_path(CATEGORY)
    lock = json.loads(path.read_text())

    edited = {**lock, "selected_threshold": lock["selected_threshold"] * 2}  # 1) stale digest
    path.write_text(json.dumps(edited))
    with pytest.raises(RuntimeError, match="does not match its digest"):
        final.evaluate_locked_on_final_test(CATEGORY)

    edited["lock_digest"] = pipe.lock_digest(edited)  # 2) digest recomputed: the saved study no longer selects this threshold
    path.write_text(json.dumps(edited))
    with pytest.raises(RuntimeError, match="not what the pre-declared rule selects"):
        final.evaluate_locked_on_final_test(CATEGORY)

    short = {**lock, "candidate_list": lock["candidate_list"][:-1]}  # 3) a missing candidate, digest recomputed
    short["lock_digest"] = pipe.lock_digest(short)
    path.write_text(json.dumps(short))
    with pytest.raises(RuntimeError, match="Candidate count differs"):
        final.evaluate_locked_on_final_test(CATEGORY)
    assert not final.result_path(CATEGORY).exists()


def test_modified_prior_phase_artifact_and_changed_dataset_are_rejected(world, sandbox, monkeypatch):
    _lock(sandbox)
    _forbid_test_access(monkeypatch)
    (sandbox / CATEGORY / "phase1_baseline" / "autoencoder.pt").write_bytes(b"overwritten")
    with pytest.raises(RuntimeError, match="Phase 1/2/3 artifact"):
        final.evaluate_locked_on_final_test(CATEGORY)
    (sandbox / CATEGORY / "phase1_baseline" / "autoencoder.pt").write_bytes(b"phase1-weights")
    _write(world["root"] / CATEGORY / "train" / "good" / "000.png", _texture(999))
    try:
        with pytest.raises(RuntimeError, match="dataset no longer matches"):
            final.evaluate_locked_on_final_test(CATEGORY)
    finally:
        _write(world["root"] / CATEGORY / "train" / "good" / "000.png", _texture(0))
