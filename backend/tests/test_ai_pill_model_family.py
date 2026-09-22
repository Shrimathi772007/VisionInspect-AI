"""Pill alternative model-family study (app.ai.evaluation.model_family_study / _pipeline / _final applied to `pill`).

Real-dataset tests (skipped when the Git-ignored MVTec files are absent) verify counts, defect types, fingerprints and the
267-image split geometry without fitting anything. Everything that fits uses ONE module-scoped tiny synthetic `pill`
world (20 texture images, 8 candidates x 8 policies on 12 splits) built with the real, hash-verified ResNet-18 weights
copied into a temporary artifact root (skipped when the weights are absent). The real experiment is run by
scripts/run_model_family_study.py, not by the suite.
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
from app.ai.evaluation.category_phase1 import file_hashes
from app.ai.models.patch_anomaly import GaussianPatchScorer, aggregate_scores, extract_patch_features
from app.ai.models.resnet18 import RESNET18_FILENAME, RESNET18_SHA256, ResNet18, pretrained_weights_path
from app.ai.training import artifacts, discover_test_samples, discover_train_samples
from app.ai.training.dataset import DATASET_ROOT

CATEGORY = "pill"
REAL_EXPECTED = {"train_good": 267, "test_good": 26, "test_defective": 141, "test_total": 167}
REAL_TYPES = {"color": 25, "combined": 17, "contamination": 21, "crack": 26, "faulty_imprint": 19, "good": 26, "pill_type": 9, "scratch": 24}
DEFECT_NAMES = ("color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch")
TINY_EXPECTED = {"train_good": 20, "test_good": 4, "test_defective": 14, "test_total": 18}
TINY_TYPES = {"good": 4, **{name: 2 for name in DEFECT_NAMES}}
REAL_WEIGHTS = pretrained_weights_path()
needs_dataset = pytest.mark.skipif(not (DATASET_ROOT / CATEGORY / "train" / "good").is_dir(), reason="MVTec pill not present")


# ---------------------------------------------------------------------------
# Registry, policies, splits
# ---------------------------------------------------------------------------

def test_exact_eight_candidate_registry():
    specs = mfs.registry()
    assert [s.candidate_id for s in specs] == ["gauss_l2_128", "gauss_l3_128", "gauss_l23_128", "gauss_l23_256",
                                               "knn_l2_128", "knn_l3_128", "knn_l23_128", "knn_l23_256"]
    assert [(s.family, s.layers, s.image_size, s.feature_dim) for s in specs] == [
        ("gaussian", (2,), 128, 128), ("gaussian", (3,), 128, 256), ("gaussian", (2, 3), 128, 384), ("gaussian", (2, 3), 256, 384),
        ("knn", (2,), 128, 128), ("knn", (3,), 128, 256), ("knn", (2, 3), 128, 384), ("knn", (2, 3), 256, 384)]
    assert all(s.pooling == "avg3x3_stride1_pad1" and s.aggregation == "top1pct_mean" for s in specs)
    assert {s.shrinkage for s in specs if s.family == "gaussian"} == {1e-2} and {s.coreset_fraction for s in specs if s.family == "knn"} == {0.10}
    assert mfs.registry_fingerprint(specs[:7]) != mfs.registry_fingerprint() != mfs.registry_fingerprint(specs + specs[:1])


def test_exact_eight_threshold_policies():
    assert mfs.policy_ids() == ["mean_std_1.5", "mean_std_2", "mean_std_2.5", "mean_std_3",
                                "percentile_95", "percentile_97", "percentile_98", "percentile_99"]
    x = np.arange(1.0, 101.0)
    assert mfs.policy_threshold("mean_std_3", x) == pytest.approx(x.mean() + 3 * x.std(ddof=0))
    assert mfs.policy_threshold("percentile_97", x) == pytest.approx(np.percentile(x, 97))
    assert len(mfs.registry()) * len(mfs.policy_ids()) == 64


def test_pill_splits_are_30_deterministic_disjoint_splits_with_derived_sizes():
    a, b = mfs.scheme_splits(267), mfs.scheme_splits(267)
    assert sum(len(v) for v in a.values()) == 30 == 15 + 5 + 10 and mfs.splits_fingerprint(a) == mfs.splits_fingerprint(b)
    assert [s["name"] for s in a["holdout_80_20"]] == [f"seed{i}" for i in range(42, 52)]
    assert sorted(a) == ["contiguous_kfold5", "holdout_80_20", "random_kfold5_seed42", "random_kfold5_seed43", "random_kfold5_seed44"]
    audit = mfs.split_audit(267, a)
    assert audit["calibration_and_held_out_disjoint_in_every_split"] and audit["every_split_covers_the_pool"] and audit["kfold_held_out_folds_partition_the_pool"]
    # the established rule: repeated 80/20 holds out round(0.2 * 267) = 53; 5 does not divide 267, so folds hold out 54 or 53
    assert {(len(s["calibration"]), len(s["held_out"])) for s in a["holdout_80_20"]} == {(214, 53)}
    for name in ("random_kfold5_seed42", "random_kfold5_seed43", "random_kfold5_seed44", "contiguous_kfold5"):
        assert sorted(len(s["held_out"]) for s in a[name]) == [53, 53, 53, 54, 54]
        assert all(len(s["calibration"]) + len(s["held_out"]) == 267 for s in a[name])
    assert audit["held_out_sizes"] == [53, 54] and audit["calibration_sizes"] == [213, 214]
    assert [len(s["held_out"]) for s in a["contiguous_kfold5"]] == [54, 54, 53, 53, 53]
    assert a["holdout_80_20"][0]["held_out"].tolist() == sorted(np.random.default_rng(42).permutation(267)[:53].tolist())
    assert mfs.splits_fingerprint(a) not in (mfs.splits_fingerprint(mfs.scheme_splits(245)), mfs.splits_fingerprint(mfs.scheme_splits(220)))


def test_pill_synthetic_diagnostic_size_is_derived_from_the_canonical_held_out_split_times_15():
    canonical = next(s for s in mfs.scheme_splits(267)[mfs.CANONICAL_SPLIT[0]] if s["name"] == mfs.CANONICAL_SPLIT[1])
    held_out, variants = len(canonical["held_out"]), len(mfs.DEFECT_TYPES) * len(mfs.SEVERITIES)
    assert (len(canonical["calibration"]), held_out, variants) == (214, 53, 15)
    # 795, not the 810 (= 54 x 15) the study brief expected: 54 would need ceil(0.2 * n), which is NOT the established rule
    assert held_out * variants == 795 != 54 * variants
    # no other category's size is inherited: Leather 49 x 15 = 735, Metal Nut 44 x 15 = 660
    sizes = {n: len(mfs.scheme_splits(n)[mfs.CANONICAL_SPLIT[0]][0]["held_out"]) * variants for n in (245, 220, 267)}
    assert sizes == {245: 735, 220: 660, 267: 795}
    declared = mfs.__doc__
    assert "held_out_images x 15" in declared and "Pill: n = 267" in declared and "53 x 15 = 795" in declared
    assert "Metal Nut: 44 x 15 = 660" in declared and "Leather: 49 x 15 = 735" in declared and "= 735 synthetic defects" not in declared


def test_pill_expected_counts_and_defect_types_are_pinned_in_the_cli():
    script = Path(mfs.__file__).parents[3] / "scripts" / "run_model_family_study.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "EXPECTED_BY_CATEGORY" for t in n.targets))
    assert table["pill"] == {**REAL_EXPECTED, "test_by_type": REAL_TYPES}
    assert table["leather"]["train_good"] == 245 and table["metal_nut"]["train_good"] == 220  # earlier categories untouched
    assert sum(v for k, v in REAL_TYPES.items() if k != "good") == 141 and sum(REAL_TYPES.values()) == 167


# ---------------------------------------------------------------------------
# Real dataset (no fitting)
# ---------------------------------------------------------------------------

@needs_dataset
def test_real_directory_counts_and_defect_types():
    counts = pipe.count_dataset_entries(CATEGORY, REAL_EXPECTED)
    assert (counts["train_good"], counts["test_good"], counts["test_defective"], counts["test_total"]) == (267, 26, 141, 167)
    assert counts["test_by_type"] == REAL_TYPES and set(counts["test_by_type"]) - {"good"} == set(DEFECT_NAMES)
    with pytest.raises(ValueError, match="test_defective"):
        pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_defective": 140})
    with pytest.raises(ValueError, match="test_by_type"):  # an unexpected or missing defect folder stops the study
        pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_by_type": {**REAL_TYPES, "crack": 27}})
    assert pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_by_type": REAL_TYPES}) == counts


@needs_dataset
def test_real_training_pool_is_sorted_train_good_only_with_a_stable_fingerprint():
    samples = discover_train_samples(CATEGORY)
    assert len(samples) == 267 and [s.path.name for s in samples] == sorted(s.path.name for s in samples)
    assert all(s.split == "train" and s.defect_type == "good" for s in samples)
    counts = pipe.count_dataset_entries(CATEGORY, REAL_EXPECTED)
    assert pipe.dataset_fingerprint(samples, counts) == pipe.dataset_fingerprint(discover_train_samples(CATEGORY), counts)
    assert len(discover_test_samples(CATEGORY)) == 167


@needs_dataset
def test_real_backbone_matches_the_verified_hash_and_size():
    if not REAL_WEIGHTS.is_file():
        pytest.skip("weights absent")
    info = pipe.backbone_info()
    assert info["sha256"] == RESNET18_SHA256 == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    assert info["size_bytes"] == 46_830_571 and info["fine_tuned"] is False


# ---------------------------------------------------------------------------
# Features, Gaussian, memory bank
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_extractor():
    torch.manual_seed(0)
    model = ResNet18().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_feature_extraction_is_deterministic_with_documented_shapes(tiny_extractor):
    images = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    for layers, shape in (((2,), (2, 16, 16, 128)), ((3,), (2, 8, 8, 256)), ((2, 3), (2, 16, 16, 384))):
        a, b = extract_patch_features(tiny_extractor, images, layers), extract_patch_features(tiny_extractor, images, layers)
        assert tuple(a.shape) == shape and torch.equal(a, b)
    assert tuple(extract_patch_features(tiny_extractor, torch.rand(1, 3, 256, 256), (2, 3)).shape) == (1, 32, 32, 384)


def test_gaussian_fit_and_leave_one_image_out_match_a_direct_fit():
    patches = torch.randn(8, 3, 3, 6, generator=torch.Generator().manual_seed(3)) + torch.arange(6.0) * 0.1
    sums, outers = mfs.image_statistics(patches)
    cal, held = np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])
    loio, held_scores = mfs.gaussian_split_scores(patches, sums, outers, cal, held)
    full = GaussianPatchScorer.fit(patches[torch.as_tensor(cal)])
    np.testing.assert_allclose(held_scores, aggregate_scores(full.patch_scores(patches[torch.as_tensor(held)]), "top1pct_mean").double().numpy(), rtol=1e-3)
    for k, i in enumerate(cal):
        rest = torch.as_tensor([j for j in cal if j != i])
        direct = aggregate_scores(GaussianPatchScorer.fit(patches[rest]).patch_scores(patches[int(i) : int(i) + 1]), "top1pct_mean").item()
        assert loio[k] == pytest.approx(direct, rel=1e-3)


def test_memory_bank_construction_and_scores_match_a_direct_nearest_neighbour_bank():
    patches = torch.randn(7, 4, 4, 8, generator=torch.Generator().manual_seed(4))
    banks = mfs.per_image_coresets(patches, 0.25, 42)
    assert all(b.shape == (4, 8) for b in banks) and all(torch.equal(x, y) for x, y in zip(banks, mfs.per_image_coresets(patches, 0.25, 42)))
    minima = mfs.pairwise_minima(patches, banks)
    assert torch.isinf(minima[range(7), range(7)]).all()  # an image's own coreset is excluded from its own score
    cal, held = np.array([0, 1, 2, 3, 4]), np.array([5, 6])
    loio, held_scores = mfs.knn_split_scores(minima, 4, 4, cal, held)
    direct = aggregate_scores(mfs.knn_bank(banks, cal).patch_scores(patches[torch.as_tensor(held)]), "top1pct_mean").double().numpy()
    np.testing.assert_allclose(held_scores, direct, atol=2e-2)
    for k, i in enumerate(cal):
        rest = mfs.knn_bank(banks, [j for j in cal if j != i])
        assert loio[k] == pytest.approx(aggregate_scores(rest.patch_scores(patches[int(i) : int(i) + 1]), "top1pct_mean").item(), abs=2e-2)


# ---------------------------------------------------------------------------
# The pre-declared selection rule (pure)
# ---------------------------------------------------------------------------

def _entry(worst, recall, cv, thr=1.0):
    return {"worst_fold_fpr": worst, "mean_fpr": worst / 2, "synthetic_recall": recall, "pooled_threshold": {"cv": cv}, "full_pool_threshold": thr}


def _results(mapping):
    out: dict = {}
    for (cid, policy), e in mapping.items():
        out.setdefault(cid, {"policies": {}, "synthetic_auroc": 0.9})["policies"][policy] = e
    return out


def test_gate_is_worst_fold_fpr_at_most_ten_percent():
    res = _results({("gauss_l23_256", "mean_std_3"): _entry(0.0204, 0.9, 0.02), ("gauss_l23_256", "mean_std_1.5"): _entry(0.10000001, 0.99, 0.01)})
    assert mfs.select_configuration(res, mfs.registry())["winner"] == "gauss_l23_256|mean_std_3"
    assert mfs.select_configuration(_results({("gauss_l2_128", "mean_std_3"): _entry(0.10, 0.9, 0.02)}), mfs.registry())["winner"] is not None
    none = mfs.select_configuration(_results({("knn_l2_128", "percentile_99"): _entry(0.30, 0.9, 0.02)}), mfs.registry())
    assert none["winner"] is None and none["message"] == "No candidate satisfied the predeclared normal-data robustness gate."


def test_synthetic_floor_is_preferred_and_falls_back_when_unmet():
    res = _results({("gauss_l2_128", "mean_std_3"): _entry(0.0, 0.50, 0.01), ("gauss_l3_128", "mean_std_3"): _entry(0.05, 0.80, 0.01)})
    sel = mfs.select_configuration(res, mfs.registry())
    assert sel["winner_candidate"] == "gauss_l3_128" and sel["preferred_floor_met"]
    res = _results({("gauss_l2_128", "mean_std_3"): _entry(0.0, 0.50, 0.01), ("gauss_l3_128", "mean_std_3"): _entry(0.05, 0.60, 0.01)})
    sel = mfs.select_configuration(res, mfs.registry())
    assert sel["winner_candidate"] == "gauss_l2_128" and not sel["preferred_floor_met"]


def test_ordering_fpr_recall_cv_simplicity_id():
    pick = lambda m: mfs.select_configuration(_results(m), mfs.registry())  # noqa: E731
    assert pick({("gauss_l2_128", "mean_std_3"): _entry(0.0408, 0.99, 0.001), ("gauss_l3_128", "mean_std_3"): _entry(0.0204, 0.80, 0.09)})["winner_candidate"] == "gauss_l3_128"
    assert pick({("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.80, 0.001), ("gauss_l3_128", "mean_std_3"): _entry(0.02, 0.90, 0.09)})["winner_candidate"] == "gauss_l3_128"
    assert pick({("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.90, 0.05), ("gauss_l3_128", "mean_std_3"): _entry(0.02, 0.88, 0.02)})["winner_candidate"] == "gauss_l3_128"
    assert pick({("knn_l2_128", "mean_std_3"): _entry(0.02, 0.90, 0.020), ("gauss_l23_256", "mean_std_3"): _entry(0.02, 0.90, 0.025)})["winner_candidate"] == "gauss_l23_256"
    assert pick({("gauss_l2_128", "percentile_99"): _entry(0.02, 0.9, 0.02), ("gauss_l2_128", "mean_std_3"): _entry(0.02, 0.9, 0.02)})["winner"] == "gauss_l2_128|mean_std_3"


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


def test_final_module_only_scores_the_locked_candidate_and_serving_has_no_pill():
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert "evaluate_candidate_on_final_test" not in source and "candidate_results" not in source
    serving = (Path(mfs.__file__).parent.parent / "inference" / "serving.py").read_text(encoding="utf-8").lower()
    assert "pill" not in serving.replace("pillow", "")


# ---------------------------------------------------------------------------
# End to end on one tiny synthetic pill world
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


def _fake_phase1(art: Path) -> None:
    metrics = {"accuracy": 0.2, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "false_defect_detection_rate": 0.0,
               "true_negatives": 4, "false_positives": 0, "false_negatives": 14, "true_positives": 0}
    per_defect = {t: {"recall": 0.0, "detected": 0, "total": 2} for t in DEFECT_NAMES}
    p1 = art / CATEGORY / "phase1_baseline"
    (p1 / "reports").mkdir(parents=True)
    (p1 / "autoencoder.pt").write_bytes(b"phase1-weights")
    (p1 / "reports" / "phase1_report.json").write_text(json.dumps({"run1": {
        "threshold": {"value": 0.0058}, "metrics": metrics, "separation": {"auroc_descriptive_only": 0.35}, "per_defect": per_defect}}))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if not REAL_WEIGHTS.is_file():
        pytest.skip("verified ResNet-18 weights not present")
    tmp = tmp_path_factory.mktemp("pill_world")
    root, art = tmp / "dataset", tmp / "ai_models"
    for i in range(20):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", _texture(i))
    for i in range(4):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", _texture(100 + i))
    for k, defect in enumerate(DEFECT_NAMES):
        for i in range(2):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", _texture(200 + 10 * k + i, "bright" if k % 2 == 0 else "dark"))
    (art / "_pretrained").mkdir(parents=True)
    shutil.copy2(REAL_WEIGHTS, art / "_pretrained" / RESNET18_FILENAME)
    _fake_phase1(art)

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
    inputs = pipe.load_inputs(CATEGORY, TINY_EXPECTED)
    study = json.loads(json.dumps(pipe.run_study(inputs, pipe.candidates_dir(CATEGORY), log=lambda m: None), sort_keys=True))
    pipe.reports_dir(CATEGORY).mkdir(parents=True, exist_ok=True)
    pipe.study_report_path(CATEGORY).write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    try:
        yield {"root": root, "art": art, "inputs": inputs, "study": study, "decoded": decoded}
    finally:
        mp.undo()


@pytest.fixture
def sandbox(world, tmp_path, monkeypatch):
    art = tmp_path / "ai_models"
    shutil.copytree(world["art"], art)
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", art)
    return art


def _lock():
    outcome = pipe.create_selection_lock(CATEGORY, json.loads(pipe.study_report_path(CATEGORY).read_text()))
    assert outcome["locked"]
    return outcome["lock"]


def _forbid_test_access(monkeypatch):
    monkeypatch.setattr(final, "discover_test_samples", lambda c: (_ for _ in ()).throw(AssertionError("test set touched")))


def test_study_covers_eight_candidates_by_eight_policies_and_labels_synthetic_metrics_diagnostic(world):
    study = world["study"]
    assert sorted(study["candidates"]) == sorted(s.candidate_id for s in mfs.registry()) and len(study["selection"]["rows"]) == 64
    assert study["synthetic"]["label"].startswith("DIAGNOSTIC ONLY") and study["synthetic"]["images"] == 4 * 15
    synthetic = study["synthetic"]  # the count is derived from the actual held-out split, never hard-coded
    assert synthetic["held_out_images"] == 4 and synthetic["variants_per_image"] == 15
    assert synthetic["images"] == synthetic["held_out_images"] * synthetic["variants_per_image"]
    assert "Pill" in synthetic["label"] and "Leather" not in synthetic["label"] and "Metal Nut" not in synthetic["label"]
    assert study["registry_fingerprint"] == mfs.registry_fingerprint() and study["category"] == CATEGORY
    assert all(study["split_audit"][k] for k in ("calibration_and_held_out_disjoint_in_every_split", "every_split_covers_the_pool", "kfold_held_out_folds_partition_the_pool"))


def test_no_final_test_image_is_decoded_or_contributes_features_before_the_lock(world):
    assert world["decoded"] and {p.parent.name for p in world["decoded"]} == {"good"} and not [p for p in world["decoded"] if "test" in p.parts]
    assert world["study"]["counts"]["test_by_type"] == TINY_TYPES
    assert len(world["inputs"].names) == 20 and all(n.endswith(".png") for n in world["inputs"].names)


def test_candidate_artifacts_record_config_fingerprints_hash_features_and_thresholds(world):
    for spec in mfs.registry():
        d = pipe.candidates_dir(CATEGORY) / spec.candidate_id
        manifest = json.loads((d / "candidate_manifest.json").read_text())
        assert manifest["artifact"]["sha256"] == hashlib.sha256((d / "model_state.pt").read_bytes()).hexdigest() == world["study"]["candidates"][spec.candidate_id]["artifact"]["sha256"]
        assert manifest["candidate_id"] == spec.candidate_id and manifest["config"] == json.loads(json.dumps(spec.config()))
        assert manifest["training_data_fingerprint"] == world["study"]["dataset_fingerprint_sha256"] and manifest["training_image_count"] == 20 and manifest["fitted_on"] == "all 20 train/good images"  # derived, not "245"
        assert manifest["backbone_sha256"] == RESNET18_SHA256 and set(manifest["thresholds_full_pool"]) == set(mfs.policy_ids())
        assert manifest["feature_config"]["feature_dim"] == spec.feature_dim


def test_two_complete_runs_reproduce_everything(world, tmp_path):
    second = json.loads(json.dumps(pipe.run_study(pipe.load_inputs(CATEGORY, TINY_EXPECTED), tmp_path / "candidates", log=lambda m: None), sort_keys=True))
    repro = pipe.compare_studies(world["study"], second)
    assert all(repro.values()), repro
    tampered = json.loads(json.dumps(second))
    tampered["candidates"][next(iter(tampered["candidates"]))]["artifact"]["sha256"] = "0" * 64
    assert not pipe.compare_studies(world["study"], tampered)["candidate_artifact_hashes_identical"]


def test_selection_lock_contents_hashes_and_isolation_from_phase1(world, sandbox):
    before = pipe.prior_phase_reference(CATEGORY)
    assert set(before) == {"phase1_baseline/autoencoder.pt", "phase1_baseline/reports/phase1_report.json"}
    lock = _lock()
    for key in ("category", "experiment_id", "candidate_list", "candidate_configurations", "backbone", "normal_dataset_fingerprint_sha256",
                "split_fingerprint_sha256", "threshold_candidates", "normal_robustness_results", "synthetic_diagnostic_results", "selection_rule",
                "selected_candidate", "selected_policy", "selected_threshold", "selected_artifact", "timestamp_utc", "lock_digest",
                "selection_rule_sha256", "prior_phase_artifacts"):
        assert key in lock, key
    assert lock["category"] == CATEGORY and lock["experiment_id"].startswith("pill-model-family-pill-") and len(lock["candidate_list"]) == 8
    assert lock["selection_rule_sha256"] == hashlib.sha256(mfs.__doc__.encode("utf-8")).hexdigest()  # rule version pinned in the lock
    assert lock["synthetic_diagnostic_results"]["synthetic_digest_sha256"] == json.loads(pipe.study_report_path(CATEGORY).read_text())["synthetic"]["digest_sha256"]
    assert lock["synthetic_diagnostic_results"]["label"] == "DIAGNOSTIC ONLY" and lock["final_test_data_used_for_selection"] is False
    selected = pipe.selected_dir(CATEGORY) / "model_state.pt"
    assert lock["selected_artifact"]["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest() and lock["backbone"]["sha256"] == RESNET18_SHA256
    assert pipe.study_root(CATEGORY).name == "model_family_study" and pipe.prior_phase_reference(CATEGORY) == before == lock["prior_phase_artifacts"]
    assert (sandbox / CATEGORY / "phase1_baseline" / "autoencoder.pt").read_bytes() == b"phase1-weights"
    with pytest.raises(FileExistsError):
        _lock()


def test_no_gate_passer_writes_no_lock(world, sandbox):
    study = json.loads(pipe.study_report_path(CATEGORY).read_text())
    study["selection"] = {**study["selection"], "winner": None, "message": mfs.NO_CANDIDATE_MESSAGE}
    outcome = pipe.create_selection_lock(CATEGORY, study)
    assert outcome["locked"] is False and not pipe.lock_path(CATEGORY).exists() and not pipe.selected_dir(CATEGORY).exists()


def test_final_test_runs_once_after_the_lock_with_a_clean_leakage_audit(world, sandbox, monkeypatch):
    lock = _lock()
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
    assert m["true_positives"] + m["false_negatives"] == 14 and m["true_negatives"] + m["false_positives"] == 4
    assert m["defect_identification_accuracy"] == m["recall"] and 0.0 <= result["auroc"] <= 1.0
    assert set(result["per_defect"]) == set(DEFECT_NAMES) and sum(v["total"] for v in result["per_defect"].values()) == 14
    for p in result["predictions"]:
        assert p["predicted_label"] == int(p["reconstruction_error"] > lock["selected_threshold"])
    assert result["leakage_audit"]["all_passed"], result["leakage_audit"]
    assert set(result["comparison"]) == {"phase1_convae", "alternative", "per_defect"}  # only Phase 1 exists for Pill
    assert set(result["comparison"]["per_defect"]["crack"]) == {"phase1", "alternative", "alternative_detected", "total"}
    assert result["selection_lock_file_sha256"] == file_hashes(pipe.lock_path(CATEGORY))["sha256"]  # the result references the exact lock file
    assert result["artifact_sha256"] == lock["selected_artifact"]["sha256"] and result["backbone_sha256"] == RESNET18_SHA256
    assert final.verify_lock_unchanged_since_final_test(CATEGORY)
    with pytest.raises(FileExistsError):
        final.evaluate_locked_on_final_test(CATEGORY)
    text = pipe.lock_path(CATEGORY).read_text()
    pipe.lock_path(CATEGORY).write_text(text.replace('"category"', '"category" ', 1))
    assert not final.verify_lock_unchanged_since_final_test(CATEGORY)


def test_tampered_candidate_artifact_is_rejected_before_the_test_set_is_touched(world, sandbox, monkeypatch):
    _lock()
    with open(pipe.selected_dir(CATEGORY) / "model_state.pt", "ab") as handle:
        handle.write(b"tamper")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="artifact no longer matches"):
        final.evaluate_locked_on_final_test(CATEGORY)
    assert not final.result_path(CATEGORY).exists()


def test_tampered_backbone_is_rejected(world, sandbox, monkeypatch):
    _lock()
    with open(sandbox / "_pretrained" / RESNET18_FILENAME, "ab") as handle:
        handle.write(b"tamper")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="backbone"):
        final.evaluate_locked_on_final_test(CATEGORY)


def test_tampered_selection_lock_is_rejected_in_three_ways(world, sandbox, monkeypatch):
    _lock()
    _forbid_test_access(monkeypatch)
    path = pipe.lock_path(CATEGORY)
    lock = json.loads(path.read_text())
    edited = {**lock, "selected_threshold": lock["selected_threshold"] * 2}
    path.write_text(json.dumps(edited))
    with pytest.raises(RuntimeError, match="does not match its digest"):
        final.evaluate_locked_on_final_test(CATEGORY)
    edited["lock_digest"] = pipe.lock_digest(edited)
    path.write_text(json.dumps(edited))
    with pytest.raises(RuntimeError, match="not what the pre-declared rule selects"):
        final.evaluate_locked_on_final_test(CATEGORY)
    short = {**lock, "candidate_list": lock["candidate_list"][:-1]}
    short["lock_digest"] = pipe.lock_digest(short)
    path.write_text(json.dumps(short))
    with pytest.raises(RuntimeError, match="Candidate count differs"):
        final.evaluate_locked_on_final_test(CATEGORY)
    assert not final.result_path(CATEGORY).exists()


def test_modified_phase1_artifact_and_changed_training_data_are_rejected(world, sandbox, monkeypatch):
    _lock()
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
