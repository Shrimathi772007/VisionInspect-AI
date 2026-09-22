"""Pill Phase 2 threshold-robustness / calibration study (app.ai.evaluation.phase2_threshold_calibration).

Pure-function tests need no dataset. The end-to-end tests build ONE tiny synthetic `pill` model-family study + lock
(the precondition Phase 2 reads) with the real, hash-verified ResNet-18 weights copied into a temporary artifact
root (skipped when the weights are absent), then run the Phase 2 study on top of it. The real experiment is run by
scripts/run_phase2_threshold_calibration.py, not by this suite.
"""

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.evaluation import model_family_pipeline as pipe
from app.ai.evaluation import model_family_study as mfs
from app.ai.evaluation import phase2_threshold_calibration as p2
from app.ai.evaluation.category_phase1 import file_hashes
from app.ai.models.resnet18 import RESNET18_FILENAME, RESNET18_SHA256, pretrained_weights_path
from app.ai.training import artifacts
from app.ai.training.dataset import DATASET_ROOT

CATEGORY = "pill"
TINY_EXPECTED = {"train_good": 20, "test_good": 4, "test_defective": 14, "test_total": 18}
REAL_WEIGHTS = pretrained_weights_path()
REAL_EXPECTED_ARTIFACT_SHA256 = "6222a2fc968f73b9cd1081f0377b9850d9400daa2f4857bb56266ec3f0f5c469"
REAL_EXPECTED_LOCK_SHA256 = "fb36b05648ffab99fab00a5a9c020f4a096208185ac39c9d2cb6028ceb669a4a"
needs_real_study = pytest.mark.skipif(
    not (artifacts.ARTIFACTS_ROOT / CATEGORY / "model_family_study" / "selection_lock.json").is_file(),
    reason="real Pill model-family study not present")


# ---------------------------------------------------------------------------
# Pure functions: no dataset, no fitting
# ---------------------------------------------------------------------------

def _entry(full_pool_threshold, worst, mean, splits_fpr, recall=0.8):
    per_scheme = {"scheme": {"splits": [{"fpr": f} for f in splits_fpr], "mean_fpr": mean, "worst_fold_fpr": worst}}
    return {"full_pool_threshold": full_pool_threshold, "mean_fpr": mean, "worst_fold_fpr": worst,
            "pooled_threshold": {"mean": full_pool_threshold, "std": 0.1, "cv": 0.1 / full_pool_threshold, "min": full_pool_threshold - 0.2, "max": full_pool_threshold + 0.2},
            "per_scheme": per_scheme, "synthetic_recall": recall, "synthetic_recall_by_type": {}, "synthetic_recall_by_severity": {}}


def test_declared_rule_is_written_before_any_result_and_states_the_mandatory_caveats():
    doc = p2.__doc__.replace("\n", " ")
    assert "PRE-DECLARED THRESHOLD-SELECTION RULE" in doc and "MANDATORY STATEMENT" in doc
    assert "does not establish that the" in doc and "not real Pill defect recall" in doc
    assert "worst-fold FPR <= 10%" in doc


def test_pooled_fpr_and_gate_baseline_already_best_gives_classification_b():
    policies = {
        "mean_std_3": _entry(37.25, worst=0.0377, mean=0.008, splits_fpr=[0.0] * 20 + [0.0377] * 10, recall=0.88),
        "mean_std_2.5": _entry(36.0, worst=0.0755, mean=0.021, splits_fpr=[0.0] * 15 + [0.0755] * 15, recall=0.90),
        "mean_std_1.5": _entry(33.5, worst=0.17, mean=0.079, splits_fpr=[0.05] * 30, recall=0.94),  # fails gate
    }
    sel = p2.select_calibration_threshold(policies, "mean_std_3")
    assert sel["classification"] == "B_no_stable_threshold" and sel["selected_policy"] == "mean_std_3"
    assert sel["gate_passers"] == ["mean_std_3", "mean_std_2.5"] and sel["stability_supported"] == ["mean_std_3"]
    assert sel["best_worst_fold_fpr_among_gate_passers"] == pytest.approx(0.0377)
    assert [r["policy"] for r in sel["rows"]] == [p for p in mfs.policy_ids() if p in policies]  # registry order, not insertion order


def test_a_less_conservative_policy_within_the_band_is_calibration_supported():
    policies = {
        "mean_std_3": _entry(37.0, worst=0.09, mean=0.05, splits_fpr=[0.09] * 30, recall=0.80),
        "mean_std_2.5": _entry(35.0, worst=0.08, mean=0.04, splits_fpr=[0.08] * 30, recall=0.90),  # within 2pp, lower threshold
    }
    sel = p2.select_calibration_threshold(policies, "mean_std_3")
    assert sel["classification"] == "A_calibration_supported_lower_threshold"
    assert sel["selected_policy"] == "mean_std_2.5" and sel["selected_threshold"] == 35.0
    assert "NOT been evaluated against" in sel["reasoning"] and "spent Pill final test" in sel["reasoning"]


def test_no_gate_passer_is_classification_b_with_no_selected_policy():
    policies = {"mean_std_3": _entry(37.0, worst=0.30, mean=0.20, splits_fpr=[0.30] * 30)}
    sel = p2.select_calibration_threshold(policies, "mean_std_3")
    assert sel["classification"] == "B_no_stable_threshold" and sel["selected_policy"] is None and sel["gate_passers"] == []


def test_synthetic_recall_alone_never_overrides_normal_only_stability():
    # a much higher synthetic recall at a failing-gate policy must not be selected (case C in the brief)
    policies = {
        "mean_std_3": _entry(37.0, worst=0.05, mean=0.02, splits_fpr=[0.05] * 30, recall=0.70),
        "mean_std_1.5": _entry(30.0, worst=0.25, mean=0.15, splits_fpr=[0.25] * 30, recall=0.99),  # fails gate despite high recall
    }
    sel = p2.select_calibration_threshold(policies, "mean_std_3")
    assert sel["selected_policy"] == "mean_std_3" and "mean_std_1.5" not in sel["gate_passers"]


def test_split_count_thresholds_le_5_10_15_percent_are_counted_per_policy():
    policies = {"mean_std_3": _entry(37.0, worst=0.12, mean=0.05, splits_fpr=[0.0] * 10 + [0.04] * 10 + [0.08] * 5 + [0.12] * 5)}
    sel = p2.select_calibration_threshold(policies, "mean_std_3")
    row = sel["rows"][0]
    assert (row["splits_fpr_le_5pct"], row["splits_fpr_le_10pct"], row["splits_fpr_le_15pct"], row["splits_total"]) == (20, 25, 30, 30)
    assert row["median_fpr"] == pytest.approx(np.median([0.0] * 10 + [0.04] * 10 + [0.08] * 5 + [0.12] * 5))


def test_frozen_state_check_detects_any_of_the_four_tampering_modes():
    before = {"selection_lock_sha256": "a", "selected_artifact_sha256": "b", "selected_candidate": "c",
              "selected_policy": "d", "selected_threshold": 1.0, "final_test_result_sha256": "e", "final_test_confusion_matrix": [[1, 0], [0, 1]]}
    assert p2.verify_frozen_state_unchanged(before, dict(before))["all_passed"]
    for key, bad in (("selection_lock_sha256", "TAMPERED"), ("selected_artifact_sha256", "TAMPERED"),
                     ("selected_threshold", 2.0), ("final_test_result_sha256", "TAMPERED"), ("final_test_confusion_matrix", [[0, 1], [1, 0]])):
        after = {**before, key: bad}
        assert not p2.verify_frozen_state_unchanged(before, after)["all_passed"], key


# ---------------------------------------------------------------------------
# Static leakage protections
# ---------------------------------------------------------------------------

def test_module_cannot_list_or_score_test_images_and_never_imports_the_final_test_module():
    source = Path(p2.__file__).read_text(encoding="utf-8")
    assert "discover_test_samples" not in source
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not [m for m in imported if m and m.endswith("model_family_final")]


def test_serving_has_no_pill_registration():
    serving = (Path(p2.__file__).parent.parent / "inference" / "serving.py").read_text(encoding="utf-8").lower()
    assert "pill" not in serving.replace("pillow", "")


# ---------------------------------------------------------------------------
# Real Pill study (read-only; skipped if the real model-family study is absent)
# ---------------------------------------------------------------------------

@needs_real_study
def test_real_frozen_state_matches_the_pinned_hashes():
    state = p2.read_frozen_state(CATEGORY)
    assert state["selection_lock_digest_valid"]
    assert state["selected_artifact_sha256"] == REAL_EXPECTED_ARTIFACT_SHA256
    assert state["selection_lock_sha256"] == REAL_EXPECTED_LOCK_SHA256
    assert state["selected_candidate"] == "gauss_l23_256" and state["selected_policy"] == "mean_std_3"
    assert state["final_test_confusion_matrix"] == [[26, 0], [63, 78]]


@needs_real_study
def test_real_frozen_candidate_spec_matches_the_registry():
    spec, policy, lock = p2.frozen_candidate_spec(CATEGORY)
    assert spec.candidate_id == "gauss_l23_256" and policy == "mean_std_3"
    assert spec.layers == (2, 3) and spec.image_size == 256 and spec.family == mfs.FAMILY_GAUSSIAN


# ---------------------------------------------------------------------------
# Tiny synthetic pill world: an end-to-end Phase 2 run on top of a tiny model-family lock
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


DEFECT_NAMES = ("color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if not REAL_WEIGHTS.is_file():
        pytest.skip("verified ResNet-18 weights not present")
    tmp = tmp_path_factory.mktemp("pill_phase2_world")
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

    mp = pytest.MonkeyPatch()
    mp.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    mp.setattr(artifacts, "ARTIFACTS_ROOT", art)
    mp.setattr(mfs, "HOLDOUT_SEEDS", (42, 43))
    mp.setattr(mfs, "RANDOM_KFOLD_SEEDS", (42,))
    mp.setattr(mfs, "GATE_FPR", 0.30)  # 4 held-out images per split: one false positive = 25%

    inputs = pipe.load_inputs(CATEGORY, TINY_EXPECTED)
    study = json.loads(json.dumps(pipe.run_study(inputs, pipe.candidates_dir(CATEGORY), log=lambda m: None), sort_keys=True))
    pipe.reports_dir(CATEGORY).mkdir(parents=True, exist_ok=True)
    pipe.study_report_path(CATEGORY).write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    outcome = pipe.create_selection_lock(CATEGORY, study)
    assert outcome["locked"], outcome  # the tiny world must produce a winner for Phase 2 to have something to read
    try:
        yield {"root": root, "art": art, "inputs": inputs, "study": study, "lock": outcome["lock"]}
    finally:
        mp.undo()


@pytest.fixture
def sandbox(world, tmp_path, monkeypatch):
    art = tmp_path / "ai_models"
    shutil.copytree(world["art"], art)
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", art)
    monkeypatch.setattr(mfs, "HOLDOUT_SEEDS", (42, 43))
    monkeypatch.setattr(mfs, "RANDOM_KFOLD_SEEDS", (42,))
    monkeypatch.setattr(mfs, "GATE_FPR", 0.30)
    return art


def test_phase2_study_reads_the_frozen_candidate_and_leaves_it_byte_identical(world, sandbox):
    before = p2.read_frozen_state(CATEGORY)
    spec, policy, lock = p2.frozen_candidate_spec(CATEGORY)
    assert spec.candidate_id == world["lock"]["selected_candidate"] and policy == world["lock"]["selected_policy"]
    report = p2.run_threshold_study(world["inputs"], spec, policy, p2.phase2_candidates_dir(CATEGORY), log=lambda m: None)
    after = p2.read_frozen_state(CATEGORY)
    check = p2.verify_frozen_state_unchanged(before, after)
    assert check["all_passed"], check
    assert report["frozen_candidate"]["candidate_id"] == spec.candidate_id
    assert report["synthetic"]["images"] == report["synthetic"]["held_out_images"] * report["synthetic"]["variants_per_image"]
    assert set(report["selection"]["baseline"].keys()) >= {"policy", "threshold_full_pool", "worst_fold_fpr"}
    assert report["selection"]["baseline_policy"] == policy
    # the recomputed artifact (fit on the SAME data with the SAME deterministic procedure) matches the locked one exactly
    assert report["recomputed_artifact_sha256"] == before["selected_artifact_sha256"]


def test_no_real_test_image_is_decoded_during_the_phase2_study(world, sandbox, monkeypatch):
    import app.ai.preprocessing.pipeline as preprocessing

    decoded = []
    real_load = preprocessing._load_image
    monkeypatch.setattr(pipe, "_load_image", lambda p: (decoded.append(Path(p)), real_load(p))[1])
    spec, policy, _ = p2.frozen_candidate_spec(CATEGORY)
    inputs = pipe.load_inputs(CATEGORY, TINY_EXPECTED)  # count_dataset_entries touches directory entries only, never decodes
    p2.run_threshold_study(inputs, spec, policy, p2.phase2_candidates_dir(CATEGORY), log=lambda m: None)
    assert decoded and all("train" in p.parts and "good" in p.parts for p in decoded)


def test_cli_study_and_repro_commands_agree_and_never_touch_the_frozen_state(world, sandbox, monkeypatch):
    script_path = Path(p2.__file__).parents[3] / "scripts" / "run_phase2_threshold_calibration.py"
    spec = importlib.util.spec_from_file_location("run_phase2_under_test", script_path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "EXPECTED_BY_CATEGORY", {CATEGORY: TINY_EXPECTED})
    monkeypatch.setattr(script, "EXPECTED_ARTIFACT_SHA256", {})
    monkeypatch.setattr(script, "EXPECTED_LOCK_SHA256", {})
    before = p2.read_frozen_state(CATEGORY)

    monkeypatch.setattr(sys, "argv", ["run_phase2_threshold_calibration.py", "study", "--category", CATEGORY])
    script.main()
    report = json.loads(p2.phase2_report_path(CATEGORY).read_text(encoding="utf-8"))
    assert report["frozen_state_check"]["all_passed"] and report["recomputed_artifact_matches_locked_selected_artifact"]

    monkeypatch.setattr(sys, "argv", ["run_phase2_threshold_calibration.py", "repro", "--category", CATEGORY])
    script.main()
    repro = json.loads((p2.phase2_reports_dir(CATEGORY) / "phase2_reproducibility.json").read_text(encoding="utf-8"))
    assert all(v for v in repro.values() if isinstance(v, bool)), repro
    assert repro["frozen_state_unchanged"] and repro["selection_identical"]
    after = p2.read_frozen_state(CATEGORY)
    assert p2.verify_frozen_state_unchanged(before, after)["all_passed"]
    # the original model-family artifacts are untouched, and Phase 2 wrote to its own, separate directory
    assert p2.phase2_root(CATEGORY) != pipe.study_root(CATEGORY)
    assert (p2.phase2_root(CATEGORY)).is_dir() and (pipe.study_root(CATEGORY) / "selection_lock.json").is_file()


def test_cli_stops_before_running_when_the_pinned_artifact_hash_does_not_match(world, sandbox, monkeypatch):
    script_path = Path(p2.__file__).parents[3] / "scripts" / "run_phase2_threshold_calibration.py"
    spec = importlib.util.spec_from_file_location("run_phase2_under_test_2", script_path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "EXPECTED_BY_CATEGORY", {CATEGORY: TINY_EXPECTED})
    monkeypatch.setattr(script, "EXPECTED_ARTIFACT_SHA256", {CATEGORY: "0" * 64})
    monkeypatch.setattr(script, "EXPECTED_LOCK_SHA256", {})
    monkeypatch.setattr(sys, "argv", ["run_phase2_threshold_calibration.py", "study", "--category", CATEGORY])
    with pytest.raises(SystemExit, match="STOP"):
        script.main()
    assert not p2.phase2_report_path(CATEGORY).exists()  # stopped before any computation


def test_tampered_lock_digest_stops_before_the_study_runs(world, sandbox, monkeypatch):
    script_path = Path(p2.__file__).parents[3] / "scripts" / "run_phase2_threshold_calibration.py"
    spec = importlib.util.spec_from_file_location("run_phase2_under_test_3", script_path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "EXPECTED_ARTIFACT_SHA256", {})
    monkeypatch.setattr(script, "EXPECTED_LOCK_SHA256", {})

    path = pipe.lock_path(CATEGORY)
    lock = json.loads(path.read_text())
    edited = {**lock, "selected_threshold": lock["selected_threshold"] * 2}  # digest now stale
    path.write_text(json.dumps(edited))
    try:
        assert p2.read_frozen_state(CATEGORY)["selection_lock_digest_valid"] is False
        with pytest.raises(SystemExit, match="digest"):
            script._check_frozen_before(CATEGORY)
        assert not p2.phase2_report_path(CATEGORY).exists()  # stopped before any computation
    finally:
        path.write_text(json.dumps(lock))  # restore for any test that runs after this one
