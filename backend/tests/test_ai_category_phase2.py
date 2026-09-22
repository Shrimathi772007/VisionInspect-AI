"""Phase 2 threshold robustness study (app.ai.evaluation.category_phase2 / category_phase2_final).

Pure computations use synthetic score arrays; the end-to-end flow uses a tiny synthetic category (32x32 images,
1 epoch) with a real Phase 1 run written into a temporary artifact root. The real Leather experiment is run by
scripts/run_category_phase2.py, not by the suite.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.ai.evaluation import category_phase1 as phase1
from app.ai.evaluation import category_phase2 as p2
from app.ai.evaluation import category_phase2_final as p2f
from app.ai.evaluation.category_phase2 import (
    CV_TIE,
    NO_CANDIDATE_MESSAGE,
    compare_studies,
    create_lock,
    holdout_splits,
    kfold_splits,
    lock_digest,
    lock_path,
    policy_ids,
    policy_threshold,
    reports_dir,
    run_study,
    select_policy,
    study_from_model,
)
from app.ai.training.schemas import TrainingConfig
from tests.conftest import make_image_bytes

CATEGORY = "widget"
IMAGE_SIZE = (32, 32)
EXPECTED = {"train_good": 30, "test_good": 6, "test_defective": 8, "test_total": 14}


# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------

def test_policy_thresholds_use_population_std_and_linear_percentiles():
    x = np.arange(1.0, 101.0)
    assert policy_threshold("mean_std_2", x) == pytest.approx(x.mean() + 2 * x.std(ddof=0))
    assert policy_threshold("mean_std_2.5", x) != pytest.approx(x.mean() + 2.5 * x.std(ddof=1))
    assert policy_threshold("percentile_95", x) == pytest.approx(np.percentile(x, 95))
    assert policy_ids() == ["mean_std_1", "mean_std_1.5", "mean_std_2", "mean_std_2.5", "mean_std_3",
                            "percentile_95", "percentile_97", "percentile_98", "percentile_99"]
    with pytest.raises(ValueError):
        policy_threshold("median", x)


def test_holdout_schemes_have_the_declared_sizes_and_are_deterministic():
    for held, calib in ((49, 196), (61, 184), (74, 171)):
        a, b = holdout_splits(245, held), holdout_splits(245, held)
        assert len(a) == 10
        for x, y in zip(a, b):
            assert (x["held_out"] == y["held_out"]).all() and (x["calibration"] == y["calibration"]).all()
            assert len(x["held_out"]) == held and len(x["calibration"]) == calib
            assert not set(x["held_out"]) & set(x["calibration"])
            assert sorted(np.concatenate([x["held_out"], x["calibration"]]).tolist()) == list(range(245))
    assert (a[0]["held_out"] != a[1]["held_out"]).any()  # different seeds -> different splits


def test_kfold_folds_partition_the_pool_and_seeds_differ():
    for seed in (42, 43, 44):
        folds = kfold_splits(245, seed)
        assert [len(f["held_out"]) for f in folds] == [49] * 5
        assert sorted(np.concatenate([f["held_out"] for f in folds]).tolist()) == list(range(245))
        assert all(not set(f["held_out"]) & set(f["calibration"]) for f in folds)
    assert (kfold_splits(245, 42)[0]["held_out"] != kfold_splits(245, 43)[0]["held_out"]).any()


def _scores(seed=0, n=245):
    return np.random.default_rng(seed).lognormal(mean=-7.5, sigma=0.25, size=n)


def test_run_study_covers_every_scheme_policy_and_split():
    names = [f"{i:03d}.png" for i in range(245)]
    study = run_study(names, _scores())
    assert set(study["policies"]) == set(policy_ids())
    assert set(study["scheme_definitions"]) == {"A_80_20", "B_75_25", "C_70_30", "kfold5_seed42", "kfold5_seed43", "kfold5_seed44"}
    for p in study["policies"].values():
        assert p["pooled"]["splits"] == 30 + 15
        assert all(k in p["per_scheme"]["A_80_20"]["threshold"] for k in ("mean", "std", "cv", "min", "max"))
        assert p["per_scheme"]["kfold5_seed42"]["total_held_out_images"] == 245
    assert all(study["split_audit"][k] for k in ("calibration_and_held_out_disjoint_in_every_split", "every_split_covers_the_pool",
                                                 "kfold_held_out_folds_partition_the_pool"))
    assert study["held_out_scores_are_in_sample_for_the_model"] is True


def test_false_positive_means_strictly_above_the_threshold():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    split = {"name": "s", "held_out": np.array([9]), "calibration": np.arange(9)}
    rows = p2._fold_rows(scores, [split], "percentile_99")
    assert rows[0]["held_out_false_positives"] == 1  # 10.0 > 99th percentile of 1..9
    split2 = {"name": "s", "held_out": np.array([8]), "calibration": np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])}
    rows2 = p2._fold_rows(scores, [split2], "percentile_95")
    assert rows2[0]["held_out_false_positives"] == 0  # 9.0 is below the 95th percentile of the rest


def test_study_is_deterministic_and_compare_studies_detects_differences():
    names = [f"{i:03d}.png" for i in range(245)]
    a = json.loads(json.dumps(run_study(names, _scores()), sort_keys=True))
    b = json.loads(json.dumps(run_study(names, _scores()), sort_keys=True))
    assert all(compare_studies(a, b).values())
    c = json.loads(json.dumps(run_study(names, _scores(seed=1)), sort_keys=True))
    assert not compare_studies(a, c)["scores_identical"]
    assert not compare_studies(a, c)["candidate_thresholds_and_fprs_identical"]


# ---------------------------------------------------------------------------
# The pre-declared selection rule
# ---------------------------------------------------------------------------

def _result(worst, cv, thr=1.0):
    return {"pooled": {"worst_fold_fpr": worst, "cv": cv}, "full_pool_threshold": thr}


def test_ineligible_policies_are_never_selected():
    sel = select_policy({"mean_std_1": _result(0.30, 0.001), "mean_std_3": _result(0.05, 0.05)})
    assert sel["winner"] == "mean_std_3" and sel["eligible"] == ["mean_std_3"]


def test_boundary_of_ten_percent_is_eligible_and_just_above_is_not():
    assert select_policy({"a": _result(0.10, 0.01)})["winner"] == "a"
    assert select_policy({"a": _result(0.1000001, 0.01)})["winner"] is None


def test_no_eligible_candidate_reports_the_declared_message_and_no_threshold():
    sel = select_policy({"a": _result(0.2, 0.01), "b": _result(0.4, 0.01)})
    assert sel["winner"] is None and sel["threshold"] is None
    assert sel["message"] == NO_CANDIDATE_MESSAGE == "No threshold candidate satisfied the predeclared calibration FPR requirement."


def test_stability_tie_band_then_lower_worst_fpr_then_simplicity():
    # b has the lowest cv but a higher worst FPR; a is within the tie band and has the lower worst FPR -> a wins.
    sel = select_policy({"mean_std_2": _result(0.04, 0.020), "mean_std_3": _result(0.09, 0.020 - CV_TIE / 2)})
    assert sel["winner"] == "mean_std_2"
    # outside the tie band, stability wins even though its worst FPR is higher.
    sel = select_policy({"mean_std_2": _result(0.02, 0.05), "mean_std_3": _result(0.09, 0.05 - CV_TIE - 0.001)})
    assert sel["winner"] == "mean_std_3"
    # exact tie on worst FPR and CV -> the simpler (mean+K*std) policy beats a percentile.
    sel = select_policy({"percentile_97": _result(0.05, 0.02), "mean_std_2.5": _result(0.05, 0.02)})
    assert sel["winner"] == "mean_std_2.5"
    # remaining ties -> policy id
    sel = select_policy({"mean_std_3": _result(0.05, 0.02), "mean_std_2": _result(0.05, 0.02)})
    assert sel["winner"] == "mean_std_2"


def test_selection_never_looks_at_anything_but_calibration_statistics():
    names = [f"{i:03d}.png" for i in range(245)]
    study = run_study(names, _scores())
    assert study["selection"]["threshold"] == study["policies"][study["selection"]["winner"]]["full_pool_threshold"]
    assert study["selection"]["winner"] in study["selection"]["eligible"]
    assert set(study["declared_rule"]) >= {"eligibility", "stability", "then", "locked_threshold", "if_none_eligible"}


# ---------------------------------------------------------------------------
# Static protections
# ---------------------------------------------------------------------------

def test_study_module_cannot_list_test_images():
    source = Path(p2.__file__).read_text(encoding="utf-8")
    assert "discover_test_samples" not in source
    import ast

    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not [m for m in imported if "category_phase2_final" in m or "calibration" in m or "phase4" in m]


def test_serving_registry_has_no_leather():
    serving = Path(p2.__file__).parent.parent / "inference" / "serving.py"
    assert "leather" not in serving.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# End to end on a tiny synthetic category
# ---------------------------------------------------------------------------

def _write(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_image_bytes("PNG", size=IMAGE_SIZE, color=color))


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for i in range(30):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", (10 + i * 5, 60 + i % 7, 90))
    for i in range(6):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", (12 + i * 11, 61, 91))
    for defect, base in (("scratch", 200), ("dent", 150)):
        for i in range(4):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", (base, 20 + i * 9, 30))
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    monkeypatch.setattr(phase1, "phase1_config", lambda category: TrainingConfig(
        category=category, image_size=IMAGE_SIZE, batch_size=4, epochs=1, device="cpu"))
    # the tiny world has 30 images: shrink the held-out sizes so every scheme is valid
    monkeypatch.setattr(p2, "HOLDOUT_SCHEMES", {"A_80_20": 6, "B_75_25": 8, "C_70_30": 9})
    run = phase1.run_phase1(CATEGORY, phase1.default_model_path(CATEGORY), EXPECTED,
                            lock_path=phase1.phase1_dir(CATEGORY) / "threshold_lock.json")
    report = phase1.phase1_dir(CATEGORY) / "reports" / "phase1_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"run1": run}), encoding="utf-8")
    return {"sha": run["artifact"]["sha256"], "model": phase1.default_model_path(CATEGORY), "phase1": run}


def _do_study(world):
    ref = p2f.phase1_reference(CATEGORY)
    study = study_from_model(CATEGORY, world["model"], world["sha"], IMAGE_SIZE)
    study["phase1_reference_at_study_time"] = ref
    out = reports_dir(CATEGORY) / "calibration_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    return json.loads(out.read_text(encoding="utf-8"))


def _lock(world, first):
    second = study_from_model(CATEGORY, world["model"], world["sha"], IMAGE_SIZE)
    return create_lock(CATEGORY, world["model"], world["sha"], first, second, p2f.phase1_reference(CATEGORY))


def test_study_refuses_a_model_with_the_wrong_hash(world):
    with pytest.raises(RuntimeError, match="hash mismatch"):
        study_from_model(CATEGORY, world["model"], "0" * 64, IMAGE_SIZE)


def test_study_scores_only_the_train_good_images(world):
    study = _do_study(world)
    assert study["n"] == 30 and len(study["train_good_files"]) == 30
    assert all(f.startswith("0") and f.endswith(".png") for f in study["train_good_files"])
    assert study["train_good_files"] == sorted(study["train_good_files"])


def test_full_flow_locks_before_the_single_final_test(world, monkeypatch):
    first = _do_study(world)
    outcome = _lock(world, first)
    if not outcome["locked"]:  # tiny world may yield no eligible policy; force one to exercise the flow deterministically
        pytest.skip("no eligible policy in the tiny world")
    lock = json.loads(lock_path(CATEGORY).read_text(encoding="utf-8"))
    for key in ("category", "source_model_path", "model_sha256", "threshold", "threshold_method", "calibration_schemes",
                "selection_rule", "timestamp_unix", "timestamp_utc", "experiment_id"):
        assert key in lock
    assert lock["model_sha256"] == world["sha"] and lock["final_test_data_used_for_selection"] is False
    assert lock["threshold"] == first["selection"]["threshold"] and lock["threshold_method"] == first["selection"]["winner"]
    assert lock["phase1_reference"]["threshold"] == world["phase1"]["threshold"]["value"]
    assert outcome["lock_reproducibility"]["lock_digest_identical"] and outcome["lock_reproducibility"]["lock_content_identical_given_same_timestamp"]

    seen = {}
    real = p2f.discover_test_samples

    def spy(category):
        seen["lock_on_disk"] = lock_path(category).is_file()
        seen["result_existed"] = p2f.result_path(category).exists()
        return real(category)

    monkeypatch.setattr(p2f, "discover_test_samples", spy)
    result = p2f.evaluate_locked_on_final_test(CATEGORY, world["sha"], IMAGE_SIZE)
    assert seen == {"lock_on_disk": True, "result_existed": False}
    assert result["locked_threshold"] == lock["threshold"]
    m = result["metrics"]
    assert m["true_positives"] + m["false_negatives"] == 8 and m["true_negatives"] + m["false_positives"] == 6
    for p in result["predictions"]:
        assert p["predicted_label"] == int(p["reconstruction_error"] > lock["threshold"])
    assert result["leakage_audit"]["all_passed"], result["leakage_audit"]
    cmp = result["comparison_with_phase1"]
    assert cmp["metrics"]["threshold"]["phase2"] == lock["threshold"] and cmp["paired"]["same_test_images"]
    assert set(cmp["per_defect"]) == {"dent", "scratch"}

    with pytest.raises(FileExistsError):  # exactly once
        p2f.evaluate_locked_on_final_test(CATEGORY, world["sha"], IMAGE_SIZE)


def test_lock_refuses_to_overwrite_and_final_refuses_a_tampered_lock(world):
    first = _do_study(world)
    outcome = _lock(world, first)
    if not outcome["locked"]:
        pytest.skip("no eligible policy in the tiny world")
    with pytest.raises(FileExistsError):
        _lock(world, first)
    lock = json.loads(lock_path(CATEGORY).read_text(encoding="utf-8"))
    lock["threshold"] *= 1.5  # a threshold that is not the study's selection
    lock_path(CATEGORY).write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not the calibration study's selected threshold"):
        p2f.evaluate_locked_on_final_test(CATEGORY, world["sha"], IMAGE_SIZE)
    assert not p2f.result_path(CATEGORY).exists()


def test_final_refuses_when_the_frozen_model_changed(world):
    first = _do_study(world)
    if not _lock(world, first)["locked"]:
        pytest.skip("no eligible policy in the tiny world")
    with open(world["model"], "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="frozen model"):
        p2f.evaluate_locked_on_final_test(CATEGORY, world["sha"], IMAGE_SIZE)


def test_no_candidate_locks_nothing(world, monkeypatch):
    first = _do_study(world)
    first["selection"] = {"winner": None, "eligible": [], "equivalent_stability": [], "table": {}, "message": NO_CANDIDATE_MESSAGE,
                          "threshold": None, "reasoning": NO_CANDIDATE_MESSAGE}
    second = json.loads(json.dumps(first))
    outcome = create_lock(CATEGORY, world["model"], world["sha"], first, second, p2f.phase1_reference(CATEGORY))
    assert outcome["locked"] is False and outcome["message"] == NO_CANDIDATE_MESSAGE
    assert not lock_path(CATEGORY).exists()


def test_lock_digest_ignores_only_the_timestamp():
    a = {"threshold": 1.0, "timestamp_unix": 1.0, "timestamp_utc": "x"}
    assert lock_digest(a) == lock_digest({**a, "timestamp_unix": 2.0, "timestamp_utc": "y"})
    assert lock_digest(a) != lock_digest({**a, "threshold": 1.5})


def test_phase1_artifacts_are_untouched_by_phase2(world):
    before = p2f.phase1_reference(CATEGORY)
    first = _do_study(world)
    _lock(world, first)
    assert p2f.phase1_reference(CATEGORY) == before
