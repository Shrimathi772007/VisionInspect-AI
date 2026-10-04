"""WRN-50-2 PatchCore final test (scripts/experiments/patchcore_wrn50/wrn50_final_logic.py, run_wrn50_final.py).

No real test images and no weights: metric maths on tiny examples, AUROC/AP including ties, Wilson intervals,
label parsing from folder names, the prediction rule at exactly the threshold, the sentinel/consumed one-shot
guard, the test-only path guard (never a mask-folder path), the access classifier, and lock verification on a
synthetic lock (plus the nine real locks when they are present).
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score

from app.ai.models.patchcore import PatchCoreConfig
from app.ai.models.wide_resnet50 import WRN50_2_SHA256
from app.ai.training.artifacts import ARTIFACTS_ROOT

STUDY_DIR = Path(__file__).resolve().parent.parent / "scripts" / "experiments" / "patchcore_wrn50"
sys.path.insert(0, str(STUDY_DIR))

import run_wrn50_final as runner  # noqa: E402
import wrn50_final_logic as fl  # noqa: E402
import wrn50_study_logic as logic  # noqa: E402

MASKS = "ground" + "_truth"  # spelled apart so the static source check below only finds the declared constant


# ---------------------------------------------------------------------------
# Confusion cells and threshold metrics
# ---------------------------------------------------------------------------

def test_confusion_counts_every_cell():
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    preds = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert fl.confusion(labels, preds) == {"true_positives": 3, "true_negatives": 4, "false_positives": 1, "false_negatives": 2}
    assert fl.confusion([0], [1])["false_positives"] == 1
    assert fl.confusion([1], [0])["false_negatives"] == 1
    assert fl.confusion([1], [1])["true_positives"] == 1
    assert fl.confusion([0], [0])["true_negatives"] == 1
    with pytest.raises(ValueError):
        fl.confusion([0, 1], [1])
    with pytest.raises(ValueError):
        fl.confusion([2], [1])


def test_binary_metrics_hand_computed():
    m = fl.binary_metrics(tp=3, tn=4, fp=1, fn=2)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["f1_score"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert m["f2_score"] == pytest.approx(5 * 0.75 * 0.6 / (4 * 0.75 + 0.6))  # 0.625
    assert m["false_positive_rate"] == pytest.approx(0.2)
    assert m["specificity"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx(0.7)
    assert m["mcc"] == pytest.approx(10 / math.sqrt(600))
    assert m["total"] == 10


def test_binary_metrics_agree_with_sklearn():
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    preds = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    c = fl.confusion(labels, preds)
    m = fl.binary_metrics(c["true_positives"], c["true_negatives"], c["false_positives"], c["false_negatives"])
    assert m["precision"] == pytest.approx(precision_score(labels, preds))
    assert m["recall"] == pytest.approx(recall_score(labels, preds))
    assert m["f1_score"] == pytest.approx(f1_score(labels, preds))
    assert m["f2_score"] == pytest.approx(fbeta_score(labels, preds, beta=2))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(labels, preds))


def test_zero_denominators_are_zero():
    m = fl.binary_metrics(tp=0, tn=5, fp=0, fn=3)  # nothing predicted defective
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1_score"] == 0.0 and m["f2_score"] == 0.0
    assert m["false_positive_rate"] == 0.0 and m["specificity"] == 1.0 and m["mcc"] == 0.0
    assert fl.binary_metrics(0, 0, 0, 0)["accuracy"] == 0.0


def test_wilson_interval_known_values():
    lo, hi = fl.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = fl.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = fl.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4) and hi == pytest.approx(1.0)
    assert fl.wilson(0, 0) is None


# ---------------------------------------------------------------------------
# AUROC and average precision
# ---------------------------------------------------------------------------

def test_auroc_and_ap_perfect_and_reversed():
    r = fl.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r == {"auroc": 1.0, "average_precision": 1.0}
    r = fl.ranking_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert r["auroc"] == 0.0
    # reversed: ranks defective 3rd and 4th -> AP = 0.5 * (1/3) + 0.5 * (2/4)
    assert r["average_precision"] == pytest.approx(0.5 / 3 + 0.25)


def test_auroc_and_ap_with_ties():
    # pairs (good, defective): (.1,.4) win, (.1,.8) win, (.4,.4) tie = 1/2, (.4,.8) win -> 3.5 / 4
    r = fl.ranking_metrics([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8])
    assert r["auroc"] == pytest.approx(0.875)
    # thresholds .8 -> P 1, R .5; .4 (tied pair enters together) -> P 2/3, R 1 -> AP = .5 * 1 + .5 * 2/3
    assert r["average_precision"] == pytest.approx(0.5 + 1 / 3)
    tied = fl.ranking_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert tied["auroc"] == pytest.approx(0.5) and tied["average_precision"] == pytest.approx(0.5)


def test_ranking_metrics_single_class_is_none():
    assert fl.ranking_metrics([1, 1], [0.2, 0.3]) == {"auroc": None, "average_precision": None}


# ---------------------------------------------------------------------------
# Labels, prediction rule, evaluate()
# ---------------------------------------------------------------------------

def test_label_parsing_from_folder_names():
    assert fl.label_from_folder("good") == (0, "good")
    assert fl.label_from_folder("broken_large") == (1, "broken_large")
    assert fl.label_from_folder("metal_contamination") == (1, "metal_contamination")
    assert fl.label_from_folder("Good") == (1, "Good")  # exact folder name only
    for bad in (MASKS, MASKS.upper(), f"x_{MASKS}"):
        with pytest.raises(fl.DatasetAccessError):
            fl.label_from_folder(bad)


def test_prediction_rule_at_exactly_the_threshold():
    t = 1.88393018105021
    assert fl.predict(t, t) == 0  # score <= threshold -> good
    assert fl.predict(np.nextafter(t, np.inf), t) == 1
    assert fl.predict(np.nextafter(t, -np.inf), t) == 0


def test_evaluate_end_to_end_on_a_tiny_example():
    labels = [0, 0, 0, 1, 1, 1]
    types = ["good", "good", "good", "crack", "crack", "hole"]
    scores = [0.5, 1.0, 1.2, 1.0, 1.5, 2.0]  # threshold 1.0: the 1.0 defective is missed, the 1.2 good is a FP
    out = fl.evaluate(labels, types, scores, 1.0)
    assert out["predictions"] == [0, 0, 1, 0, 1, 1]
    m = out["metrics"]
    assert (m["true_positives"], m["true_negatives"], m["false_positives"], m["false_negatives"]) == (2, 2, 1, 1)
    assert out["per_defect"]["crack"] == {"total": 2, "detected": 1, "missed": 1, "recall": 0.5, "wilson95": fl.wilson(1, 2)}
    assert out["per_defect"]["hole"]["recall"] == 1.0
    assert out["gate_classification"] == "NOT PRODUCTION READY"
    assert out["wilson95"]["recall"] == fl.wilson(2, 3) and out["wilson95"]["false_positive_rate"] == fl.wilson(1, 3)


def test_gate_thresholds_come_from_category_phase1():
    assert fl.evaluate([1] * 9 + [0] * 10, ["d"] * 9 + ["good"] * 10, [2.0] * 9 + [0.0] * 10, 1.0)["gate_classification"] == "EXCELLENT"


# ---------------------------------------------------------------------------
# One-shot guard
# ---------------------------------------------------------------------------

def _finish(out_dir: Path):
    (out_dir / fl.RESULT).write_text("{}", encoding="utf-8")
    return fl.finish_one_shot(out_dir, {"lock_digest": "a"})


def test_one_shot_refuses_a_second_run(tmp_path):
    body = fl.begin_one_shot(tmp_path, {"lock_digest": "a"})
    assert body["attempts"] == 1 and (tmp_path / fl.SENTINEL).exists()
    _finish(tmp_path)
    assert (tmp_path / fl.CONSUMED).exists()
    for rerun in (False, True):
        with pytest.raises(fl.OneShotError, match="already scored"):
            fl.begin_one_shot(tmp_path, {"lock_digest": "a"}, technical_rerun=rerun)


def test_one_shot_refuses_when_results_exist_even_without_consumed(tmp_path):
    fl.begin_one_shot(tmp_path, {"lock_digest": "a"})
    (tmp_path / fl.PER_IMAGE).write_text("x", encoding="utf-8")
    with pytest.raises(fl.OneShotError, match="never re-run"):
        fl.begin_one_shot(tmp_path, {"lock_digest": "a"}, technical_rerun=True)


def test_one_shot_allows_exactly_one_technical_rerun(tmp_path):
    fl.begin_one_shot(tmp_path, {"lock_digest": "a"})
    with pytest.raises(fl.OneShotError, match="technical re-run"):
        fl.begin_one_shot(tmp_path, {"lock_digest": "a"})  # crashed attempt, no flag
    with pytest.raises(fl.OneShotError, match="identical locked model"):
        fl.begin_one_shot(tmp_path, {"lock_digest": "b"}, technical_rerun=True)
    body = fl.begin_one_shot(tmp_path, {"lock_digest": "a"}, technical_rerun=True)
    assert body["attempts"] == 2 and body["technical_rerun"] and body["first_attempt"]["attempts"] == 1
    with pytest.raises(fl.OneShotError, match="already used"):
        fl.begin_one_shot(tmp_path, {"lock_digest": "a"}, technical_rerun=True)


def test_technical_rerun_flag_without_an_attempt_is_refused(tmp_path):
    with pytest.raises(fl.OneShotError):
        fl.begin_one_shot(tmp_path, {"lock_digest": "a"}, technical_rerun=True)


def test_consumed_needs_a_result_and_is_exclusive(tmp_path):
    fl.begin_one_shot(tmp_path, {"lock_digest": "a"})
    with pytest.raises(fl.OneShotError):
        fl.finish_one_shot(tmp_path, {})
    _finish(tmp_path)
    with pytest.raises(FileExistsError):
        fl.finish_one_shot(tmp_path, {})


# ---------------------------------------------------------------------------
# Test-only path guard and listing (never a mask-folder path)
# ---------------------------------------------------------------------------

def _fake_dataset(root: Path) -> Path:
    for sub, names in {"good": ["000.png", "001.png"], "crack": ["000.png", "notes.txt"], "hole": ["000.PNG"]}.items():
        d = root / "bottle" / "test" / sub
        d.mkdir(parents=True)
        for name in names:
            (d / name).write_bytes(b"x")
    (root / "bottle" / MASKS / "crack").mkdir(parents=True)
    (root / "bottle" / MASKS / "crack" / "000_mask.png").write_bytes(b"x")
    return root


def test_listing_reads_only_test_and_labels_by_folder(tmp_path):
    root = _fake_dataset(tmp_path)
    listed = []
    images, info = fl.list_test_images("bottle", root, record=lambda action, path: listed.append(str(path)))
    assert [im.relative for im in images] == ["bottle/test/crack/000.png", "bottle/test/good/000.png",
                                              "bottle/test/good/001.png", "bottle/test/hole/000.PNG"]
    assert [(im.label, im.defect_type) for im in images] == [(1, "crack"), (0, "good"), (0, "good"), (1, "hole")]
    assert info["ignored_non_png"] == ["crack/notes.txt"]
    assert all(MASKS not in p.lower() for p in listed + [str(im.path) for im in images])
    assert all(Path(p).resolve().is_relative_to((root / "bottle" / "test").resolve()) for p in listed)


@pytest.mark.parametrize("sub, name", [
    (MASKS, None), (MASKS.upper(), None), ("crack", f"{MASKS}.png"), ("..", None), ("crack", "../../x.png"),
    ("crack", "..\\x.png"), ("", None), (None, "000.png"),
])
def test_guard_refuses_mask_folder_and_escapes(tmp_path, sub, name):
    with pytest.raises(fl.DatasetAccessError):
        fl.guarded_path("bottle", sub, name, dataset_root=tmp_path)


def test_guard_refuses_unknown_split(tmp_path):
    with pytest.raises(fl.DatasetAccessError):
        fl.guarded_path("bottle", dataset_root=tmp_path, split="train")


def test_listing_refuses_a_mask_named_test_subfolder(tmp_path):
    (tmp_path / "bottle" / "test" / MASKS).mkdir(parents=True)
    with pytest.raises(fl.DatasetAccessError):
        fl.list_test_images("bottle", tmp_path)


def test_sources_never_build_a_mask_folder_path():
    runner_source = (STUDY_DIR / "run_wrn50_final.py").read_text(encoding="utf-8")
    logic_source = (STUDY_DIR / "wrn50_final_logic.py").read_text(encoding="utf-8")
    # The name may appear only inside leakage-audit key labels ("no_..._path_opened_or_listed": ...), never in a path.
    keys = re.findall(r'"[a-z_]*' + MASKS + r'[a-z_]*":', runner_source)
    assert len(keys) == 3 and all(k.endswith('":') for k in keys)
    assert MASKS not in re.sub(r'"[a-z_]*' + MASKS + r'[a-z_]*":', "", runner_source)
    lines = [line for line in logic_source.splitlines() if MASKS in line]
    assert len(lines) == 1 and lines[0].startswith("FORBIDDEN_PART = ")


def test_access_classifier_flags_mask_paths_and_outside_accesses(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DATASET_ROOT", tmp_path)
    model_dir, out_dir = tmp_path.parent / "m" / "final_model", tmp_path.parent / "out"
    ok = [(0, "listing", "os.scandir", str(tmp_path / "bottle" / "test"), "os.scandir"),
          (0, "scoring", "open", str(tmp_path / "bottle" / "test" / "good" / "000.png"), "rb"),
          (0, "scoring", "open", str(model_dir / "model_state.pt"), "rb"),
          (0, "scoring", "open", str(out_dir / "x.json"), "w")]
    summary = runner.classify_events(ok, "bottle", "test", model_dir, out_dir)
    assert summary["no_ground_truth_path_opened_or_listed"] and summary["all_dataset_accesses_inside_allowed_split"]
    assert summary["dataset_files_opened"] == 1 and summary["dataset_directories_listed"] == 1
    assert summary["no_other_file_opened_during_listing_and_scoring"]
    bad = ok + [(0, "verify", "open", str(tmp_path / "bottle" / MASKS / "a.png"), "rb"),
                (0, "scoring", "open", str(tmp_path / "bottle" / "train" / "good" / "000.png"), "rb"),
                (0, "scoring", "open", str(tmp_path.parent / "elsewhere.csv"), "rb")]
    summary = runner.classify_events(bad, "bottle", "test", model_dir, out_dir)
    assert not summary["no_ground_truth_path_opened_or_listed"]
    assert len(summary["dataset_accesses_outside_allowed_split"]) == 2
    assert not summary["no_other_file_opened_during_listing_and_scoring"]


# ---------------------------------------------------------------------------
# Lock verification
# ---------------------------------------------------------------------------

def _synthetic_lock(tmp_path: Path) -> tuple[Path, Path]:
    study = tmp_path / "study"
    study.mkdir()
    for name in ("run_wrn50_study.py", "wrn50_study_logic.py", "PROTOCOL.md", "STUDY_DECLARATION.md"):
        (study / name).write_text(name, encoding="utf-8")
    directory = tmp_path / "bottle" / "patchcore_wrn50"
    (directory / "final_model").mkdir(parents=True)
    state = directory / "final_model" / "model_state.pt"
    state.write_bytes(b"bank")
    config = PatchCoreConfig().to_dict()
    record = {"detector": "patchcore", "format_version": 1, **config, "bank_shape": [4, 1536], "state_sha256": logic.sha256_file(state)}
    (directory / "final_model" / "model_config.json").write_text(json.dumps(record), encoding="utf-8")
    rows = [{"mode": "crop224", "aggregation": "max", "policy": "mean_std_2.5", "synthetic_recall": 0.9, "worst_fold_fpr": 0.05,
             "mean_fold_fpr": 0.02, "threshold": 1.5, "meets_fpr_limit": True},
            {"mode": "full256", "aggregation": "max", "policy": "mean_std_3", "synthetic_recall": 0.8, "worst_fold_fpr": 0.0,
             "mean_fold_fpr": 0.0, "threshold": 1.9, "meets_fpr_limit": True}]
    (directory / "selection_table.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    lock = {"category": "bottle", "mode": "crop224", "aggregation": "max", "policy": "mean_std_2.5", "threshold": 1.5,
            "config": config, "backbone_weights_sha256": WRN50_2_SHA256, "model_state_sha256": logic.sha256_file(state),
            "model_config_sha256": logic.sha256_file(directory / "final_model" / "model_config.json"), "final_bank_shape": [4, 1536],
            "selection_table_sha256": logic.sha256_file(directory / "selection_table.json"),
            "runner_sha256": logic.sha256_file(study / "run_wrn50_study.py"), "logic_sha256": logic.sha256_file(study / "wrn50_study_logic.py"),
            "protocol_sha256": logic.sha256_file(study / "PROTOCOL.md"), "declaration_sha256": logic.sha256_file(study / "STUDY_DECLARATION.md"),
            "test_directory_listed": False, "timestamp_utc": "x"}
    lock["lock_digest"] = logic.lock_digest(lock)
    (directory / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return directory, study


def _rewrite_lock(directory: Path, redigest: bool, **changes):
    lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
    lock.update(changes)
    if redigest:
        lock["lock_digest"] = logic.lock_digest(lock)
    (directory / "lock.json").write_text(json.dumps(lock), encoding="utf-8")


def test_synthetic_lock_verifies(tmp_path):
    directory, study = _synthetic_lock(tmp_path)
    record, lock = fl.verify_standard_lock(directory, "bottle", study)
    assert record["verified"] and record["selection_rederived"] and lock["threshold"] == 1.5


@pytest.mark.parametrize("tamper", ["threshold_no_digest", "threshold_with_digest", "state", "protected", "table", "listed", "category"])
def test_tampered_lock_is_rejected(tmp_path, tamper):
    directory, study = _synthetic_lock(tmp_path)
    category = "bottle"
    if tamper == "threshold_no_digest":
        _rewrite_lock(directory, False, threshold=1.4)
    elif tamper == "threshold_with_digest":  # consistent digest, but the selection rule no longer gives the lock
        _rewrite_lock(directory, True, threshold=1.4)
    elif tamper == "state":
        (directory / "final_model" / "model_state.pt").write_bytes(b"other")
    elif tamper == "protected":
        (study / "PROTOCOL.md").write_text("edited", encoding="utf-8")
    elif tamper == "table":
        (directory / "selection_table.json").write_text('{"rows": []}', encoding="utf-8")
    elif tamper == "listed":
        _rewrite_lock(directory, True, test_directory_listed=True)
    else:
        category = "capsule"
    with pytest.raises(fl.LockVerificationError):
        fl.verify_standard_lock(directory, category, study)


def test_declared_digest_must_match(tmp_path):
    directory, study = _synthetic_lock(tmp_path)
    with pytest.raises(fl.LockVerificationError, match="declared"):
        fl.verify_standard_lock(directory, "bottle", study, expected_digest="0" * 64)


def test_grid_is_scored_with_the_full320_lock():
    assert fl.lock_dir("grid", Path("m")) == Path("m") / "grid" / "patchcore_wrn50_full320"
    assert fl.lock_dir("bottle", Path("m")) == Path("m") / "bottle" / "patchcore_wrn50"
    assert fl.model_label("grid") == "WRN-50 full320 (addendum 1)"
    assert fl.EXPECTED_LOCK_DIGESTS["grid"].startswith("4cdf78b2") and fl.GRID_SUPERSEDED_LOCK_DIGEST.startswith("de0bd22a")
    assert tuple(fl.EXPECTED_LOCK_DIGESTS) == fl.RUN_ORDER
    with pytest.raises(ValueError):
        fl.lock_dir("tile", Path("m"))


@pytest.mark.skipif(not all((fl.lock_dir(c, ARTIFACTS_ROOT) / "lock.json").is_file() for c in fl.RUN_ORDER),
                    reason="locked WRN-50 models are Git-ignored and not present")
@pytest.mark.parametrize("category", fl.RUN_ORDER)
def test_real_locks_verify(category):
    record, lock = fl.verify_category_lock(category, ARTIFACTS_ROOT)
    assert record["lock_digest"] == fl.EXPECTED_LOCK_DIGESTS[category]
    if category == "grid":
        assert lock["mode"] == "full320" and lock["threshold"] == 1.88393018105021
        assert record["original_grid_lock"]["lock_digest"] == fl.GRID_SUPERSEDED_LOCK_DIGEST
        assert record["original_grid_lock"]["used_for_scoring"] is False
