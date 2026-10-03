"""Logic of the WRN-50-2 PatchCore normal-data study (scripts/experiments/patchcore_wrn50/wrn50_study_logic.py).

No real images and no weights: folds, the synthetic plan, the train/good-only path guard and audit, the
memory-bounded coreset's equivalence with the repo's greedy coreset, the metric arithmetic and every branch
and tie-break of the selection rule.
"""

import sys
from pathlib import Path

import pytest
import torch

from app.ai.models.patch_anomaly import greedy_coreset_indices

STUDY_DIR = Path(__file__).resolve().parent.parent / "scripts" / "experiments" / "patchcore_wrn50"
sys.path.insert(0, str(STUDY_DIR))

import wrn50_study_logic as logic  # noqa: E402


# ---------------------------------------------------------------------------
# Folds and synthetic plan
# ---------------------------------------------------------------------------

def test_fold_is_index_mod_5_over_sorted_names():
    assert [logic.fold_of(i) for i in range(12)] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert logic.fold_members(12, 1) == [1, 6, 11]
    sizes = [len(logic.fold_members(209, k)) for k in range(5)]
    assert sizes == [42, 42, 42, 42, 41] and sum(sizes) == 209


def test_synthetic_plan_cycles_types_and_severities_with_declared_seed():
    plan0 = logic.synthetic_plan(fold=2, position=0)
    plan1 = logic.synthetic_plan(fold=2, position=1)
    assert len(plan0) == 4
    assert [p["defect_type"] for p in plan0 + plan1] == [
        "color_blob", "dark_blob", "line", "wavy_line", "object_patch", "color_blob", "dark_blob", "line"]
    assert [p["severity"] for p in plan0 + plan1] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert {p["seed"] for p in plan0} == {2000} and {p["seed"] for p in plan1} == {2001}
    # Over 15 consecutive defects every (type, severity) pair occurs exactly once.
    combos = [(p["defect_type"], p["severity"]) for pos in range(4) for p in logic.synthetic_plan(0, pos)][:15]
    assert len(set(combos)) == 15


# ---------------------------------------------------------------------------
# Path guard and audit
# ---------------------------------------------------------------------------

def test_path_guard_returns_only_train_good_paths(tmp_path):
    path = logic.train_good_path("bottle", "000.png", dataset_root=tmp_path)
    assert path == (tmp_path / "bottle" / "train" / "good" / "000.png").resolve()
    assert logic.train_good_path("bottle", dataset_root=tmp_path) == (tmp_path / "bottle" / "train" / "good").resolve()


@pytest.mark.parametrize("category, filename", [
    ("bottle", "../../test/broken_large/000.png"),
    ("bottle", "..\\..\\ground_truth\\x.png"),
    ("bottle", "test"),
    ("bottle", "ground_truth"),
    ("bottle", "TEST"),
    ("bottle", ".."),
    ("bottle", ""),
    ("plastic", "000.png"),
])
def test_path_guard_refuses_test_ground_truth_and_escapes(tmp_path, category, filename):
    with pytest.raises(logic.DatasetAccessError):
        logic.train_good_path(category, filename, dataset_root=tmp_path)


@pytest.mark.parametrize("relative", ["bottle/test/good/000.png", "bottle/ground_truth/x/000_mask.png",
                                      "bottle/train/good/../../test/x.png", "bottle/train/000.png"])
def test_audit_check_refuses_entries_outside_train_good(relative):
    with pytest.raises(logic.DatasetAccessError):
        logic.assert_train_good_relative(relative)


def test_audit_records_and_verifies(tmp_path):
    good = tmp_path / "data" / "bottle" / "train" / "good"
    good.mkdir(parents=True)
    (good / "000.png").write_bytes(b"x")
    audit = logic.AccessAudit(tmp_path / "audit.txt", dataset_root=tmp_path / "data")
    names = logic.list_train_good("bottle", audit, dataset_root=tmp_path / "data")
    audit.record("open", logic.train_good_path("bottle", names[0], dataset_root=tmp_path / "data"))
    assert names == ["000.png"]
    assert audit.verify() == {"entries": 2, "all_under_train_good": True, "opened": 1}
    with open(tmp_path / "audit.txt", "a", encoding="utf-8") as handle:
        handle.write("open\tbottle/test/good/000.png\n")
    with pytest.raises(logic.DatasetAccessError):
        audit.verify()


# ---------------------------------------------------------------------------
# Memory-bounded coreset == repo greedy coreset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [16, 96])
def test_fold_coreset_matches_repo_greedy_coreset(dim):
    n_images, patches = 7, 30
    flat = torch.randn(n_images * patches, dim, generator=torch.Generator().manual_seed(5))
    images = [0, 2, 3, 5, 6]
    rows = torch.cat([torch.arange(i * patches, (i + 1) * patches) for i in images])
    expected_local = greedy_coreset_indices(flat[rows], max(1, int(round(len(rows) * 0.10))), seed=0, projection_dim=64)
    ours = logic.greedy_coreset_over_images(flat, images, patches, ratio=0.10, seed=0)
    assert torch.equal(ours, rows[expected_local])
    assert set(ours.tolist()) <= set(rows.tolist())


# ---------------------------------------------------------------------------
# Metrics arithmetic
# ---------------------------------------------------------------------------

def test_metrics_on_a_tiny_example():
    # 10 normals, folds 0..4 twice; percentile_99 of [1..10] = 9.91 -> only the score 10 (index 9, fold 4) is flagged.
    oof = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    folds = [logic.fold_of(i) for i in range(10)]
    synthetic = [9.5, 10.5, 20.0, None]
    m = logic.combination_metrics(oof, folds, synthetic, "percentile_99")
    assert m["threshold"] == pytest.approx(9.91)
    assert m["fold_fpr"] == [0.0, 0.0, 0.0, 0.0, 0.5]
    assert m["worst_fold_fpr"] == 0.5
    assert m["mean_fold_fpr"] == pytest.approx(0.1)
    assert m["pooled_fpr"] == pytest.approx(0.1)
    assert m["synthetic_recall"] == pytest.approx(0.5)  # 10.5 and 20.0 detected; 9.5 missed; failed sample counts as missed
    assert m["synthetic_failed"] == 1

    s = logic.combination_metrics(oof, folds, synthetic, "mean_std_2.5")
    import numpy as np

    assert s["threshold"] == pytest.approx(np.mean(oof) + 2.5 * np.std(oof))  # population std, as in category_phase2


def test_build_selection_table_has_12_rows_in_declared_order():
    data = {"oof": list(range(10)), "folds": [i % 5 for i in range(10)], "synthetic": [100.0] * 4,
            "synthetic_types": ["color_blob", "dark_blob", "line", "wavy_line"]}
    table = logic.build_selection_table({m: {a: data for a in logic.AGGREGATIONS} for m in logic.MODES})
    assert len(table) == 12
    assert [(r["mode"], r["aggregation"], r["policy"]) for r in table][:4] == [
        ("crop224", "max", "mean_std_2.5"), ("crop224", "max", "mean_std_3"), ("crop224", "max", "percentile_99"),
        ("crop224", "top1pct_mean", "mean_std_2.5")]
    assert table[0]["synthetic_recall_by_type"]["object_patch"] is None


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------

def _row(mode="crop224", agg="max", policy="mean_std_3", recall=0.5, worst=0.05, mean=0.03, threshold=1.0):
    return {"mode": mode, "aggregation": agg, "policy": policy, "synthetic_recall": recall, "worst_fold_fpr": worst,
            "mean_fold_fpr": mean, "threshold": threshold, "meets_fpr_limit": worst <= logic.FPR_LIMIT + logic.FPR_EPSILON}


def _pick(rows):
    d = logic.select(rows)
    return d["mode"], d["aggregation"], d["policy"]


def test_highest_recall_among_eligible_wins_even_over_an_ineligible_higher_recall():
    rows = [_row(recall=0.6, worst=0.08), _row(agg="top1pct_mean", recall=0.7, worst=0.09),
            _row(mode="full256", recall=0.95, worst=0.12)]
    assert _pick(rows) == ("crop224", "top1pct_mean", "mean_std_3")
    assert logic.select(rows)["any_combination_met_fpr_limit"] is True


def test_exactly_ten_percent_is_eligible():
    rows = [_row(recall=0.9, worst=0.10), _row(agg="top1pct_mean", recall=0.5, worst=0.0)]
    assert _pick(rows) == ("crop224", "max", "mean_std_3")


def test_recall_tie_goes_to_lower_worst_fold_fpr():
    rows = [_row(recall=0.7, worst=0.08), _row(mode="full256", agg="top1pct_mean", recall=0.7, worst=0.02)]
    assert _pick(rows) == ("full256", "top1pct_mean", "mean_std_3")


def test_then_crop224_then_max():
    rows = [_row(mode="full256", agg="max"), _row(mode="crop224", agg="top1pct_mean")]
    assert _pick(rows) == ("crop224", "top1pct_mean", "mean_std_3")
    rows = [_row(agg="top1pct_mean"), _row(agg="max")]
    assert _pick(rows) == ("crop224", "max", "mean_std_3")


def test_then_lower_mean_fpr_then_higher_threshold_then_policy_order():
    rows = [_row(policy="mean_std_2.5", mean=0.04), _row(policy="percentile_99", mean=0.01)]
    assert _pick(rows)[2] == "percentile_99"
    rows = [_row(policy="mean_std_2.5", threshold=1.0), _row(policy="mean_std_3", threshold=2.0)]
    assert _pick(rows)[2] == "mean_std_3"
    rows = [_row(policy="percentile_99"), _row(policy="mean_std_3"), _row(policy="mean_std_2.5")]
    assert _pick(rows)[2] == "mean_std_2.5"


def test_no_eligible_combination_chooses_lowest_worst_fold_fpr():
    rows = [_row(recall=0.99, worst=0.20), _row(agg="top1pct_mean", recall=0.10, worst=0.12),
            _row(mode="full256", recall=0.90, worst=0.15)]
    decision = logic.select(rows)
    assert (decision["mode"], decision["aggregation"]) == ("crop224", "top1pct_mean")
    assert decision["any_combination_met_fpr_limit"] is False and decision["eligible_count"] == 0


def test_no_eligible_ties_go_to_recall_then_crop224_then_max():
    rows = [_row(recall=0.3, worst=0.2), _row(agg="top1pct_mean", recall=0.6, worst=0.2)]
    assert _pick(rows)[1] == "top1pct_mean"
    rows = [_row(mode="full256", worst=0.2), _row(agg="top1pct_mean", worst=0.2)]
    assert _pick(rows) == ("crop224", "top1pct_mean", "mean_std_3")
    rows = [_row(agg="top1pct_mean", worst=0.2), _row(agg="max", worst=0.2)]
    assert _pick(rows)[1] == "max"


def test_lock_digest_ignores_volatile_fields():
    lock = {"category": "bottle", "threshold": 1.5, "timestamp_utc": "a", "timestamp_unix": 1.0, "run_seconds": 3}
    other = dict(lock, timestamp_utc="b", timestamp_unix=2.0, run_seconds=9, peak_ram_mb=1.0,
                 dataset_access_audit={"entries": 5})
    assert logic.lock_digest(lock) == logic.lock_digest(other)
    assert logic.lock_digest(lock) != logic.lock_digest(dict(lock, threshold=1.6))
