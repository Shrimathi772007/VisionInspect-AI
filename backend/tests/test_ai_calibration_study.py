"""Good-only calibration study: deterministic folds, cross-validated scoring, threshold transfer and the
pre-declared selection policy. Uses random tensors as 'patch features' - no images, no backbone, no test data."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ai.evaluation import calibration_study as cs
from app.ai.evaluation.calibration_study import (
    POLICIES, ConfigRow, Fold, FoldScores, GoodPool, all_schemes, block_kfold, drift_diagnostics, existing_split,
    fit_scorer, fold_maps, policy_thresholds, pooled_transfer, random_kfold, repeated_holdout, select_configuration,
    single_fold_transfer,
)
from app.ai.models import patch_anomaly as pa

N = 40


def _pool(n=N, h=4, w=4, c=8, seed=0) -> GoodPool:
    g = torch.Generator().manual_seed(seed)
    patches = torch.randn(n, h, w, c, generator=g)
    flat = patches.reshape(n, -1, c).double()
    return GoodPool([f"{i:03d}.png" for i in range(n)], patches, flat.sum(1),
                    torch.einsum("npc,npd->ncd", flat, flat), h * w)


# ---------------------------------------------------------------------------
# Deterministic good-only splits
# ---------------------------------------------------------------------------

def test_random_kfold_partitions_the_pool_deterministically():
    folds = random_kfold(280, 5, 42)
    held = [i for f in folds for i in f.heldout_idx]
    assert sorted(held) == list(range(280)) and [len(f.heldout_idx) for f in folds] == [56] * 5
    for f in folds:
        assert not set(f.train_idx) & set(f.heldout_idx) and len(f.train_idx) + len(f.heldout_idx) == 280
    assert folds == random_kfold(280, 5, 42) and folds != random_kfold(280, 5, 43)


def test_repeated_holdout_sizes_and_determinism():
    for frac, size in ((0.20, 56), (0.25, 70), (0.30, 84)):
        folds = repeated_holdout(280, frac, 4, 7)
        assert [len(f.heldout_idx) for f in folds] == [size] * 4 and len({f.heldout_idx for f in folds}) == 4
        assert folds == repeated_holdout(280, frac, 4, 7)


def test_block_folds_are_contiguous_in_file_order():
    folds = block_kfold(280, 5)
    assert [f.heldout_idx for f in folds] == [tuple(range(i * 56, (i + 1) * 56)) for i in range(5)]


def test_existing_split_reproduces_the_given_validation_files():
    files = [f"{i:03d}.png" for i in range(10)]
    fold = existing_split(files, ["002.png", "007.png"])
    assert fold.heldout_idx == (2, 7) and len(fold.train_idx) == 8


def test_all_schemes_cover_the_documented_designs():
    schemes = all_schemes(280)
    assert set(schemes) == {"random_5fold", "contiguous_blocks", "holdout80_20", "holdout75_25", "holdout70_30"}
    assert len(schemes["random_5fold"]) == 15 and len(schemes["holdout80_20"]) == 10


# ---------------------------------------------------------------------------
# Cross-validated scoring
# ---------------------------------------------------------------------------

def test_statistics_fit_equals_direct_fit():
    pool = _pool()
    idx = list(range(0, 30))
    stats = fit_scorer(pool, idx)
    direct = pa.GaussianPatchScorer.fit(pool.patches[idx])
    assert torch.allclose(stats.mean, direct.mean, atol=1e-5)
    assert torch.allclose(stats.precision, direct.precision, rtol=1e-3, atol=1e-3)


def test_fold_maps_score_heldout_images_with_a_model_that_never_saw_them():
    pool = _pool()
    fold = random_kfold(N, 5, 1)[0]
    maps = fold_maps(pool, fold)
    assert tuple(maps.shape) == (len(fold.heldout_idx), 4, 4)
    # scoring the same images WITH them in the fit gives lower (in-sample) scores on average
    in_sample = fit_scorer(pool, range(N)).patch_scores(pool.patches[list(fold.heldout_idx)])
    assert in_sample.mean() < maps.mean()


def test_new_aggregations_are_correct_and_the_old_ones_unchanged():
    maps = torch.arange(1, 101, dtype=torch.float32).reshape(1, 10, 10)  # patch scores 1..100

    def agg(method):
        return float(pa.aggregate_scores(maps, method)[0])

    assert agg(pa.AGG_MAX) == 100.0
    assert agg(pa.AGG_TOP1PCT) == 100.0  # k = 1
    assert agg(pa.AGG_TOP5PCT) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert agg(pa.AGG_TOP1PCT_MEDIAN) == 100.0
    assert agg(pa.AGG_TOP5PCT_MEDIAN) == pytest.approx(98.0)
    with pytest.raises(ValueError):
        pa.aggregate_scores(maps, "top10pct_mean")


# ---------------------------------------------------------------------------
# Threshold policies and transfer
# ---------------------------------------------------------------------------

def test_nine_established_policies():
    assert len(POLICIES) == 9 and ("mean_std", 3.0) in POLICIES and ("percentile", 99.0) in POLICIES
    t = policy_thresholds(np.arange(100.0))
    assert t[("mean_std", 1.0)] < t[("mean_std", 3.0)] and t[("percentile", 95.0)] < t[("percentile", 99.0)]


def _fold_scores(sets, agg="a"):
    return [FoldScores(Fold("s", f"seed1_fold{i}", (), tuple(idx)), {agg: np.asarray(v, float)})
            for i, (idx, v) in enumerate(sets)]


def test_single_fold_transfer_excludes_the_calibration_images_and_detects_drift():
    rng = np.random.default_rng(0)
    same = [(range(i * 10, (i + 1) * 10), rng.normal(10, 1, 10)) for i in range(4)]
    rows = single_fold_transfer(_fold_scores(same), "a", ("mean_std", 3.0))
    assert all(r["n_eval"] == 30 for r in rows) and all(r["fpr"] <= 0.1 for r in rows)
    drifted = same[:3] + [(range(30, 40), rng.normal(20, 1, 10))]
    rows = single_fold_transfer(_fold_scores(drifted), "a", ("mean_std", 3.0))
    assert rows[0]["fpr"] >= 10 / 30 - 1e-9  # calibrated on a normal block; the drifted block exceeds it


def test_pooled_transfer_calibrates_on_other_folds_and_measures_the_held_out_fold():
    rng = np.random.default_rng(1)
    sets = [(range(i * 10, (i + 1) * 10), rng.normal(10, 1, 10)) for i in range(3)]
    sets.append((range(30, 40), rng.normal(20, 1, 10)))  # shifted last fold
    rows = pooled_transfer(_fold_scores(sets), "a", ("mean_std", 3.0), group_key=lambda f: "one")
    assert len(rows) == 4 and rows[3]["fpr"] == 1.0 and rows[0]["fpr"] <= 0.2


def test_drift_diagnostics_detect_block_structure_but_not_iid_noise():
    rng = np.random.default_rng(2)
    blocks = [FoldScores(Fold("b", f"block{i}", (), tuple(range(i * 10, (i + 1) * 10))),
                         {"a": rng.normal(10 + 3 * i, 1, 10)}) for i in range(5)]
    rand = [FoldScores(Fold("r", f"seed{cs.RANDOM_KFOLD_SEEDS[0]}_fold{i}", (), tuple(range(10))),
                       {"a": rng.normal(10, 1, 10)}) for i in range(5)]
    d = drift_diagnostics(blocks, rand, "a", 50)
    assert d["kruskal_wallis_p_between_blocks"] < 0.001 and d["spearman_file_index_vs_score"]["rho"] > 0.8
    flat = [FoldScores(Fold("b", f"block{i}", (), tuple(range(i * 10, (i + 1) * 10))),
                       {"a": rng.normal(10, 1, 10)}) for i in range(5)]
    assert drift_diagnostics(flat, rand, "a", 50)["kruskal_wallis_p_between_blocks"] > 0.01


# ---------------------------------------------------------------------------
# Pre-declared selection policy
# ---------------------------------------------------------------------------

def _row(agg, policy, fpr_max, fpr_mean, recall, cov=0.1, thr=1.0):
    return ConfigRow(agg, policy, fpr_max, fpr_mean, thr, recall, cov)


def test_gate_and_recall_floor_are_required():
    rows = [_row("a", ("mean_std", 1.0), 0.30, 0.20, 0.99), _row("b", ("mean_std", 3.0), 0.08, 0.04, 0.60),
            _row("c", ("mean_std", 2.5), 0.09, 0.04, 0.80)]
    sel = select_configuration(rows)
    assert sel.winner.aggregation == "c" and sel.passes_primary_gate and sel.eligible == ["c|mean_std_2.5"]


def test_secondary_target_is_preferred_then_highest_recall():
    rows = [_row("hi_recall", ("mean_std", 2.0), 0.10, 0.09, 0.95), _row("safe", ("mean_std", 3.0), 0.06, 0.03, 0.85),
            _row("safe2", ("percentile", 99.0), 0.05, 0.02, 0.90)]
    sel = select_configuration(rows)
    assert sel.winner.aggregation == "safe2" and sel.meets_secondary_target


def test_recall_ties_go_to_lower_cov_then_simpler_policy():
    rows = [_row("a", ("percentile", 99.0), 0.05, 0.02, 0.90, cov=0.10), _row("b", ("mean_std", 3.0), 0.05, 0.02, 0.89, cov=0.05)]
    assert select_configuration(rows).winner.aggregation == "b"
    rows = [_row("a", ("percentile", 99.0), 0.05, 0.02, 0.90), _row("b", ("mean_std", 3.0), 0.05, 0.02, 0.90)]
    assert select_configuration(rows).winner.aggregation == "b"


def test_fallback_when_nothing_passes_is_flagged_not_validated():
    rows = [_row("a", ("mean_std", 3.0), 0.40, 0.30, 0.9), _row("b", ("percentile", 99.0), 0.25, 0.20, 0.8)]
    sel = select_configuration(rows)
    assert sel.winner.aggregation == "b" and not sel.passes_primary_gate and "does NOT meet" in sel.reasoning
    assert sel.eligible == []
    with pytest.raises(ValueError):
        select_configuration([])


def test_selection_is_deterministic_and_order_independent():
    rows = [_row("a", ("mean_std", 3.0), 0.05, 0.02, 0.9), _row("b", ("mean_std", 2.5), 0.05, 0.02, 0.9)]
    assert select_configuration(rows).winner == select_configuration(list(reversed(rows))).winner


def test_gates_are_the_declared_ones():
    assert (cs.PRIMARY_FPR, cs.SECONDARY_FPR, cs.RECALL_FLOOR, cs.RECALL_TIE) == (0.10, 0.05, 0.75, 0.03)
    assert cs.RANDOM_KFOLD_SEEDS == (42, 43, 44) and cs.HOLDOUT_FRACTIONS == (0.20, 0.25, 0.30)


def test_calibration_study_cannot_list_test_images():
    tree = ast.parse(Path(cs.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"discover_test_samples"}
