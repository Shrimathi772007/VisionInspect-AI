"""Addendum 1, the Grid full320 candidate (scripts/experiments/patchcore_wrn50/grid_full320_logic.py).

No real images and no weights: the combined 18-row selection rule (an existing row wins exact ties), full320
patch counts and preprocessing, the config wrapper and its loader, the train/good-only path guard, and the
equivalence of the memory-light fold buffer and streaming final fit with the all-in-memory method.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from app.ai.models.patch_anomaly import greedy_coreset_indices
from app.ai.models.patchcore import PatchCoreConfig, PatchCoreDetector, PatchCoreModelError

STUDY_DIR = Path(__file__).resolve().parent.parent / "scripts" / "experiments" / "patchcore_wrn50"
sys.path.insert(0, str(STUDY_DIR))

import grid_full320_logic as g  # noqa: E402
import wrn50_study_logic as logic  # noqa: E402


def row(mode, agg="max", policy="mean_std_2.5", recall=0.8, worst=0.05, mean=0.02, threshold=1.0):
    return {"mode": mode, "aggregation": agg, "policy": policy, "synthetic_recall": recall, "worst_fold_fpr": worst,
            "mean_fold_fpr": mean, "threshold": threshold, "meets_fpr_limit": worst <= logic.FPR_LIMIT + logic.FPR_EPSILON}


def picked(decision):
    return decision["mode"], decision["aggregation"], decision["policy"]


# ---------------------------------------------------------------------------
# Combined selection rule
# ---------------------------------------------------------------------------

def test_existing_locked_row_wins_an_exact_tie_with_full320():
    locked = row("crop224", recall=0.853219696969697, worst=0.03773584905660377, mean=0.0151669, threshold=1.841)
    challenger = row("full320", recall=0.853219696969697, worst=0.03773584905660377, mean=0.001, threshold=9.0)
    assert picked(g.select_combined([challenger, locked])) == ("crop224", "max", "mean_std_2.5")


def test_full320_wins_only_when_strictly_better():
    locked = row("crop224", recall=0.853)
    assert picked(g.select_combined([locked, row("full320", "top1pct_mean", recall=0.8531)]))[0] == "full320"
    assert picked(g.select_combined([locked, row("full320", recall=0.853, worst=0.06)]))[0] == "crop224"
    assert picked(g.select_combined([locked, row("full320", recall=0.853, worst=0.049)]))[0] == "full320"


def test_mode_order_is_crop224_then_full256_then_full320_then_max():
    rows = [row("full320"), row("full256", "top1pct_mean"), row("full256")]
    assert picked(g.select_combined(rows)) == ("full256", "max", "mean_std_2.5")
    assert picked(g.select_combined([row("full320", "top1pct_mean"), row("full320")]))[1] == "max"


def test_ineligible_full320_with_higher_recall_loses():
    assert picked(g.select_combined([row("crop224", recall=0.5), row("full320", recall=0.99, worst=0.11)]))[0] == "crop224"


def test_no_eligible_row_falls_back_to_lowest_worst_fold_fpr():
    decision = g.select_combined([row("crop224", worst=0.2), row("full320", worst=0.15, recall=0.1)])
    assert decision["mode"] == "full320" and decision["any_combination_met_fpr_limit"] is False


def test_combined_rule_equals_study_rule_on_the_existing_modes():
    rng = np.random.default_rng(3)
    for _ in range(200):
        rows = [row(mode, agg, policy, recall=float(rng.choice([0.8, 0.85])), worst=float(rng.choice([0.04, 0.09, 0.12])),
                    mean=float(rng.choice([0.01, 0.02])), threshold=float(rng.choice([1.0, 2.0])))
                for mode in logic.MODES for agg in logic.AGGREGATIONS for policy in logic.POLICIES]
        assert g.select_combined(rows) == {**logic.select(rows), "reason": g.select_combined(rows)["reason"]}


def test_full320_rows_match_the_study_table_builder():
    rng = np.random.default_rng(0)
    folds = [i % 5 for i in range(30)]
    data = {agg: {"oof": rng.normal(1, 0.1, 30).tolist(), "folds": folds,
                  "synthetic": [None] + rng.normal(1.3, 0.2, 39).tolist(),
                  "synthetic_types": [logic.DEFECT_TYPES[i % 5] for i in range(40)]} for agg in logic.AGGREGATIONS}
    study_rows = logic.build_selection_table({"crop224": data, "full256": data})[:6]
    ours = g.build_full320_rows(data)
    assert len(ours) == 6
    for a, b in zip(ours, study_rows):
        assert a["mode"] == "full320" and {**a, "mode": "crop224"} == b


# ---------------------------------------------------------------------------
# full320 patch counts, preprocessing, config wrapper
# ---------------------------------------------------------------------------

def test_full320_patch_counts_and_top1pct():
    config = g.Full320Config()
    assert (config.input_size, config.grid_side, config.patches_per_image) == (320, 40, 1600)
    detector = PatchCoreDetector(config, extractor=nn.Identity())
    scores = torch.arange(1600, dtype=torch.float32).reshape(1, 1600)
    assert float(detector.image_scores(scores, "top1pct_mean")[0]) == pytest.approx(float(torch.arange(1584, 1600).float().mean()))
    assert float(detector.image_scores(scores, "max")[0]) == 1599.0


def test_full320_config_validation_and_existing_config_unchanged():
    with pytest.raises(ValueError):
        g.Full320Config(preprocessing="crop224")
    with pytest.raises(ValueError):
        g.Full320Config(input_size=256)
    with pytest.raises(ValueError):
        g.Full320Config(aggregation="median")
    with pytest.raises(ValueError):
        PatchCoreConfig(preprocessing="full320")  # patchcore.py itself is not extended
    assert g.Full320Config.from_dict(g.Full320Config(aggregation="top1pct_mean").to_dict()) == g.Full320Config(aggregation="top1pct_mean")


def test_resize_full320_is_whole_image_inter_area_rgb():
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = g.resize_full320(image)
    assert out.shape == (320, 320, 3)
    expected = cv2.cvtColor(cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    assert np.array_equal(out, expected)


def test_full320_model_save_and_load_roundtrip(tmp_path):
    bank = torch.randn(12, 1536, generator=torch.Generator().manual_seed(2))
    PatchCoreDetector(g.Full320Config(aggregation="top1pct_mean"), extractor=nn.Identity(), bank=bank).save(tmp_path)
    loaded = g.load_full320_detector(tmp_path, extractor=nn.Identity())
    assert torch.equal(loaded.bank, bank) and loaded.config == g.Full320Config(aggregation="top1pct_mean")
    with pytest.raises(ValueError):
        PatchCoreDetector.load(tmp_path, extractor=nn.Identity())  # the unchanged loader refuses full320 ...
    (tmp_path / "model_state.pt").write_bytes(b"tampered")
    with pytest.raises(PatchCoreModelError):
        g.load_full320_detector(tmp_path, extractor=nn.Identity())  # ... and ours keeps the integrity checks


# ---------------------------------------------------------------------------
# Path guard (the study's single path function is reused)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["../../test/broken/000.png", "..", "a/b.png", "ground_truth"])
def test_grid_path_guard_refuses_anything_outside_train_good(tmp_path, filename):
    with pytest.raises(logic.DatasetAccessError):
        logic.train_good_path(g.CATEGORY, filename, dataset_root=tmp_path)


def test_grid_path_guard_returns_train_good_only(tmp_path):
    assert logic.train_good_path(g.CATEGORY, "000.png", dataset_root=tmp_path) == \
        (tmp_path / "grid" / "train" / "good" / "000.png").resolve()


# ---------------------------------------------------------------------------
# Memory-light methods == all-in-memory method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fold", range(5))
def test_fold_buffer_bank_equals_all_in_memory_bank(fold):
    n, patches, dim = 13, 20, 96
    features = torch.randn(n, patches, dim, generator=torch.Generator().manual_seed(fold))
    flat = features.view(n * patches, dim)
    train = [i for i in range(n) if logic.fold_of(i) != fold]
    reference = flat[logic.greedy_coreset_over_images(flat, train, patches)]
    buffer = torch.empty((len(train), patches, dim))
    for slot, i in enumerate(train):
        buffer[slot] = features[i]
    assert torch.equal(g.fold_bank_from_buffer(buffer), reference)


@pytest.mark.parametrize("dim", [16, 96])
def test_streaming_final_fit_equals_all_in_memory(dim):
    n, patches = 9, 25
    features = torch.randn(n, patches, dim, generator=torch.Generator().manual_seed(dim))
    flat = features.view(n * patches, dim)
    selected = g.streaming_coreset_select(iter(features), n, patches, dim)
    assert torch.equal(selected, logic.greedy_coreset_over_images(flat, list(range(n)), patches))
    assert torch.equal(selected, greedy_coreset_indices(flat, max(1, round(n * patches * 0.10)), seed=0))
    assert torch.equal(g.gather_selected_rows(iter(features), selected, patches, dim), flat[selected])


def test_streaming_select_rejects_wrong_block_count():
    features = torch.randn(3, 4, 8)
    with pytest.raises(ValueError):
        g.streaming_coreset_select(iter(features), 4, 4, 8)


def test_iter_image_features_matches_extract_features_into():
    class Fake(nn.Module):  # deterministic per-image "backbone" with layer2/layer3-shaped outputs
        def forward(self, x, layers):
            return {2: x[:, :1].repeat(1, 4, 1, 1)[:, :, ::8, ::8], 3: x[:, 1:2].repeat(1, 4, 1, 1)[:, :, ::16, ::16]}

    detector = PatchCoreDetector(g.Full320Config(), extractor=Fake())
    images = [torch.randn(3, 320, 320, generator=torch.Generator().manual_seed(i)) for i in range(11)]
    reference = detector.extract_features_into(iter(images), len(images))
    streamed = torch.stack(list(g.iter_image_features(detector, iter(images))))
    assert reference.shape[1] == 1600 and torch.equal(streamed, reference)
