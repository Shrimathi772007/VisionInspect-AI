"""scripts/experiments/localization_map/run_box_eval.py (PROTOCOL.md there): hand-checked metric cases, the
--allow-test guard, equivalence with the served localization steps, and an end-to-end run on synthetic data with a
fake predictor. No test reads dataset/<category>/test/ or ground_truth/, loads a model or touches the database."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND_DIR / "scripts" / "experiments" / "localization_map" / "run_box_eval.py"
BUILDER = SCRIPT.parent / "build_synthetic_dataset.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


be = _load(SCRIPT, "run_box_eval_under_test")


# --- IoU -------------------------------------------------------------------------------------------------------

def test_iou_hand_checked():
    assert be.iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert be.iou([0, 0, 2, 2], [2, 0, 4, 2]) == 0.0  # touching edges, no overlap
    assert be.iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(2 / 6)
    assert be.iou([0, 0, 4, 4], [1, 1, 3, 3]) == pytest.approx(4 / 16)  # contained
    assert be.iou([1, 1, 1, 3], [0, 0, 2, 2]) == 0.0  # zero-area box


# --- AP --------------------------------------------------------------------------------------------------------

def test_average_precision_hand_checked():
    # TP, FP, TP with 2 GT: precision envelope 1.0 up to recall 0.5, 2/3 up to 1.0 -> 0.5 + 0.5 * 2/3.
    assert be.average_precision([1, 0, 1], 2) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert be.average_precision([1, 1], 2) == 1.0
    assert be.average_precision([0, 1], 2) == pytest.approx(0.25)  # recall 0.5 reached at precision 0.5
    assert be.average_precision([1], 4) == pytest.approx(0.25)  # misses lower recall
    assert be.average_precision([], 3) == 0.0
    assert be.average_precision([0, 0], 0) is None


# --- GT boxes --------------------------------------------------------------------------------------------------

def test_gt_boxes_from_a_multi_piece_mask():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[1:3, 1:4] = 255  # piece A: x 1..3, y 1..2 (6 px)
    mask[3, 4] = 255  # touches A diagonally -> same 8-connected component
    mask[6:9, 10:15] = 1  # piece B: any value > 0 is a defect (15 px)
    components = be.gt_boxes_from_mask(mask)
    assert sorted(g["box"] for g in components) == [[1, 1, 5, 4], [10, 6, 15, 9]]
    assert sorted(g["area_fraction"] for g in components) == pytest.approx([7 / 200, 15 / 200])
    merged = be.gt_boxes_from_mask(mask, merged=True)
    assert merged == [{"box": [1, 1, 15, 9], "area_fraction": pytest.approx(22 / 200)}]
    assert be.gt_boxes_from_mask(np.zeros((5, 5), dtype=np.uint8)) == []


# --- Matching --------------------------------------------------------------------------------------------------

def test_matching_is_one_to_one_and_per_image():
    gt = [0, 0, 10, 10]
    images = [
        {"pred": [(gt, 3.0), ([0, 0, 10, 9], 2.0)], "gt": [gt]},  # duplicate on the same GT -> one TP, one FP
        {"pred": [(gt, 1.5)], "gt": []},  # the same box on another image is not matched to image 0's GT
    ]
    result = be.match_category(images)
    assert result["tp"] == [1, 0, 0]
    assert (result["n_gt"], result["n_pred"], result["n_tp"]) == (1, 3, 1)
    assert result["ap"] == 1.0  # the TP comes first, so precision 1.0 at recall 1.0


def test_matching_sorts_by_confidence_and_needs_iou_half():
    images = [{"pred": [([0, 0, 10, 10], 0.9), ([50, 50, 60, 60], 1.8), ([0, 0, 10, 4], 2.0)],
               "gt": [[0, 0, 10, 10]]}]
    result = be.match_category(images)
    # Order 2.0 (IoU 0.4 -> FP), 1.8 (no overlap -> FP), 0.9 (IoU 1 -> TP).
    assert result["tp"] == [0, 0, 1]
    assert result["ap"] == pytest.approx(1 / 3)


def _record(defect_type, boxes, gt_components):
    gt = {"components": [{"box": b, "area_fraction": a} for b, a in gt_components],
          "merged": [{"box": b, "area_fraction": a} for b, a in gt_components[:1]]}
    return {"defect_type": defect_type, "boxes": [{"box_px": b, "confidence": c} for b, c in boxes], "gt": gt}


def test_good_image_boxes_are_false_positives_and_missed_images_are_misses():
    records = [
        _record("crack", [([0, 0, 10, 10], 2.0)], [([0, 0, 10, 10], 0.01)]),
        _record("good", [([20, 20, 30, 30], 3.0)], []),  # highest confidence, on a good image -> FP first
        _record("crack", [], [([5, 5, 6, 6], 0.001)]),  # predicted good: missed, and below the 0.25% minimum
    ]
    pos, neg = np.zeros(be.HIST_SIZE, dtype=np.int64), np.zeros(be.HIST_SIZE, dtype=np.int64)
    metrics = be.category_metrics(records, pos, neg, 0)
    assert (metrics["n_images"], metrics["n_defective"], metrics["n_good"]) == (3, 2, 1)
    assert metrics["missed_images"] == 1 and metrics["good_images_with_boxes"] == 1
    components = metrics["components"]
    assert (components["gt_boxes"], components["pred_boxes"], components["tp"]) == (2, 2, 1)
    assert components["gt_below_min_area"] == 1
    # Order: FP (good image), TP -> recall 0.5 at precision 0.5.
    assert components["ap50"] == pytest.approx(0.25)


# --- Pixel AUROC -----------------------------------------------------------------------------------------------

def test_histogram_bins_edges():
    bins = be.histogram_bins(np.array([-1.0, 0.0, 1e-4, 1e-4 * 1.0005, 1.0, 1e3, 1e3 * 1.01, np.nan]))
    assert bins[:3].tolist() == [0, 0, 0]  # <= 1e-4 (incl. <= 0) -> underflow
    assert bins[3] == 1
    assert bins[4] == be.HIST_BINS * 4 // 7 + 1  # log10(1) = 0 sits 4/7 of the way through [-4, 3]
    assert bins[5] == be.HIST_BINS and bins[6] == be.HIST_BINS + 1 and bins[7] == 0


def test_pixel_auroc_matches_sklearn_on_a_synthetic_case():
    rng = np.random.default_rng(0)
    labels = rng.random(20_000) < 0.1
    ratio = np.exp(rng.normal(0.0, 0.5, labels.size) + labels * 0.8)
    pos, neg = np.zeros(be.HIST_SIZE, dtype=np.int64), np.zeros(be.HIST_SIZE, dtype=np.int64)
    be.histogram_add(pos, neg, ratio, labels)
    assert be.histogram_auroc(pos, neg) == pytest.approx(roc_auc_score(labels, ratio), abs=1e-3)


def test_pixel_auroc_perfect_tied_and_undefined():
    pos, neg = np.zeros(be.HIST_SIZE, dtype=np.int64), np.zeros(be.HIST_SIZE, dtype=np.int64)
    be.histogram_add(pos, neg, np.array([5.0, 6.0, 0.1, 0.2]), np.array([1, 1, 0, 0]))
    assert be.histogram_auroc(pos, neg) == 1.0
    pos[:], neg[:] = 0, 0
    be.histogram_add(pos, neg, np.array([1.0, 1.0]), np.array([1, 0]))
    assert be.histogram_auroc(pos, neg) == 0.5  # same bin -> tie
    assert be.histogram_auroc(np.zeros(be.HIST_SIZE), neg) is None


def test_mask_to_map_crops_to_the_region_and_counts_outside_pixels():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:5, 0:5] = 255  # 25 px outside the central region
    mask[40:60, 40:60] = 255
    resized, outside = be.mask_to_map(mask, (0.1, 0.1, 0.9, 0.9), 40)
    assert outside == 25
    assert resized.shape == (40, 40) and resized.dtype == bool
    assert resized[20, 20] and not resized[2, 2]
    assert resized.sum() == pytest.approx(20 * 20 * (40 / 80) ** 2, abs=20)


# --- Guard -----------------------------------------------------------------------------------------------------

def test_the_real_dataset_needs_allow_test(tmp_path):
    real = be.REPO_ROOT / "dataset"
    with pytest.raises(be.TestSplitRefused, match="--allow-test"):
        be.check_dataset_root(real, allow_test=False)
    assert be.check_dataset_root(real, allow_test=True) == "real"
    with pytest.raises(be.TestSplitRefused, match="neither"):
        be.check_dataset_root(tmp_path, allow_test=True)  # unmarked: refused even with the flag
    (tmp_path / be.SYNTHETIC_MARKER).write_text("x")
    assert be.check_dataset_root(tmp_path, allow_test=False) == "synthetic"


def test_main_refuses_before_reading_anything(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not be reached")

    monkeypatch.setattr(be, "list_images", forbidden)
    monkeypatch.setattr(be, "current_hashes", forbidden)
    assert be.main(["--out-dir", "unused", "--categories", "tile"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_the_script_never_imports_the_database():
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('m', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "import app.ai.inference, app.ai.inference.serving; "
            "print('app.database' in sys.modules, 'sqlalchemy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], cwd=BACKEND_DIR, capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False False"


# --- Served localization equivalence ---------------------------------------------------------------------------

def _fake_result(grid: np.ndarray, threshold: float, prediction: str, input_mode="resize", side=64, score=None):
    return SimpleNamespace(patch_scores=grid, input_mode=input_mode, input_size=(side, side), prediction=prediction,
                           threshold=threshold, reconstruction_error=score if score is not None else float(grid.max()),
                           model_name="fake")


def _peaked_grid(side=16, at=(4, 10), peak=5.0):
    grid = np.full((side, side), 0.2, dtype=np.float32)
    grid[at[0]:at[0] + 3, at[1]:at[1] + 3] = peak
    return grid


@pytest.mark.parametrize("input_mode", ["resize", "crop224"])
def test_served_localization_equals_the_service_steps(tmp_path, monkeypatch, input_mode):
    import app.inspections.service as service

    image = tmp_path / "img.png"
    cv2.imwrite(str(image), np.full((90, 120, 3), 128, dtype=np.uint8))
    side = 224 if input_mode == "crop224" else 64
    result = _fake_result(_peaked_grid(), threshold=1.0, prediction="defective", input_mode=input_mode, side=side)
    monkeypatch.setattr(service.localization, "write_png_atomic", lambda rgba, path: None)
    monkeypatch.setattr(service, "add_defect_form", lambda located, w, h: None)
    expected, _path = service._localize_and_render(SimpleNamespace(id=1), result, image)

    _r, _amap, region, located, size = be.served_localization(image, "tile", predict=lambda *a, **k: result)
    assert located["boxes"] and located["boxes"] == expected["boxes"]
    assert located["mask_rule"] == expected["mask_rule"]
    assert [round(v, 6) for v in region] == expected["analysed_region"] and list(size) == expected["image_size"]


def test_good_predictions_get_no_boxes(tmp_path):
    image = tmp_path / "img.png"
    cv2.imwrite(str(image), np.full((64, 64, 3), 128, dtype=np.uint8))
    result = _fake_result(_peaked_grid(), threshold=10.0, prediction="good")
    _r, _amap, _region, located, _size = be.served_localization(image, "tile", predict=lambda *a, **k: result)
    assert located["boxes"] == []


# --- End to end on synthetic data (fake predictor, no model) -----------------------------------------------------

def _synthetic_dataset(root: Path) -> None:
    for name in ("000", "001"):
        good = root / "tile" / "test" / "good"
        good.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(good / f"{name}.png"), np.full((64, 64, 3), 100, dtype=np.uint8))
    defect = root / "tile" / "test" / "crack"
    masks = root / "tile" / "ground_truth" / "crack"
    defect.mkdir(parents=True)
    masks.mkdir(parents=True)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image[16:40, 28:52] = 250  # the "defect" the fake predictor finds
    cv2.imwrite(str(defect / "000.png"), image)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:40, 28:52] = 255
    cv2.imwrite(str(masks / "000_mask.png"), mask)
    (root / be.SYNTHETIC_MARKER).write_text("synthetic")


def _fake_predict(path, category, return_patch_scores=False):
    assert return_patch_scores is True and category == "tile"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    grid = cv2.resize((image > 200).astype(np.float32) * 5.0 + 0.1, (16, 16), interpolation=cv2.INTER_AREA)
    defective = bool(grid.max() > 1.0)
    return _fake_result(grid, threshold=1.0, prediction="defective" if defective else "good", side=64)


FAKE_HASHES = {"script_sha256": "s", "protocol_sha256": "p", "locks": {"tile": {"path": "lock.json", "sha256": "l"}}}


def test_run_end_to_end_and_resume(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    _synthetic_dataset(data)
    monkeypatch.setattr(be, "current_hashes", lambda categories: FAKE_HASHES)
    report = be.run(out, ["tile"], data, allow_test=False, predict=_fake_predict, log=lambda m: None)

    metrics = report["per_category"]["tile"]
    assert report["dataset_kind"] == "synthetic"
    assert (metrics["n_images"], metrics["n_defective"], metrics["missed_images"]) == (3, 1, 0)
    assert metrics["components"]["tp"] == 1 and metrics["components"]["ap50"] == 1.0
    assert metrics["merged"]["ap50"] == 1.0
    assert metrics["pixel_auroc"] > 0.95
    assert report["summary"]["map50_components"]["all_15"] == 1.0
    assert report["summary"]["map50_components"]["wrn50_9"] is None
    assert "second scoring of final test sets" in report["note"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["script_sha256"] == "s" and set(manifest["app_files"]) == set(be.APP_FILES)

    # Resume: a finished category is reused, never recomputed.
    monkeypatch.setattr(be, "evaluate_category", lambda *a, **k: pytest.fail("recomputed a finished category"))
    assert be.run(out, ["tile"], data, allow_test=False, log=lambda m: None)["per_category"]["tile"] == metrics

    # Different hashes: refuse to mix results.
    monkeypatch.setattr(be, "current_hashes", lambda categories: {**FAKE_HASHES, "script_sha256": "changed"})
    with pytest.raises(RuntimeError, match="refusing to mix"):
        be.run(out, ["tile"], data, allow_test=False, log=lambda m: None)


def test_a_defective_image_without_a_mask_stops_the_category(tmp_path):
    _synthetic_dataset(tmp_path)
    (tmp_path / "tile" / "ground_truth" / "crack" / "000_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="without a mask"):
        be.list_images(tmp_path, "tile")


def test_synthetic_builder_reads_train_good_only(tmp_path, monkeypatch):
    builder = _load(BUILDER, "build_synthetic_under_test")
    source = tmp_path / "source"
    train = source / "tile" / "train" / "good"
    train.mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(str(train / f"{i:03d}.png"), np.full((50, 50, 3), 90, dtype=np.uint8))
    out = tmp_path / "synthetic"
    builder.build(out, ["tile"], 2, source_root=source)
    assert sorted(p.name for p in (out / "tile" / "test" / "good").iterdir()) == ["000.png", "001.png"]
    mask = cv2.imread(str(out / "tile" / "ground_truth" / "synthetic_patch" / "000_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask.sum() > 0 and (out / be.SYNTHETIC_MARKER).is_file()
    assert be.check_dataset_root(out, allow_test=False) == "synthetic"
