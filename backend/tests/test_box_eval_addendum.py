"""The localization evaluation addendum (app/ai/box_eval_addendum.json): its content and provenance, the loader's
fallbacks, the writer script, and the box AP / pixel AUROC fields of GET /ai/models. Display metadata only - nothing
here touches a model or a threshold, and nothing reads dataset/<category>/test/ or ground_truth/."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

import app.ai.box_eval as box_eval
from app.ai.box_eval import BOX_EVAL_ADDENDUM_PATH, METRIC_KEYS, load_box_eval_addendum
from app.ai.inference.serving import SERVING_CONFIGS

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
EVAL_DIR = BACKEND_DIR / "scripts" / "experiments" / "localization_map"
NOTE = ("second scoring of final test sets for localization only; no model or threshold changed; anomaly-map boxes, "
        "not a trained detector")
WRN50 = ("bottle", "capsule", "carpet", "grid", "hazelnut", "pill", "screw", "wood", "zipper")
RESNET18 = ("cable", "leather", "metal_nut", "tile", "toothbrush", "transistor")
# MVTec AD test split sizes (images, defective images).
TEST_COUNTS = {"bottle": (83, 63), "cable": (150, 92), "capsule": (132, 109), "carpet": (117, 89), "grid": (78, 57),
               "hazelnut": (110, 70), "leather": (124, 92), "metal_nut": (115, 93), "pill": (167, 141),
               "screw": (160, 119), "tile": (117, 84), "toothbrush": (42, 30), "transistor": (100, 40),
               "wood": (79, 60), "zipper": (151, 119)}
CROP224 = {c for c, config in SERVING_CONFIGS.items() if config.input_mode == "crop224"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _addendum() -> dict:
    return json.loads(BOX_EVAL_ADDENDUM_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- The addendum file -----------------------------------------------------------------------------------------

def test_addendum_provenance():
    addendum = _addendum()
    assert addendum["note"] == NOTE
    assert addendum["protocol"] == "backend/scripts/experiments/localization_map/PROTOCOL.md"
    # The committed protocol and script are exactly the ones that produced the numbers.
    assert addendum["protocol_sha256"] == _sha256(EVAL_DIR / "PROTOCOL.md")
    assert addendum["script_sha256"] == _sha256(EVAL_DIR / "run_box_eval.py")
    raw = addendum["raw_output"]
    assert raw["path"].endswith("box_eval_2026-10-09")
    assert HEX64.match(raw["report_json_sha256"]) and HEX64.match(raw["manifest_json_sha256"])
    assert addendum["date"].startswith("2026-10-")


def test_addendum_has_every_category_with_consistent_counts():
    categories = _addendum()["categories"]
    assert set(categories) == set(SERVING_CONFIGS) == set(TEST_COUNTS)
    for category, entry in categories.items():
        assert (entry["n_images"], entry["n_defective"]) == TEST_COUNTS[category], category
        assert entry["n_good"] == entry["n_images"] - entry["n_defective"]
        for key in METRIC_KEYS:
            assert 0 <= entry[key] <= 1, (category, key)
        assert 0 <= entry["tp"] <= min(entry["gt_boxes"], entry["pred_boxes"])
        assert entry["gt_boxes_merged"] == entry["n_defective"]  # one merged box per defective image
        assert 0 <= entry["tp_merged"] <= entry["gt_boxes_merged"]
        assert entry["gt_boxes"] >= entry["n_defective"]
        assert 0 <= entry["gt_boxes_below_min_area"] <= entry["gt_boxes"]
        assert 0 <= entry["missed_images"] <= entry["n_defective"]
        assert 0 <= entry["gt_pixels_outside_analysed_area_fraction"] < 1
        if category not in CROP224:  # the whole image is analysed: nothing can fall outside
            assert entry["gt_boxes_outside_analysed_area"] == 0
            assert entry["gt_pixels_outside_analysed_area"] == 0
        assert HEX64.match(entry["lock_sha256"])


def test_addendum_means_are_the_unweighted_category_means():
    addendum = _addendum()
    categories, means = addendum["categories"], addendum["means"]
    for mean_key, key in (("map50_components", "box_ap50"), ("map50_merged", "box_ap50_merged"),
                          ("pixel_auroc", "pixel_auroc")):
        for group_key, group in (("all_15", tuple(categories)), ("wrn50_9", WRN50), ("resnet18_6", RESNET18)):
            assert means[mean_key][group_key] == pytest.approx(np.mean([categories[c][key] for c in group]))


def test_addendum_lock_hashes_match_the_local_locks_when_present():
    run_box_eval = _load(EVAL_DIR / "run_box_eval.py", "run_box_eval_for_addendum")
    checked = 0
    for category, entry in _addendum()["categories"].items():
        path = run_box_eval.lock_path_for(SERVING_CONFIGS[category])
        if path.is_file():
            assert _sha256(path) == entry["lock_sha256"], category
            checked += 1
    if not checked:
        pytest.skip("ai_models/ (Git-ignored) is not present")


# --- Loader ----------------------------------------------------------------------------------------------------

def test_loader_reads_the_tracked_file():
    loaded = load_box_eval_addendum()
    categories = _addendum()["categories"]
    assert loaded == {c: {k: categories[c][k] for k in METRIC_KEYS} for c in categories}


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"no_categories": {}}', '{"categories": []}'])
def test_loader_falls_back_to_empty(tmp_path, content):
    path = tmp_path / "addendum.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert load_box_eval_addendum(path) == {}


def test_loader_drops_out_of_range_values(tmp_path):
    path = tmp_path / "addendum.json"
    path.write_text(json.dumps({"categories": {
        "tile": {"box_ap50": 1.5, "box_ap50_merged": True, "pixel_auroc": 0.9},
        "wood": "not an entry",
    }}), encoding="utf-8")
    assert load_box_eval_addendum(path) == {"tile": {"box_ap50": None, "box_ap50_merged": None, "pixel_auroc": 0.9}}


def test_missing_category_gives_none(monkeypatch):
    monkeypatch.setattr(box_eval, "_BOX_EVAL", {})
    assert box_eval.box_eval_for("tile") == {"box_ap50": None, "box_ap50_merged": None, "pixel_auroc": None}


# --- GET /ai/models --------------------------------------------------------------------------------------------

def test_ai_models_serves_the_addendum_values(client, qe_headers):
    response = client.get("/ai/models", headers=qe_headers)
    assert response.status_code == 200
    categories = _addendum()["categories"]
    rows = response.json()
    assert len(rows) == 15
    for row in rows:
        for key in METRIC_KEYS:
            assert row[key] == pytest.approx(categories[row["category"]][key]), (row["category"], key)
    assert "sha256" not in response.text and "box_eval_2026" not in response.text


def test_ai_models_shows_none_without_the_addendum(client, qe_headers, monkeypatch):
    monkeypatch.setattr(box_eval, "_BOX_EVAL", {})
    rows = client.get("/ai/models", headers=qe_headers).json()
    assert all(row[key] is None for row in rows for key in METRIC_KEYS)
    assert all(row["final_test_auroc"] is not None for row in rows)  # the image-level metrics are untouched


# --- Writer ----------------------------------------------------------------------------------------------------

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_raw_dir(root: Path, kind: str = "real", n_categories: int = 15) -> Path:
    cats = sorted(TEST_COUNTS)[:n_categories]
    metrics = {"n_images": 2, "n_defective": 1, "n_good": 1, "missed_images": 0, "good_images_with_boxes": 0,
               "components": {"ap50": 0.5, "gt_boxes": 2, "pred_boxes": 2, "tp": 1, "gt_below_min_area": 1,
                              "gt_outside_region": 0},
               "merged": {"ap50": 1.0, "gt_boxes": 1, "pred_boxes": 2, "tp": 1, "gt_below_min_area": 0,
                          "gt_outside_region": 0},
               "pixel_auroc": 0.9, "pixel_counts": {"positive": 10, "negative": 90, "gt_pixels_outside_region": 25}}
    (root / "categories").mkdir(parents=True)
    for c in cats:
        record = {"image_size": [10, 10], "gt": {"components": [{"area_fraction": 0.5}, {"area_fraction": 0.5}]}}
        (root / "categories" / f"{c}.json").write_text(json.dumps({"records": [record]}), encoding="utf-8")
    (root / "report.json").write_text(json.dumps({
        "dataset_kind": kind, "per_category": {c: metrics for c in cats}, "summary": {"map50_components": {}},
        "seconds_per_category": {c: 1.0 for c in cats}}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "dataset_kind": kind, "protocol_sha256": "p", "script_sha256": "s", "date": "2026-10-09T00:00:00+00:00",
        "locks": {c: {"sha256": "l"} for c in cats}}), encoding="utf-8")
    return root


def test_writer_copies_metrics_and_records_hashes(tmp_path):
    writer = _load(EVAL_DIR / "write_addendum.py", "write_addendum_under_test")
    raw = _fake_raw_dir(tmp_path / "raw")
    built = writer.build(raw, "Documents/x/box_eval_2026-10-09")
    entry = built["categories"]["tile"]
    assert (entry["box_ap50"], entry["box_ap50_merged"], entry["pixel_auroc"]) == (0.5, 1.0, 0.9)
    assert entry["gt_pixels_outside_analysed_area_fraction"] == 0.25  # 25 of 100 GT pixels
    assert built["note"] == NOTE
    assert built["raw_output"]["report_json_sha256"] == _sha256(raw / "report.json")
    out = tmp_path / "addendum.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="overwrite"):
        writer.main(["--raw-dir", str(raw), "--raw-dir-label", "x", "--out", str(out)])


@pytest.mark.parametrize("kind, n", [("synthetic", 15), ("real", 14)])
def test_writer_refuses_synthetic_or_incomplete_runs(tmp_path, kind, n):
    writer = _load(EVAL_DIR / "write_addendum.py", "write_addendum_refusals")
    with pytest.raises(SystemExit, match="Refusing"):
        writer.build(_fake_raw_dir(tmp_path / "raw", kind, n), "x")
