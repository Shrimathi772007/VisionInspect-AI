"""The post-hoc image-level AP addendum (app/ai/ap_addendum.json) of the six ResNet-18 categories: its content and
provenance, the loader's fallbacks, and GET /ai/models. Display metadata only - nothing here touches a model or a
threshold. The consistency check against the saved final-test reports (Git-ignored ai_models/) skips when they are
absent; it reads only those reports' saved per-image scores and never opens dataset/<category>/test/."""

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import app.ai.inference.serving as serving
from app.ai.inference.serving import AP_ADDENDUM_PATH, SERVING_CONFIGS, load_ap_addendum

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_AP = {"cable": 0.99288, "leather": 1.0, "metal_nut": 0.99816, "tile": 0.99886, "toothbrush": 0.9885,
               "transistor": 0.98539}
EXPECTED_COUNTS = {"cable": (150, 92), "leather": (124, 92), "metal_nut": (115, 93), "tile": (117, 84),
                   "toothbrush": (42, 30), "transistor": (100, 40)}
# The nine WRN-50 values come from their own final-test reports and must never change with the addendum.
WRN50_AP = {"bottle": 0.9999999999999998, "capsule": 0.9951537644854546, "zipper": 0.9969916300662345,
            "wood": 0.9961573954852546, "grid": 0.9970737264738176, "pill": 0.9899105320054837,
            "carpet": 0.9967858790094236, "screw": 0.9834617248701818, "hazelnut": 0.9999999999999998}


def _addendum() -> dict:
    return json.loads(AP_ADDENDUM_PATH.read_text(encoding="utf-8"))


def test_addendum_is_tracked_app_metadata_with_provenance():
    assert AP_ADDENDUM_PATH == REPO_ROOT / "backend" / "app" / "ai" / "ap_addendum.json"
    data = _addendum()
    assert data["note"] == "computed post-hoc from saved final-test scores; no re-scoring; no selection use"
    assert "5 decimal places" in data["rounding"] and "defective = positive" in data["method"]
    assert set(data["categories"]) == set(EXPECTED_AP)
    for category, entry in data["categories"].items():
        assert entry["category"] == category
        assert entry["average_precision"] == EXPECTED_AP[category]
        assert entry["average_precision"] == round(entry["average_precision"], 5)
        assert (entry["n_images"], entry["n_defective"]) == EXPECTED_COUNTS[category]
        assert entry["source_file"] == f"backend/ai_models/{category}/model_family_study/reports/final_test_result.json"
        assert re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
        assert entry["computed_on"] == data["computed_on"] == "2026-10-07"


def test_served_ap_comes_from_the_addendum_and_wrn50_values_are_unchanged():
    assert load_ap_addendum() == EXPECTED_AP
    for category, value in EXPECTED_AP.items():
        assert SERVING_CONFIGS[category].final_test_average_precision == value
    for category, value in WRN50_AP.items():
        assert SERVING_CONFIGS[category].final_test_average_precision == value


def test_missing_or_unreadable_addendum_falls_back_to_no_ap(tmp_path):
    assert load_ap_addendum(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_ap_addendum(broken) == {}
    no_categories = tmp_path / "no_categories.json"
    no_categories.write_text(json.dumps({"note": "x"}), encoding="utf-8")
    assert load_ap_addendum(no_categories) == {}


def test_invalid_entries_are_skipped_and_missing_categories_stay_absent(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"categories": {
        "tile": {"average_precision": 0.9},
        "cable": {"average_precision": "0.9"},
        "leather": {"average_precision": 1.5},
        "metal_nut": {"average_precision": True},
        "toothbrush": "not an object",
    }}), encoding="utf-8")
    addendum = load_ap_addendum(path)
    assert addendum == {"tile": 0.9}
    assert addendum.get("transistor") is None


def test_ai_models_shows_the_ap_and_null_when_a_category_has_none(client, qe_headers, monkeypatch):
    rows = {row["category"]: row for row in client.get("/ai/models", headers=qe_headers).json()}
    for category, value in {**EXPECTED_AP, **WRN50_AP}.items():
        assert rows[category]["final_test_average_precision"] == value

    without = dataclasses.replace(SERVING_CONFIGS["tile"], final_test_average_precision=None)
    monkeypatch.setitem(serving.SERVING_CONFIGS, "tile", without)
    rows = {row["category"]: row for row in client.get("/ai/models", headers=qe_headers).json()}
    assert rows["tile"]["final_test_average_precision"] is None
    assert rows["cable"]["final_test_average_precision"] == EXPECTED_AP["cable"]


@pytest.mark.parametrize("category", sorted(EXPECTED_AP))
def test_addendum_matches_the_saved_final_test_scores(category):
    entry = _addendum()["categories"][category]
    source = REPO_ROOT / entry["source_file"]
    if not source.is_file():
        pytest.skip(f"{category} final-test report not present (ai_models/ is Git-ignored)")
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["source_sha256"]
    predictions = json.loads(raw)["predictions"]
    labels = [p["label"] for p in predictions]
    scores = [p["reconstruction_error"] for p in predictions]
    assert (len(labels), sum(labels)) == (entry["n_images"], entry["n_defective"])
    assert round(average_precision_score(labels, scores), 5) == entry["average_precision"]
    # Consistency of the saved data: AUROC from the same scores equals the served (reported) AUROC.
    assert roc_auc_score(labels, scores) == pytest.approx(SERVING_CONFIGS[category].final_test_auroc, abs=1e-12)
