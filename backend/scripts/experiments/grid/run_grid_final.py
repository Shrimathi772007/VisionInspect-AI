"""Grid final test through the UNCHANGED `final` command of backend/scripts/run_model_family_study.py.

This is run_capsule_final.py (itself run_cable_final.py, sha256 6f164291742e..., plus checks) with the grid constants. Additions (checks only):
  * the lock's registry fingerprint, selection-rule SHA-256, normal dataset fingerprint and
    final_test_data_used_for_selection flag are also checked against constants before any test access;
  * a hash snapshot of every artifact outside grid/model_family_study is taken immediately before scoring and
    compared after scoring (in addition to the study-start snapshot comparison).

Follows run_tile_final.py (preflight | final). Runtime-only adaptations (no project file edited):
  * Grid's (originally Cable's) expected counts are injected into EXPECTED_BY_CATEGORY and verified with the framework's metadata-only
    count_dataset_entries BEFORE anything else (STOP on mismatch).
  * model_family_final._build_comparison is replaced by the same read-only adapter Tile used: the Cable Phase 1
    report has the ["final_test"]["metrics"] / ["lock"]["threshold"] / ["final_test"]["auroc"] layout, which the
    generic comparison (["run1"]...) would crash on AFTER scoring and BEFORE saving.
Evaluator guards (Cable additions):
  * every locked value (lock SHA-256 + digest, model SHA-256/MD5, config SHA-256, manifest SHA-256, candidate,
    policy, experiment id, threshold bit-exact) is checked against constants; any difference -> STOP, no repair;
  * a one-shot sentinel is created (exclusive create) immediately before cmd_final: a second `final` is refused even
    if the first crashed; the framework also refuses when final_test_result.json exists;
  * training / study / selection / lock / artifact-save entry points are replaced by functions that raise, so the
    evaluator cannot retrain, reselect, re-lock or write a model; writes into selected_candidate/ and to
    selection_lock.json are refused at the open() level;
  * every test-tree directory listing and every test-image open/decode is logged with a timestamp, so metadata
    access before scoring can be separated from image access during scoring.
"""

import builtins
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

import run_model_family_study as runner  # noqa: E402
from app.ai.evaluation import model_family_final as mff  # noqa: E402
from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.models import patch_anomaly  # noqa: E402
from app.ai.training import artifacts  # noqa: E402

CATEGORY = "grid"
EXPECTED = {"train_good": 264, "test_good": 21, "test_defective": 57, "test_total": 78,
            "test_by_type": {"bent": 12, "broken": 12, "glue": 11, "good": 21, "metal_contamination": 11, "thread": 11}}
runner.EXPECTED_BY_CATEGORY[CATEGORY] = EXPECTED
LOCKED = {  # filled from the grid selection lock AFTER `lock` and `repro`; the run refuses while any value is unfilled
    "lock_sha256": "9f0e006162b90286402b26de561a98a90e5b27e463303fed9b9e7beb7d38d14b",
    "lock_digest": "6c4e915c30e44d87108c372babed8b863bf5a19adcde781d63354fd018eab882",
    "model_sha256": "403955ef57280f94194cc15f352a576ace8881868dca442a6532fa5ba7a56246",
    "model_md5": "8b90809056f2dc748fe55e04b3bc0c0a",
    "config_sha256": "2002f1b57b2df760aa33590b79d499883e25d729e73d713d1f9d4d53063875a0",
    "manifest_sha256": "bcb0795a348d65726608ddf1636f232ece3f0c6cd6523e3d3e77cbd83d41ecb1",
    "candidate": "knn_l2_128", "policy": "mean_std_3", "threshold": 1.222411046849779, "threshold_repr": "1.222411046849779",
    "experiment_id": "grid-model-family-grid-c8c3ec13f156",
    "registry_fingerprint": "f98bc697b8b11910177f6917c52c74d81348f0974e8ca994aac6d037ac450392",
    "selection_rule_sha256": "4e705e56d0664e8b0607ab669225551ead9ed80b45a23a2c88ca916b0a759dad",
    "normal_dataset_fingerprint": "be2b31503b84f9e2c56541ecdddfa79f3411c879b75f7c49cfd5e5347629c3c9",
}
if "__FILL__" in LOCKED.values():
    sys.exit("STOP: locked constants not filled in; refusing to run.")
SENTINEL = SCRATCH / "grid_final_test_EXECUTED.sentinel"
TEST_ROOT = (BACKEND.parent / "dataset" / CATEGORY / "test").resolve()
SELECTED_DIR = mfp.selected_dir(CATEGORY).resolve()
LOCK_FILE = mfp.lock_path(CATEGORY).resolve()
EVENTS: list[dict] = []


# ---------------------------------------------------------------- read-only comparison adapter (as Tile)
def _cable_comparison(category, metrics, threshold, auroc, per_defect):
    """phase1_convae row read (read-only) from the grid phase1_report.json (same layout as Cable's); descriptive only."""
    p1 = mff._optional_json(artifacts.ARTIFACTS_ROOT / category / "phase1_baseline" / "reports" / "phase1_report.json")
    comparison, earlier = {}, {}
    if p1:
        comparison["phase1_convae"] = mff._metric_row(p1["final_test"]["metrics"], p1["lock"]["threshold"], p1["final_test"]["auroc"])
        earlier["phase1"] = p1["per_defect"]
    comparison["alternative"] = mff._metric_row(metrics, threshold, auroc)
    comparison["per_defect"] = {n: {**{ph: d[n]["recall"] for ph, d in earlier.items() if n in d}, "alternative": v["recall"],
                                    "alternative_detected": v["detected"], "total": v["total"]} for n, v in per_defect.items()}
    return comparison


mff._build_comparison = _cable_comparison


# ---------------------------------------------------------------- evaluator guards
def _refuse(name):
    def _f(*_a, **_k):
        raise RuntimeError(f"EVALUATOR GUARD: {name} is forbidden during the final test.")
    return _f


for mod, names in ((mfp, ("run_study", "evaluate_group", "create_selection_lock", "build_lock", "load_inputs")),
                   (runner, ("cmd_study", "cmd_lock", "cmd_repro"))):
    for n in names:
        if hasattr(mod, n):
            setattr(mod, n, _refuse(f"{mod.__name__}.{n}"))
patch_anomaly.PatchAnomalyDetector.save = _refuse("PatchAnomalyDetector.save")
import app.ai.training.phase3_train as _p3  # noqa: E402
_p3.train_anomaly_model_on_samples = _refuse("train_anomaly_model_on_samples")

_real_open, _real_scandir = builtins.open, os.scandir
_real_mff_load = mff._load_image


def _resolved(path):
    try:
        return Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError):
        return None


def _under(p, root):
    return p is not None and (p == root or root in p.parents)


def _guarded_open(file, mode="r", *a, **k):
    if not isinstance(file, int):
        p = _resolved(file)
        writing = any(c in mode for c in "wax+")
        if writing and (_under(p, SELECTED_DIR) or p == LOCK_FILE):
            raise RuntimeError(f"EVALUATOR GUARD: write to locked artifact refused: {p}")
        if _under(p, TEST_ROOT):
            if writing:
                raise RuntimeError(f"EVALUATOR GUARD: write into the test tree refused: {p}")
            EVENTS.append({"t": time.time(), "kind": "open_read", "path": str(p.relative_to(TEST_ROOT.parent))})
    return _real_open(file, mode, *a, **k)


def _logged_scandir(path="."):
    p = _resolved(path)
    if _under(p, TEST_ROOT):
        EVENTS.append({"t": time.time(), "kind": "scandir_metadata", "path": str(p.relative_to(TEST_ROOT.parent))})
    return _real_scandir(path)


def _logged_load(path):
    p = _resolved(path)
    if _under(p, TEST_ROOT):
        EVENTS.append({"t": time.time(), "kind": "image_decode", "path": str(p.relative_to(TEST_ROOT.parent))})
    return _real_mff_load(path)


builtins.open = io.open = _guarded_open
os.scandir = _logged_scandir
mff._load_image = _logged_load


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Same 95% Wilson formula as the Tile/Cable Phase 1 runners."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def locked_files() -> dict:
    sel = mfp.selected_dir(CATEGORY)
    return {name: file_hashes(p) for name, p in {"model_state.pt": sel / "model_state.pt", "model_config.json": sel / "model_config.json",
            "candidate_manifest.json": sel / "candidate_manifest.json", "selection_lock.json": mfp.lock_path(CATEGORY)}.items()}


def other_artifacts() -> dict:
    own = mfp.study_root(CATEGORY).resolve()
    return {str(p.relative_to(artifacts.ARTIFACTS_ROOT)).replace("\\", "/"): file_hashes(p)["sha256"]
            for p in sorted(artifacts.ARTIFACTS_ROOT.rglob("*")) if p.is_file() and own not in p.resolve().parents}


def preconditions() -> dict:
    counts = mfp.count_dataset_entries(CATEGORY, EXPECTED)  # metadata only; raises on any mismatch -> STOP
    files = locked_files()
    lock = json.loads(mfp.lock_path(CATEGORY).read_text(encoding="utf-8"))
    checks = {
        "1_selection_lock_sha256": files["selection_lock.json"]["sha256"] == LOCKED["lock_sha256"],
        "1b_selection_lock_digest_recorded": lock["lock_digest"] == LOCKED["lock_digest"],
        "1c_selection_lock_digest_recomputed": mfp.lock_digest(lock) == LOCKED["lock_digest"],
        "2_model_sha256": files["model_state.pt"]["sha256"] == LOCKED["model_sha256"],
        "2b_model_md5": files["model_state.pt"]["md5"] == LOCKED["model_md5"],
        "3_config_sha256": files["model_config.json"]["sha256"] == LOCKED["config_sha256"],
        "4_manifest_sha256": files["candidate_manifest.json"]["sha256"] == LOCKED["manifest_sha256"],
        "5_lock_selected_candidate": lock["selected_candidate"] == LOCKED["candidate"],
        "5b_lock_selected_policy": lock["selected_policy"] == LOCKED["policy"],
        "5c_lock_experiment_id": lock["experiment_id"] == LOCKED["experiment_id"],
        "5d_lock_artifact_sha256_matches_file": lock["selected_artifact"]["sha256"] == files["model_state.pt"]["sha256"],
        "5e_manifest_candidate_id": json.loads((mfp.selected_dir(CATEGORY) / "candidate_manifest.json").read_text(encoding="utf-8"))["candidate_id"] == LOCKED["candidate"],
        "6_threshold_bit_exact": lock["selected_threshold"] == LOCKED["threshold"] and repr(lock["selected_threshold"]) == LOCKED["threshold_repr"],
        "7_registry_fingerprint": lock["registry_fingerprint"] == LOCKED["registry_fingerprint"],
        "8_selection_rule_sha256": lock["selection_rule_sha256"] == LOCKED["selection_rule_sha256"],
        "9_normal_dataset_fingerprint": lock["normal_dataset_fingerprint_sha256"] == LOCKED["normal_dataset_fingerprint"],
        "10_final_test_data_used_for_selection_is_false": lock["final_test_data_used_for_selection"] is False,
    }
    if not all(checks.values()):
        sys.exit(f"STOP: locked provenance mismatch: {[k for k, v in checks.items() if not v]}")
    return {"counts": counts, "locked_files_before": files, "precondition_checks": checks}


def main() -> None:
    mode = sys.argv[1]
    if SENTINEL.exists() or mff.result_path(CATEGORY).exists():
        sys.exit("STOP: the grid final test has already been executed; a second execution is refused.")
    pre = preconditions()
    if mode == "preflight":
        mff.verify_lock(CATEGORY)  # all framework pre-test checks; never lists a test image
        dummy = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "false_defect_detection_rate": 0,
                 "true_negatives": 0, "false_positives": 0, "false_negatives": 0, "true_positives": 0}
        c = _cable_comparison(CATEGORY, dummy, 0.0, 0.0, {"bent": {"recall": 0, "detected": 0, "total": 12}})
        json.dumps(c)
        print("PREFLIGHT OK", json.dumps(pre["precondition_checks"]), json.dumps(pre["counts"]), json.dumps(c["phase1_convae"]))
        print("test-tree events:", json.dumps(EVENTS))
        return
    if mode != "final":
        sys.exit("usage: preflight | final")

    snapshot_before = json.loads((SCRATCH / "artifact_snapshot_before.json").read_text())
    snapshot_pre_scoring = other_artifacts()
    with _real_open(SENTINEL, "x", encoding="utf-8") as fh:  # exclusive create: one shot
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runner.cmd_final(CATEGORY)  # the single final-test execution
    result = json.loads(mff.result_path(CATEGORY).read_text(encoding="utf-8"))
    m = result["metrics"]
    after = locked_files()
    listed = result["test_listed_at_unix"]
    decodes = [e for e in EVENTS if e["kind"] == "image_decode"]
    access = {
        "test_listed_at_unix": listed,
        "metadata_events_before_listing": [e for e in EVENTS if e["t"] < listed and e["kind"] == "scandir_metadata"],
        "image_opens_or_decodes_before_listing": [e for e in EVENTS if e["t"] < listed and e["kind"] != "scandir_metadata"],
        "image_decodes_during_scoring": len(decodes), "unique_images_decoded": len({e["path"] for e in decodes}),
        "first_decode_at_unix": min((e["t"] for e in decodes), default=None),
        "read_opens_after_scoring_for_leakage_hash_check": len([e for e in EVENTS if e["kind"] == "open_read" and e["t"] >= listed]),
    }
    supplement = {
        "experiment_id": result["experiment_id"], "command": "run_model_family_study.cmd_final('grid') via run_grid_final.py final",
        "execution_started_utc": started, "test_listed_at_unix": listed, "final_test_executions": 1,
        "dataset_counts_verified_before_test": pre["counts"], "precondition_checks": pre["precondition_checks"],
        "locked_provenance": LOCKED,
        "wilson95": {"recall": wilson(m["true_positives"], m["true_positives"] + m["false_negatives"]),
                     "fpr": wilson(m["false_positives"], m["false_positives"] + m["true_negatives"]),
                     "precision": wilson(m["true_positives"], m["true_positives"] + m["false_positives"]) if m["true_positives"] + m["false_positives"] else None,
                     "accuracy": wilson(m["true_positives"] + m["true_negatives"], result["counts"]["test_total"]),
                     "per_defect_recall": {n: wilson(v["detected"], v["total"]) for n, v in result["per_defect"].items()}},
        "locked_files_before": pre["locked_files_before"], "locked_files_after": after,
        "locked_files_unchanged": {k: pre["locked_files_before"][k]["sha256"] == after[k]["sha256"] for k in after},
        "other_artifacts_unchanged_vs_study_start_snapshot": other_artifacts() == snapshot_before,
        "other_artifacts_unchanged_vs_pre_scoring_snapshot": other_artifacts() == snapshot_pre_scoring,
        "other_artifacts_count": len(snapshot_pre_scoring),
        "test_tree_access": access,
        "comparison_adapter_note": _cable_comparison.__doc__,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = mfp.reports_dir(CATEGORY) / "final_test_supplement.json"
    out.write_text(json.dumps(supplement, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in supplement.items() if k not in ("locked_files_before",)}, indent=1))


if __name__ == "__main__":
    main()
