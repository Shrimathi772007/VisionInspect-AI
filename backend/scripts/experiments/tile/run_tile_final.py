"""Tile final test through the UNCHANGED `final` command of backend/scripts/run_model_family_study.py.

Runtime-only adaptations (no project file edited):
  * Tile's expected counts are injected into EXPECTED_BY_CATEGORY, and the folder counts are verified with the
    framework's metadata-only count_dataset_entries BEFORE anything else (STOP on mismatch).
  * model_family_final._build_comparison is replaced by a read-only adapter: the framework expects the generic
    phase1_report.json layout (["run1"]["metrics"]...), while the Tile Phase 1 report was written with
    ["final_test"]["metrics"] / ["lock"]["threshold"] / ["final_test"]["auroc"]. Without this the evaluator would
    raise AFTER scoring and BEFORE saving, forcing a forbidden second final test. Output structure is identical.
Modes:
  preflight  verify counts + verify_lock + exercise the adapter on dummy values. Never lists a test image.
  final      run cmd_final exactly once, then write the supplementary report (Wilson CIs from saved counts, hashes).
"""

import hashlib
import json
import math
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
from app.ai.training import artifacts  # noqa: E402

CATEGORY = "tile"
EXPECTED = {"train_good": 230, "test_good": 33, "test_defective": 84, "test_total": 117,
            "test_by_type": {"crack": 17, "glue_strip": 18, "good": 33, "gray_stroke": 16, "oil": 18, "rough": 15}}
runner.EXPECTED_BY_CATEGORY[CATEGORY] = EXPECTED
LOCKED_SHA = "de00e2774daf02d97e1414fe75b10305e2ad392696449b1627a250b1ec0dd4d7"
LOCK_SHA = "e2b45b716ca8f9e24c1cd87167f8f89fef0b06e611740406dcf8e1157c757c90"


def _tile_comparison(category, metrics, threshold, auroc, per_defect):
    p1 = mff._optional_json(artifacts.ARTIFACTS_ROOT / category / "phase1_baseline" / "reports" / "phase1_report.json")
    comparison, earlier = {}, {}
    if p1:
        comparison["phase1_convae"] = mff._metric_row(p1["final_test"]["metrics"], p1["lock"]["threshold"], p1["final_test"]["auroc"])
        earlier["phase1"] = p1["per_defect"]
    comparison["alternative"] = mff._metric_row(metrics, threshold, auroc)
    comparison["per_defect"] = {n: {**{ph: d[n]["recall"] for ph, d in earlier.items() if n in d}, "alternative": v["recall"],
                                    "alternative_detected": v["detected"], "total": v["total"]} for n, v in per_defect.items()}
    return comparison


mff._build_comparison = _tile_comparison


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Same 95% Wilson formula as the Tile Phase 1 runner."""
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
    counts = mfp.count_dataset_entries(CATEGORY, EXPECTED)  # raises on any mismatch - STOP
    before = locked_files()
    if before["model_state.pt"]["sha256"] != LOCKED_SHA or before["selection_lock.json"]["sha256"] != LOCK_SHA:
        sys.exit("STOP: locked artifact or lock hash differs from the recorded values.")
    return {"counts": counts, "locked_files_before": before}


def main() -> None:
    mode = sys.argv[1]
    pre = preconditions()
    if mode == "preflight":
        mff.verify_lock(CATEGORY)  # never lists a test image
        dummy = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "false_defect_detection_rate": 0,
                 "true_negatives": 0, "false_positives": 0, "false_negatives": 0, "true_positives": 0}
        c = _tile_comparison(CATEGORY, dummy, 0.0, 0.0, {"crack": {"recall": 0, "detected": 0, "total": 17}})
        json.dumps(c)
        print("PREFLIGHT OK", json.dumps(pre["counts"]), json.dumps(c["phase1_convae"]))
        return
    if mode != "final":
        sys.exit("usage: preflight | final")
    snapshot_before = json.loads((SCRATCH / "artifact_snapshot_before.json").read_text())
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runner.cmd_final(CATEGORY)  # the single final-test execution
    result = json.loads(mff.result_path(CATEGORY).read_text(encoding="utf-8"))
    m = result["metrics"]
    after = locked_files()
    supplement = {
        "experiment_id": result["experiment_id"], "command": "run_model_family_study.cmd_final('tile') via run_tile_final.py final",
        "execution_started_utc": started, "test_listed_at_unix": result["test_listed_at_unix"], "final_test_executions": 1,
        "dataset_counts_verified_before_test": pre["counts"],
        "wilson95": {"recall": wilson(m["true_positives"], m["true_positives"] + m["false_negatives"]),
                     "fpr": wilson(m["false_positives"], m["false_positives"] + m["true_negatives"]),
                     "precision": wilson(m["true_positives"], m["true_positives"] + m["false_positives"]) if m["true_positives"] + m["false_positives"] else None,
                     "accuracy": wilson(m["true_positives"] + m["true_negatives"], result["counts"]["test_total"]),
                     "per_defect_recall": {n: wilson(v["detected"], v["total"]) for n, v in result["per_defect"].items()}},
        "locked_files_before": pre["locked_files_before"], "locked_files_after": after,
        "locked_files_unchanged": {k: pre["locked_files_before"][k]["sha256"] == after[k]["sha256"] for k in after},
        "other_artifacts_unchanged_vs_study_start_snapshot": {k: v for k, v in other_artifacts().items()} == snapshot_before,
        "comparison_adapter_note": _tile_comparison.__doc__ or "phase1_convae row read from the Tile phase1_report.json layout (read-only)",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = mfp.reports_dir(CATEGORY) / "final_test_supplement.json"
    out.write_text(json.dumps(supplement, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in supplement.items() if k not in ("locked_files_before",)}, indent=1))


if __name__ == "__main__":
    main()
