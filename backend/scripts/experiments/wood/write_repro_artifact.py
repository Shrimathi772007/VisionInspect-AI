"""Build the first-run vs second-run value table from the saved first study and the captured second study, re-check
it against the framework's reports/reproducibility.json, and (only if all 12 checks pass) write
model_family_study/reproducibility.json. Refuses to overwrite. Reads no image.
"""

import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402

CATEGORY = "wood"
SCRATCH = Path(__file__).resolve().parent
root = mfp.study_root(CATEGORY)
OUT = root / "reproducibility.json"
canon = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
digest = lambda x: hashlib.sha256(canon(x).encode()).hexdigest()  # noqa: E731


def views(s: dict) -> dict:
    c = s["candidates"]
    return {
        "candidate_configurations": digest([s["registry"], s["registry_fingerprint"]]),
        "dataset_fingerprint": s["dataset_fingerprint_sha256"],
        "split_fingerprints": s["splits_sha256"],
        "synthetic_digest": s["synthetic"]["digest_sha256"],
        "candidate_artifact_hashes": digest({k: r["artifact"]["sha256"] for k, r in c.items()}),
        "threshold_values": digest({k: {p: e["full_pool_threshold"] for p, e in r["summary"]["policies"].items()} for k, r in c.items()}),
        "robustness_metrics": digest({k: {p: (e["worst_fold_fpr"], e["mean_fpr"], e["pooled_threshold"]) for p, e in r["summary"]["policies"].items()} for k, r in c.items()}),
        "synthetic_diagnostics": digest({k: (r["summary"]["synthetic_auroc"], {p: e["synthetic_recall"] for p, e in r["summary"]["policies"].items()}) for k, r in c.items()}),
        "full_summaries": digest({k: r["summary"] for k, r in c.items()}),
        "selection_trace": digest(s["selection"]),
        "selected_candidate": s["selection"]["winner"],
        "selected_threshold": repr(s["selection"]["threshold"]),
    }


def main() -> None:
    if OUT.exists():
        sys.exit(f"STOP: {OUT} exists; refusing to overwrite.")
    first = json.loads(mfp.study_report_path(CATEGORY).read_text(encoding="utf-8"))
    second = json.loads((SCRATCH / "second_study.json").read_text(encoding="utf-8"))
    fw = json.loads((mfp.reports_dir(CATEGORY) / "reproducibility.json").read_text(encoding="utf-8"))
    a, b = views(first), views(second)
    fw_keys = ["candidate_configurations_identical", "dataset_fingerprint_identical", "split_fingerprints_identical",
               "synthetic_digest_identical", "candidate_artifact_hashes_identical", "threshold_values_identical",
               "robustness_metrics_identical", "synthetic_diagnostics_identical", "full_summaries_identical",
               "selection_identical", "selected_candidate_identical", "selected_threshold_identical"]
    table = [{"check": name, "framework_key": key, "first_run": a[name], "second_run": b[name],
              "values_match": a[name] == b[name], "framework_result": fw[key]} for name, key in zip(a, fw_keys)]
    consistent = all(r["values_match"] == r["framework_result"] for r in table)
    passed = sum(r["framework_result"] is True for r in table)
    decodes = json.loads((SCRATCH / "decodes_repro.json").read_text(encoding="utf-8"))
    lock = json.loads(mfp.lock_path(CATEGORY).read_text(encoding="utf-8"))
    decl = json.loads((root / "study_declaration.json").read_text(encoding="utf-8"))
    result = {
        "study_id": decl["study_id"], "framework_experiment_id": lock["experiment_id"],
        "original_study_sha256": file_hashes(mfp.study_report_path(CATEGORY))["sha256"],
        "selection_lock_sha256": file_hashes(mfp.lock_path(CATEGORY))["sha256"], "selection_lock_digest": lock["lock_digest"],
        "framework_reproducibility_report": {"path": str(mfp.reports_dir(CATEGORY) / "reproducibility.json"),
                                             "sha256": file_hashes(mfp.reports_dir(CATEGORY) / "reproducibility.json")["sha256"]},
        "second_run": {"seconds": fw.get("second_run_seconds"), "second_study_canonical_sha256": digest(second),
                       "generated_independently": "complete run_study into a temporary directory from the raw train/good images"},
        "comparison_method": "exact equality of canonical JSON (framework compare_studies); no floating-point tolerance",
        "checks": table, "checks_passed": passed, "checks_failed": len(table) - passed,
        "value_table_consistent_with_framework": consistent,
        "dataset_fingerprint_sha256": first["dataset_fingerprint_sha256"], "split_fingerprint_sha256": first["splits_sha256"],
        "candidate_registry_fingerprint": first["registry_fingerprint"], "policy_registry_sha256": decl["policy_registry"]["policy_registry_sha256"],
        "synthetic_diagnostic_digest_sha256": first["synthetic"]["digest_sha256"],
        "selected_candidate": lock["selected_candidate"], "selected_policy": lock["selected_policy"],
        "selected_threshold": lock["selected_threshold"], "selected_threshold_repr": repr(lock["selected_threshold"]),
        "test_access_audit": {"second_run_decodes": decodes["decode_calls"], "all_decodes_under_train_good": decodes["all_under_train_good"],
                              "test_images_opened_or_decoded": decodes["test_files_opened_or_decoded"], "any_decode_under_test": decodes["any_under_test"],
                              "test_labels_used": 0, "ground_truth_masks_opened": 0,
                              "test_tree_directory_listings": decodes["test_tree_scandir_calls"], "listed": decodes["test_tree_dirs_listed"],
                              "final_test_scored_in_second_run": fw.get("final_test_scored_in_second_run")},
        "overall": "PASS" if passed == 12 and consistent else "FAIL",
    }
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=1))
    for r in table:
        print(f"{r['check']:28s} {str(r['first_run'])[:24]:26s} {str(r['second_run'])[:24]:26s} {r['framework_result']}")
    if result["overall"] != "PASS":
        (SCRATCH / "reproducibility_FAILED.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        sys.exit("Reproducibility did not pass; evidence preserved in scratch; reproducibility.json NOT written.")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} sha256={file_hashes(OUT)['sha256']}")


if __name__ == "__main__":
    main()
