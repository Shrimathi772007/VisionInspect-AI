"""Write model_family_study/selection_lock_supplement.json (Toothbrush): the fields requested beyond the framework lock
format. The framework's selection_lock.json is NOT modified (its lock_digest is verified by verify_lock); this
supplement references it by SHA-256. Same as the Screw supplement plus the selection trace, git HEAD and execution
provenance. Refuses to overwrite.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402

CATEGORY = "toothbrush"
root = mfp.study_root(CATEGORY)
OUT = root / "selection_lock_supplement.json"


def main() -> None:
    if OUT.exists():
        sys.exit(f"STOP: {OUT} exists; refusing to overwrite.")
    lock = json.loads(mfp.lock_path(CATEGORY).read_text(encoding="utf-8"))
    decl = json.loads((root / "study_declaration.json").read_text(encoding="utf-8"))
    verification = json.loads((SCRATCH / "selection_verification.json").read_text(encoding="utf-8"))
    execution = json.loads((mfp.reports_dir(CATEGORY) / "study_execution_record.json").read_text(encoding="utf-8"))
    if not verification["all_passed"] or verification["saved_winner"] != f"{lock['selected_candidate']}|{lock['selected_policy']}" \
            or verification["threshold"] != lock["selected_threshold"]:
        sys.exit("STOP: verification record does not match the lock")
    sel = mfp.selected_dir(CATEGORY)
    strip = lambda d: {k: v for k, v in d.items() if k != "path"}  # noqa: E731
    sup = {
        "category": CATEGORY, "study_id": decl["study_id"], "framework_experiment_id": lock["experiment_id"],
        "selection_lock_file": {"path": str(mfp.lock_path(CATEGORY)), **strip(file_hashes(mfp.lock_path(CATEGORY)))},
        "selection_lock_digest": lock["lock_digest"],
        "selected_candidate": lock["selected_candidate"], "selected_policy": lock["selected_policy"],
        "locked_threshold": lock["selected_threshold"], "locked_threshold_repr": repr(lock["selected_threshold"]),
        "candidate_registry_fingerprint": lock["registry_fingerprint"],
        "policy_registry_sha256": decl["policy_registry"]["policy_registry_sha256"],
        "selection_rule_sha256": lock["selection_rule_sha256"],
        "model_state": strip(file_hashes(sel / "model_state.pt")),
        "model_config": strip(file_hashes(sel / "model_config.json")),
        "candidate_manifest": strip(file_hashes(sel / "candidate_manifest.json")),
        "dataset_fingerprint_sha256": lock["normal_dataset_fingerprint_sha256"],
        "normal_file_list_sha256": lock["normal_file_list_sha256"],
        "split_fingerprint_sha256": lock["split_fingerprint_sha256"],
        "synthetic_diagnostic_digest_sha256": lock["synthetic_diagnostic_results"]["synthetic_digest_sha256"],
        "study_declaration_sha256": file_hashes(root / "study_declaration.json")["sha256"],
        "model_family_study_artifact_sha256": file_hashes(mfp.study_report_path(CATEGORY))["sha256"],
        "selection_trace": {"framework_reasoning": lock["selection_reasoning"], "independent_trace": verification["independent_trace"],
                            "verification_checks": verification["checks"], "framework_and_independent_agree": True},
        "provenance": {"framework_git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND.parent, capture_output=True, text=True).stdout.strip(),
                       "study_execution_id": execution["execution_id"], "study_started_utc": execution["started_utc"],
                       "study_ended_utc": execution["ended_utc"], "environment": execution["environment"],
                       "run_model_family_study.py_sha256": file_hashes(BACKEND / "scripts" / "run_model_family_study.py")["sha256"],
                       "wrapper_sha256": file_hashes(SCRATCH / "run_toothbrush_model_family.py")["sha256"],
                       "verifier_sha256": file_hashes(SCRATCH / "verify_toothbrush_selection.py")["sha256"]},
        "lock_timestamp_utc": lock["timestamp_utc"], "supplement_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_test_status": "NOT RUN for the model-family candidate (no Toothbrush test image opened or decoded for this study)",
        "final_test_data_used_for_selection": lock["final_test_data_used_for_selection"],
        "statement": "No Toothbrush final-test data (images, labels, masks, or Phase 1 test results) was used for selection.",
    }
    OUT.write_text(json.dumps(sup, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sha256": file_hashes(OUT)["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
