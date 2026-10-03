"""Write model_family_study/selection_lock_supplement.json: the fields requested beyond the framework lock format.

The framework's selection_lock.json is NOT modified (its lock_digest is verified by model_family_final.verify_lock);
this supplement references it by SHA-256. Refuses to overwrite.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402

CATEGORY = "screw"
root = mfp.study_root(CATEGORY)
OUT = root / "selection_lock_supplement.json"


def main() -> None:
    if OUT.exists():
        sys.exit(f"STOP: {OUT} exists; refusing to overwrite.")
    lock = json.loads(mfp.lock_path(CATEGORY).read_text(encoding="utf-8"))
    decl = json.loads((root / "study_declaration.json").read_text(encoding="utf-8"))
    sel = mfp.selected_dir(CATEGORY)
    sup = {
        "category": CATEGORY, "study_id": decl["study_id"], "framework_experiment_id": lock["experiment_id"],
        "selection_lock_file": {"path": str(mfp.lock_path(CATEGORY)), **{k: v for k, v in file_hashes(mfp.lock_path(CATEGORY)).items() if k != "path"}},
        "selection_lock_digest": lock["lock_digest"],
        "selected_candidate": lock["selected_candidate"], "selected_policy": lock["selected_policy"],
        "locked_threshold": lock["selected_threshold"], "locked_threshold_repr": repr(lock["selected_threshold"]),
        "candidate_registry_fingerprint": lock["registry_fingerprint"],
        "policy_registry_sha256": decl["policy_registry"]["policy_registry_sha256"],
        "selection_rule_sha256": lock["selection_rule_sha256"],
        "model_state": {k: v for k, v in file_hashes(sel / "model_state.pt").items() if k != "path"},
        "model_config": {k: v for k, v in file_hashes(sel / "model_config.json").items() if k != "path"},
        "candidate_manifest": {k: v for k, v in file_hashes(sel / "candidate_manifest.json").items() if k != "path"},
        "dataset_fingerprint_sha256": lock["normal_dataset_fingerprint_sha256"],
        "split_fingerprint_sha256": lock["split_fingerprint_sha256"],
        "synthetic_diagnostic_digest_sha256": lock["synthetic_diagnostic_results"]["synthetic_digest_sha256"],
        "study_declaration_sha256": file_hashes(root / "study_declaration.json")["sha256"],
        "model_family_study_artifact_sha256": file_hashes(mfp.study_report_path(CATEGORY))["sha256"],
        "lock_timestamp_utc": lock["timestamp_utc"], "supplement_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_test_status": "NOT RUN (no Screw test image opened or decoded for this study)",
        "final_test_data_used_for_selection": lock["final_test_data_used_for_selection"],
        "statement": "No Screw final-test data (images, labels, masks, or Screw Phase 1 test results) was used for selection.",
    }
    OUT.write_text(json.dumps(sup, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sha256": file_hashes(OUT)["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
