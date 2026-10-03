"""Write backend/ai_models/screw/model_family_study/study_declaration.json BEFORE the study decodes anything.

Values come from the live, unchanged framework modules. Reads train/good file BYTES only (fingerprint, no decode);
the test tree is touched only by the framework's directory-entry count. Refuses to overwrite.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_study as mfs  # noqa: E402
from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.anomaly_model_selection import SYNTHETIC_SEED  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES, SEVERITY_NAMES  # noqa: E402
from app.ai.models.resnet18 import RESNET18_SHA256  # noqa: E402
from app.ai.training.dataset import discover_train_samples  # noqa: E402

CATEGORY = "screw"
STUDY_ID = "screw-model-family-2026-09-28"
EXPECTED = {"train_good": 320, "test_good": 41, "test_defective": 119, "test_total": 160,
            "test_by_type": {"good": 41, "manipulated_front": 24, "scratch_head": 24, "scratch_neck": 25, "thread_side": 23, "thread_top": 23}}
SCRATCH = Path(__file__).resolve().parent
OUT = mfp.study_root(CATEGORY) / "study_declaration.json"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def main() -> None:
    if OUT.exists() or mfp.study_report_path(CATEGORY).exists() or mfp.candidates_dir(CATEGORY).exists():
        sys.exit("STOP: screw model-family outputs already exist; refusing to overwrite.")
    counts = mfp.count_dataset_entries(CATEGORY, EXPECTED)  # directory entries only
    samples = discover_train_samples(CATEGORY)
    names = [s.path.name for s in samples]
    n = len(samples)
    schemes = mfs.scheme_splits(n)
    specs = mfs.registry()
    configs = [s.config() for s in specs]
    policies = mfs.policy_ids()
    rule_text = mfs.__doc__
    split_membership = {f"{k}/{s['name']}": {"held_out_indices": s["held_out"].tolist(),
                                             "held_out_files": [names[i] for i in s["held_out"]]} for k, v in schemes.items() for s in v}
    mod = lambda m: sha(Path(m.__file__).read_bytes())  # noqa: E731
    import app.ai.models.patch_anomaly as pa
    import app.ai.evaluation.synthetic_defects as sd
    import app.ai.evaluation.category_phase2 as cp2
    decl = {
        "category": CATEGORY, "study_id": STUDY_ID, "phase": "model_family_study",
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_data": {"definition": "dataset/screw/train/good only (320 images, file-name sorted)",
                             "dataset_fingerprint_sha256": mfp.dataset_fingerprint(samples, counts),
                             "normal_file_list_sha256": sha("\n".join(names).encode()), "counts_from_directory_entries": counts},
        "held_out_split_definition": {"schemes": {k: len(v) for k, v in schemes.items()}, "total_splits": sum(len(v) for v in schemes.values()),
                                      "held_out_per_split": int(round(n * 0.2)),
                                      "seeds": {"holdout_80_20": list(mfs.HOLDOUT_SEEDS), "random_kfold5": list(mfs.RANDOM_KFOLD_SEEDS),
                                                "contiguous_kfold5": "deterministic (sorted file-name blocks)"},
                                      "expected_split_fingerprint_sha256": mfs.splits_fingerprint(schemes),
                                      "split_audit": mfs.split_audit(n, schemes), "membership": split_membership,
                                      "calibration_scoring": "leave-one-image-out within calibration; held-out scored by the calibration model"},
        "candidate_registry": {"candidates": configs, "count": len(specs),
                               "per_candidate_config_sha256": {c["candidate_id"]: sha(json.dumps(c, sort_keys=True).encode()) for c in configs},
                               "registry_fingerprint": mfs.registry_fingerprint(specs), "backbone_sha256": RESNET18_SHA256},
        "policy_registry": {"policies": policies, "count": len(policies),
                            "definitions": "mean_std_K = mean + K*population std; percentile_P = P-th percentile (linear) of calibration scores",
                            "user_alias_note": "p95/p97/p98/p99 in the instruction = percentile_95/97/98/99 here (same definitions)",
                            "policy_registry_sha256": sha(json.dumps(policies).encode())},
        "selection_rule": {"text_is_framework_docstring": True, "declared_rule_sha256": sha(rule_text.encode("utf-8")),
                           "gate_worst_fold_fpr_max": mfs.GATE_FPR, "preferred_synthetic_recall_floor": mfs.SYNTHETIC_FLOOR,
                           "recall_tie": mfs.RECALL_TIE, "cv_tie": mfs.CV_TIE,
                           "fallback": "if no gate passer meets the floor, keep all gate passers and flag floor_not_met (framework rule)"},
        "synthetic_diagnostic_protocol": {"source": f"held-out images of {'/'.join(mfs.CANONICAL_SPLIT)}", "variants_per_image": len(DEFECT_TYPES) * len(SEVERITIES),
                                          "expected_images": int(round(n * 0.2)) * len(DEFECT_TYPES) * len(SEVERITIES),
                                          "defect_types": list(DEFECT_TYPES), "severities": {str(v): SEVERITY_NAMES[v] for v in SEVERITIES},
                                          "seed": f"SYNTHETIC_SEED ({SYNTHETIC_SEED}) + source image index",
                                          "label": "DIAGNOSTIC ONLY - not real Screw defect recall"},
        "final_test_isolation": {"statement": "No file under dataset/screw/test or dataset/screw/ground_truth is opened, decoded or scored; "
                                              "only directory entries are counted. Guard refuses open/decode/imread and logs listings.",
                                 "phase1_test_results_used_for_selection": False,
                                 "phase1_statement": "Screw Phase 1 test metrics are historical context only and play no part in any "
                                                     "candidate, policy, threshold or selection decision."},
        "scope_of_this_step": "study only: NO selection lock, NO reproducibility rerun, NO final test in this step",
        "framework_versions": {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND.parent, capture_output=True, text=True).stdout.strip(),
                               "run_model_family_study.py": sha((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()),
                               "model_family_study.py": mod(mfs), "model_family_pipeline.py": mod(mfp), "patch_anomaly.py": mod(pa),
                               "synthetic_defects.py": mod(sd), "category_phase2.py(policy_threshold)": mod(cp2),
                               "wrapper run_screw_model_family.py": sha((SCRATCH / "run_screw_model_family.py").read_bytes())},
        "immutable_prior_artifacts": {"screw/phase1_declaration.json": "9a165905174a499204736cf23410c10905cf569e3d187d653662431cd335f478",
                                      "screw/phase1_baseline/autoencoder.pt": "4a18e28a2d5e7ba4a26a2bf59d3b3095b379e007d7524629ed9a85013c6543f2",
                                      "screw/phase1_baseline/threshold_lock.json": "a86ad9bddee6179d4604daaa4bcc584c53be88139b3750dc31b7e86faa5422f3",
                                      "screw/phase1_baseline/reports/phase1_report.json": "0138aafc471ad5d7d93e76a9399e83b928cb4e8fad3d829dc3874f83ac53645d"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(decl, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sha256": file_hashes(OUT)["sha256"],
                      "dataset_fingerprint": decl["development_data"]["dataset_fingerprint_sha256"],
                      "file_list": decl["development_data"]["normal_file_list_sha256"],
                      "expected_split_fp": decl["held_out_split_definition"]["expected_split_fingerprint_sha256"],
                      "split_audit": decl["held_out_split_definition"]["split_audit"],
                      "registry": decl["candidate_registry"]["registry_fingerprint"],
                      "policy_registry": decl["policy_registry"]["policy_registry_sha256"],
                      "rule": decl["selection_rule"]["declared_rule_sha256"],
                      "synthetic_expected": decl["synthetic_diagnostic_protocol"]["expected_images"],
                      "versions": decl["framework_versions"]}, indent=1))


if __name__ == "__main__":
    main()
