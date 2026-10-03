"""Write backend/ai_models/toothbrush/model_family_study/study_declaration.json BEFORE the study decodes anything.

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

CATEGORY = "toothbrush"
STUDY_ID = "toothbrush-model-family-2026-09-28"
EXPECTED = {"train_good": 60, "test_good": 12, "test_defective": 30, "test_total": 42,
            "test_by_type": {"defective": 30, "good": 12}}
SCRATCH = Path(__file__).resolve().parent
OUT = mfp.study_root(CATEGORY) / "study_declaration.json"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


EXECUTION_ID = sys.argv[1] if len(sys.argv) > 1 else "unset"


def _phase1_style_fp(samples) -> str:
    """The Phase 1 runner's train/good fingerprint (name + bytes SHA-256, sorted), recomputed for verification."""
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.path.name}:{sha(s.path.read_bytes())}\n".encode())
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or mfp.study_report_path(CATEGORY).exists() or mfp.candidates_dir(CATEGORY).exists():
        sys.exit("STOP: toothbrush model-family outputs already exist; refusing to overwrite.")
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
        "development_data": {"definition": "dataset/toothbrush/train/good only (60 images, file-name sorted)",
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
                                          "label": "DIAGNOSTIC ONLY - not real Toothbrush defect recall"},
        "final_test_isolation": {"statement": "No file under dataset/toothbrush/test or dataset/toothbrush/ground_truth is opened, decoded or scored; "
                                              "only directory entries are counted. Guard refuses open/decode/imread and logs listings.",
                                 "phase1_test_results_used_for_selection": False,
                                 "phase1_statement": "Toothbrush Phase 1 test metrics are historical context only and play no part in any "
                                                     "candidate, policy, threshold or selection decision."},
        "scope_of_this_step": "study only: NO selection lock, NO reproducibility rerun, NO final test in this step",
        "framework_versions": {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND.parent, capture_output=True, text=True).stdout.strip(),
                               "run_model_family_study.py": sha((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()),
                               "model_family_study.py": mod(mfs), "model_family_pipeline.py": mod(mfp), "patch_anomaly.py": mod(pa),
                               "synthetic_defects.py": mod(sd), "category_phase2.py(policy_threshold)": mod(cp2),
                               "wrapper run_toothbrush_model_family.py": sha((SCRATCH / "run_toothbrush_model_family.py").read_bytes())},
        "immutable_prior_artifacts": {"toothbrush/phase1_declaration.json": "4d187bdc59cb1566d5a88a8ddfa07e47089cd2302cd0e2f3eea7b1e28ee8ae2a",
                                      "toothbrush/phase1_declaration_addendum.json": "f49e1e990e2d7613b85c0fd8e00a179dc66ec48118436a57dd4f93b86857dcaa",
                                      "toothbrush/phase1_baseline/autoencoder.pt": "cf4e8fc24b0856807c3535aa7e999253b309fa206b033420b9df3187a923a055",
                                      "toothbrush/phase1_baseline/threshold_lock.json": "9d633284e5ea874683084eea9ff4221ba40ac35b7c6b72299046e776e1fd4dc6",
                                      "toothbrush/phase1_baseline/reports/phase1_report.json": "8a4f88d5241ea06984dfb30ec2ca0b510a9b0a38f60367e9a741296647f7aa3a",
                                      "toothbrush/phase1_baseline/final_test_sentinel.json": "af9087c40ed14fbb235947d5ffb6bf03ca9939acdd5a61fec132baa8fd86571c",
                                      "toothbrush/phase1_baseline/final_test_consumed.json": "7eb717967f86241ec5eaea16ee76e7342a9faf4ee8bbc903d82c838858bf03a9"},
        "phase1_train_good_fingerprint_verified": {"expected": "f7e6f3c526d722aff2496abcf7757739d72e79518ec647d6f95dbe7b682e9413",
                                                   "recomputed": _phase1_style_fp(samples)},
        "split_size_note": "The instruction quoted 256 calibration / 64 held-out (Screw's n=320). The unchanged framework holds out "
                           "round(0.2*n) = 12 of Toothbrush's n=60 images (48 calibration / 12 held-out) in every split; the split logic "
                           "is NOT modified. Per-split FPR granularity is therefore 1/12 = 8.33%, so the 10% gate allows at most one "
                           "held-out false positive in the worst split.",
        "planned_execution_id": EXECUTION_ID,
    }
    if decl["phase1_train_good_fingerprint_verified"]["recomputed"] != decl["phase1_train_good_fingerprint_verified"]["expected"]:
        sys.exit("STOP: train/good does not match the Phase 1 fingerprint")
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
