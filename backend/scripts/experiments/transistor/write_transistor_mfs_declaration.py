"""Write backend/ai_models/transistor/model_family_study/study_declaration.json BEFORE the study decodes anything.

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

CATEGORY = "transistor"
STUDY_ID = "transistor-model-family-2026-09-28"
EXPECTED = {"train_good": 213, "test_good": 60, "test_defective": 40, "test_total": 100,
            "test_by_type": {"bent_lead": 10, "cut_lead": 10, "damaged_case": 10, "good": 60, "misplaced": 10}}
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
        sys.exit("STOP: transistor model-family outputs already exist; refusing to overwrite.")
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
        "development_data": {"definition": "dataset/transistor/train/good only (213 images, file-name sorted)",
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
                                          "label": "DIAGNOSTIC ONLY - not real Transistor defect recall"},
        "final_test_isolation": {"statement": "No file under dataset/transistor/test or dataset/transistor/ground_truth is opened, decoded or scored; "
                                              "only directory entries are counted. Guard refuses open/decode/imread and logs listings.",
                                 "phase1_test_results_used_for_selection": False,
                                 "phase1_statement": "Transistor Phase 1 test metrics are historical context only and play no part in any "
                                                     "candidate, policy, threshold or selection decision."},
        "scope_of_this_step": "study only: NO selection lock, NO reproducibility rerun, NO final test in this step",
        "framework_versions": {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND.parent, capture_output=True, text=True).stdout.strip(),
                               "run_model_family_study.py": sha((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()),
                               "model_family_study.py": mod(mfs), "model_family_pipeline.py": mod(mfp), "patch_anomaly.py": mod(pa),
                               "synthetic_defects.py": mod(sd), "category_phase2.py(policy_threshold)": mod(cp2),
                               "wrapper run_transistor_model_family.py": sha((SCRATCH / "run_transistor_model_family.py").read_bytes())},
        "immutable_prior_artifacts": {"transistor/phase1_declaration.json": "8620835e99232ac131ea723cd742169473e4f945083c30564590e689b1a38a06",
                                      "transistor/phase1_baseline/autoencoder.pt": "b492b9154f84487bb58ea5aa2b304ca8c85368fbc39c960e66a5c794356c89f2",
                                      "transistor/phase1_baseline/threshold_lock.json": "50e00889239e10671d76cd3c6d655a8ac2239e5d44ae6ce7ce1f6c498202aed9",
                                      "transistor/phase1_baseline/reports/phase1_report.json": "a379cdf129824d19f02b49fcb908fdfd1e50fcbab31aa85a569f56442f348e24",
                                      "transistor/phase1_baseline/final_test_sentinel.json": "8558f114212f332a95c0a19130644410c97b4bcd6a67c372365ce50f02abd6c3",
                                      "transistor/phase1_baseline/final_test_consumed.json": "086643cb6b091f59dd0a2d3027345daaf713177894a16d53b8930395efdf115d"},
        "phase1_train_good_fingerprint_verified": {"expected": "d0adb8dc65107214fae9f3d814b63656424885398d9f6d6466b44078eb51df5a",
                                                   "recomputed": _phase1_style_fp(samples)},
        "split_note": "The model-family framework does NOT reuse the Phase 1 170/43 split (sha 6320b482...1196, verified reproducible "
                      "before this study); it uses its own established 30 normal-only splits, as for every prior category. With n=213 "
                      "each split holds out round(0.2*n)=43 (170 calibration); the k-fold folds hold out 43 or 42. The canonical 80/20 "
                      "seed-42 held-out set shares 0 images with the Phase 1 validation set (the framework takes the FIRST 43 of the "
                      "seeded permutation; the Phase 1 split takes its validation from the other end) - an unchanged property of the "
                      "framework. The synthetic diagnostic is on the canonical split only (43 x 15 = 645 images); the 30 splits measure "
                      "held-out-normal FPR. Per-fold recall and synthetic F1 are not framework metrics and are not computed.",
        "planned_execution_id": EXECUTION_ID,
    }
    if decl["phase1_train_good_fingerprint_verified"]["recomputed"] != decl["phase1_train_good_fingerprint_verified"]["expected"]:
        sys.exit("STOP: train/good does not match the Transistor Phase 1 fingerprint")
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
