"""Write backend/ai_models/zipper/model_family_study/study_declaration.json BEFORE the study decodes anything.

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

CATEGORY = "zipper"
STUDY_ID = "zipper-model-family-2026-09-29"
EXPECTED = {"train_good": 240, "test_good": 32, "test_defective": 119, "test_total": 151,
            "test_by_type": {"broken_teeth": 19, "combined": 16, "fabric_border": 17, "fabric_interior": 16, "good": 32,
                             "rough": 17, "split_teeth": 18, "squeezed_teeth": 16}}
SCRATCH = Path(__file__).resolve().parent
OUT = mfp.study_root(CATEGORY) / "study_declaration.json"
PHASE1_DIR = mfp.study_root(CATEGORY).parent / "phase1_baseline"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


EXECUTION_ID = sys.argv[1] if len(sys.argv) > 1 else "unset"


def _phase1_style_fp(samples) -> str:
    """The Phase 1 runner's train/good fingerprint (name + bytes SHA-256, sorted), recomputed for verification."""
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.path.name}:{sha(s.path.read_bytes())}\n".encode())
    return h.hexdigest()


def _canonical_vs_phase1(schemes, names) -> dict:
    """Overlap of the framework's canonical held-out files with the Phase 1 validation files (from the Phase 1 threshold
    lock, which lists train/good file names only - no test data)."""
    lock = json.loads((PHASE1_DIR / "threshold_lock.json").read_text(encoding="utf-8"))
    p1_val = set(lock["split"]["validation_files"])
    kind, name = mfs.CANONICAL_SPLIT
    canon = next(s for s in schemes[kind] if s["name"] == name)
    held = {names[i] for i in canon["held_out"]}
    return {"canonical_split": f"{kind}/{name}", "canonical_held_out": len(held), "phase1_validation": len(p1_val),
            "shared_images": len(held & p1_val), "shared_files": sorted(held & p1_val)}


def main() -> None:
    if OUT.exists() or mfp.study_report_path(CATEGORY).exists() or mfp.candidates_dir(CATEGORY).exists():
        sys.exit("STOP: zipper model-family outputs already exist; refusing to overwrite.")
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
        "development_data": {"definition": "dataset/zipper/train/good only (240 images, file-name sorted)",
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
                                          "label": "DIAGNOSTIC ONLY - not real Zipper defect recall"},
        "final_test_isolation": {"statement": "No file under dataset/zipper/test or dataset/zipper/ground_truth is opened, decoded or scored; "
                                              "only directory entries are counted. Guard refuses open/decode/imread and logs listings.",
                                 "phase1_test_results_used_for_selection": False,
                                 "phase1_statement": "Zipper Phase 1 test metrics (and the Phase 1 report/scores) are historical context only, are not read by this study, and play no part in any "
                                                     "candidate, policy, threshold or selection decision."},
        "scope_of_this_step": "study only: NO selection lock, NO reproducibility rerun, NO final test in this step",
        "framework_versions": {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND.parent, capture_output=True, text=True).stdout.strip(),
                               "run_model_family_study.py": sha((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()),
                               "model_family_study.py": mod(mfs), "model_family_pipeline.py": mod(mfp), "patch_anomaly.py": mod(pa),
                               "synthetic_defects.py": mod(sd), "category_phase2.py(policy_threshold)": mod(cp2),
                               "wrapper run_zipper_model_family.py": sha((SCRATCH / "run_zipper_model_family.py").read_bytes())},
        "immutable_prior_artifacts": {"zipper/phase1_declaration.json": "7a16ab4f68e3b5e1b0c6764e7aad7ae8fb994ef6e4c481003e3561054efc769f",
                                      "zipper/phase1_baseline/autoencoder.pt": "7fb1d74a5a56bfea956dfd45b0eb99efb0dd1cbc77256daceb6521cedf15da14",
                                      "zipper/phase1_baseline/threshold_lock.json": "66651557e20953c075c2af17c7ce6b9d65ade16f064fd516bfc2c18c776be1b1",
                                      "zipper/phase1_baseline/reports/phase1_report.json": "3ac3502bbfdf22f33245f8efe0077b16988d769f708c9b7c10d353bf571be035",
                                      "zipper/phase1_baseline/final_test_sentinel.json": "78f3826b5ad0878a4cb867edd445654ecf8d6813fa9045cda9997ca0dc6180fd",
                                      "zipper/phase1_baseline/final_test_consumed.json": "ff2e046919d61d1ca2c8ed4f1a3ad84d37c159215d0044d7e5298eb400107726"},
        "phase1_train_good_fingerprint_verified": {"expected": "b3db7ec6c22701c450cba49c9075d0b792414b66d094129bae6bf33c6094dee5",
                                                   "recomputed": _phase1_style_fp(samples)},
        "split_note": "The model-family framework does NOT reuse the Phase 1 192/48 split (sha 88aa6aa3...77de, verified reproducible "
                      "before this study); it uses its own established 30 normal-only splits, as for every prior category. With n=240 "
                      "each split holds out round(0.2*n)=48 (192 calibration); 5 divides 240, so every k-fold fold holds out 48. The overlap "
                      "between the canonical 80/20 seed-42 held-out set and the Phase 1 validation set is COMPUTED below (not assumed). "
                      "The synthetic diagnostic is on the canonical split only (48 x 15 = 720 images); the 30 splits measure "
                      "held-out-normal FPR. Per-fold recall and synthetic F1 are not framework metrics and are not computed.",
        "canonical_heldout_vs_phase1_validation": _canonical_vs_phase1(schemes, names),
        "planned_execution_id": EXECUTION_ID,
    }
    if decl["phase1_train_good_fingerprint_verified"]["recomputed"] != decl["phase1_train_good_fingerprint_verified"]["expected"]:
        sys.exit("STOP: train/good does not match the Zipper Phase 1 fingerprint")
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
