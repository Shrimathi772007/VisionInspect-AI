"""Write the pre-execution study declaration for grid (before any grid image is decoded).

Records the registry, policies, splits, rule and acceptance gates exactly as the unchanged framework defines them
(read from the live modules, so the hashes are the ones that will actually be applied). Refuses to overwrite.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_study as mfs  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes, phase1_config  # noqa: E402
from app.ai.models.resnet18 import RESNET18_SHA256  # noqa: E402
from app.ai.training import artifacts  # noqa: E402

CATEGORY = "grid"
STUDY_ID = "grid-full-pipeline-2026-09-28"
OUT = artifacts.ARTIFACTS_ROOT / CATEGORY / "study_declaration.json"
SCRATCH = Path(__file__).resolve().parent


def main() -> None:
    if OUT.exists():
        sys.exit(f"STOP: {OUT} already exists; refusing to overwrite.")
    capsule = artifacts.ARTIFACTS_ROOT / "capsule"
    cfg = phase1_config(CATEGORY)
    n = 264
    schemes = mfs.scheme_splits(n)
    decl = {
        "study_id": STUDY_ID,
        "category": CATEGORY,
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "category_selection_rule": "NEW CONVENTION (not prescribed by the repository, which has no next-category roadmap): "
                                   "alphabetical order among eligible MVTec categories that have data but no completed artifacts "
                                   "(grid, screw, toothbrush, transistor, wood, zipper) -> grid.",
        "post_capsule_statement": "This study was initiated AFTER the Capsule final-test result was known. Capsule's result was used "
                                  "only as historical context for deciding to continue to another category; it plays no part in any "
                                  "grid model, threshold or selection decision.",
        "capsule_immutable": {"statement": "Capsule artifacts are immutable and are not read, rerun or modified by this study.",
                              "hashes_at_declaration": {str(p.relative_to(capsule)).replace("\\", "/"): file_hashes(p)["sha256"]
                                                        for p in sorted(capsule.rglob("*")) if p.is_file()}},
        "dataset": {"root": "dataset/grid", "expected_counts_from_directory_metadata": {
            "train_good": 264, "test_good": 21, "test_defective": 57, "test_total": 78,
            "test_by_type": {"bent": 12, "broken": 12, "glue": 11, "good": 21, "metal_contamination": 11, "thread": 11}},
            "selection_data": "train/good only; ground_truth masks never used"},
        "pipeline": ["Phase 1 ConvAE baseline (established Cable/Capsule Phase 1 runner): 80/20 split of train/good seed 42, "
                     "threshold = validation mean + 3*population std, lock, reproducibility retrain without test access, "
                     "then ONE final-test scoring of the locked ConvAE",
                     "Model-family study (unchanged backend/scripts/run_model_family_study.py): study -> lock -> repro; "
                     "test images guarded (refuse open/decode), never reads Phase 1 metrics (only hashes Phase 1 files)",
                     "ONE final-test scoring of the locked model-family winner (framework cmd_final behind the Cable/Capsule "
                     "final wrapper: lock constants verified, one-shot sentinel)"],
        "final_test_protocol": {"executions": {"phase1_convae": 1, "model_family_selected": 1},
                                "approved_by_user": "established pipeline: one test scoring per pre-locked model (2 in total); "
                                                    "no rerun of either; no result used to change any model or threshold",
                                "confidence_intervals": "Wilson 95% (accuracy, precision, recall, FPR, per-defect recall)"},
        "phase1_convae": {"model": "ConvAutoencoder (app.ai.training.model), 3-16-32-64-128 stride-2, Sigmoid",
                          "image_size": list(cfg.image_size), "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                          "learning_rate": cfg.learning_rate, "seed": cfg.seed, "device": cfg.device, "optimizer": "Adam",
                          "loss": "MSELoss", "split": "split_train_validation(fraction 0.8, seed 42)",
                          "threshold": "mean + 3*std (population) of validation reconstruction MSE"},
        "model_family": {
            "candidates": [s.config() for s in mfs.registry()], "candidate_count": mfs.EXPECTED_CANDIDATES,
            "registry_fingerprint": mfs.registry_fingerprint(),
            "policies": mfs.policy_ids(), "policy_count": mfs.EXPECTED_POLICIES,
            "splits": {"holdout_80_20_seeds": list(mfs.HOLDOUT_SEEDS), "random_kfold5_seeds": list(mfs.RANDOM_KFOLD_SEEDS),
                       "contiguous_kfold5": 1, "total": sum(len(v) for v in schemes.values()),
                       "held_out_per_split": int(round(n * 0.2)), "split_fingerprint_expected": mfs.splits_fingerprint(schemes)},
            "synthetic_diagnostic": f"canonical split {'/'.join(mfs.CANONICAL_SPLIT)}: {int(round(n * 0.2))} held-out images x 15 = "
                                    f"{int(round(n * 0.2)) * 15} synthetic images (DIAGNOSTIC ONLY, never real recall)",
            "selection_rule": {"gate_worst_fold_fpr_max": mfs.GATE_FPR, "preferred_synthetic_recall_floor": mfs.SYNTHETIC_FLOOR,
                               "recall_tie": mfs.RECALL_TIE, "cv_tie": mfs.CV_TIE,
                               "order": ["gate", "preferred floor", "lowest worst-fold FPR", "synthetic recall (tie 0.03)",
                                         "threshold CoV (tie 0.01)", "simpler model", "candidate id, policy id"],
                               "declared_rule_sha256": hashlib.sha256(mfs.__doc__.encode("utf-8")).hexdigest()},
            "locked_threshold_rule": "winning policy applied to leave-one-image-out scores of all normal images under the model fitted on all",
            "backbone_sha256": RESNET18_SHA256},
        "acceptance_gates_unchanged": {"EXCELLENT": "recall>=0.90, F1>=0.85, FPR<=0.10", "GOOD": "recall>=0.85, F1>=0.80, FPR<=0.10",
                                       "ACCEPTABLE": "recall>=0.75, F1>=0.70, FPR<=0.15",
                                       "otherwise": "NOT PRODUCTION READY / BELOW ACCEPTABLE",
                                       "note": "point estimates are read together with Wilson intervals"},
        "reproducibility": ["Phase 1: identical retrain (dataset fingerprint, split, model SHA/MD5, loss history, validation errors, threshold) without test access",
                            "Model-family: framework compare_studies over a complete second study in a temp dir (all checks must be true)"],
        "runners": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SCRATCH.glob("run_grid_*.py"))}
                   | {"backend/scripts/run_model_family_study.py": hashlib.sha256((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()).hexdigest()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(decl, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sha256": file_hashes(OUT)["sha256"], "rule_sha256": decl["model_family"]["selection_rule"]["declared_rule_sha256"],
                      "registry": decl["model_family"]["registry_fingerprint"], "splits": decl["model_family"]["splits"],
                      "runners": decl["runners"]}, indent=1))


if __name__ == "__main__":
    main()
