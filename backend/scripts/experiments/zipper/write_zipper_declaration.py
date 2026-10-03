"""Write the Zipper Phase 1 (ConvAE) declaration before any Zipper image is decoded. Refuses to overwrite.

Reads only train/good file bytes (SHA-256 and the PNG IHDR header for dimensions); no image is decoded and nothing
under dataset/zipper/test is opened (test folders are counted from directory entries only).
"""

import hashlib
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation.category_phase1 import file_hashes, phase1_config  # noqa: E402
from app.ai.training import artifacts  # noqa: E402

CATEGORY = "zipper"
STUDY_ID = "zipper-phase1-convae-2026-09-29"
ROOT = BACKEND.parent / "dataset" / CATEGORY
OUT = artifacts.ARTIFACTS_ROOT / CATEGORY / "phase1_declaration.json"
SCRATCH = Path(__file__).resolve().parent


def png_header(path: Path) -> tuple[int, int, int, int]:
    with open(path, "rb") as fh:
        head = fh.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"not a PNG: {path}")
    w, h, depth, colour = struct.unpack(">IIBB", head[16:26])
    return w, h, depth, colour


def main() -> None:
    if OUT.exists():
        sys.exit(f"STOP: {OUT} already exists; refusing to overwrite.")
    train = sorted(p for p in (ROOT / "train" / "good").iterdir() if p.is_file())
    headers = Counter(png_header(p) for p in train)
    listing = hashlib.sha256("".join(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in train).encode()).hexdigest()
    count = lambda d: sum(1 for e in os.scandir(d) if e.is_file())  # noqa: E731
    test_by_type = {e.name: count(e.path) for e in sorted(os.scandir(ROOT / "test"), key=lambda e: e.name) if e.is_dir()}
    masks = {e.name: count(e.path) for e in sorted(os.scandir(ROOT / "ground_truth"), key=lambda e: e.name) if e.is_dir()}
    cfg = phase1_config(CATEGORY)
    decl = {
        "study_id": STUDY_ID, "dataset": "MVTec AD", "category": CATEGORY, "phase": "phase1_convae",
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "category_selection_rule": "alphabetical order among eligible MVTec categories without completed artifacts "
                                   "(convention recorded in the grid declaration); grid, screw, toothbrush, transistor and wood closed, zipper next",
        "scope": "Phase 1 ConvAE baseline ONLY. No model-family study, no alternative models, no policy search, no serving.",
        "model": {"name": "ConvAutoencoder (app.ai.training.model.build_model), existing Phase 1 baseline, unchanged",
                  "image_size": list(cfg.image_size), "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                  "learning_rate": cfg.learning_rate, "seed": cfg.seed, "device": cfg.device, "optimizer": "Adam", "loss": "MSELoss",
                  "preprocessing": "cv2.imread(IMREAD_COLOR) -> BGR->RGB -> INTER_AREA resize 128x128 -> /255; no augmentation"},
        "training_data": "dataset/zipper/train/good (the training subset of the split only)",
        "validation_data": "split_train_validation(train/good, training_fraction=0.8, seed=42): deterministic good-only validation subset",
        "threshold_policy": "mean + 3 * population std of per-image reconstruction MSE on the validation-good subset",
        "test_data": "dataset/zipper/test/good + every dataset/zipper/test/<defect> folder; ground_truth masks never used",
        "final_test": "scored exactly once, after the model and threshold are locked and the pre-test reproducibility retrain passes",
        "no_post_hoc_tuning": True,
        "acceptance_gates_unchanged": {"EXCELLENT": "recall>=0.90, F1>=0.85, FPR<=0.10", "GOOD": "recall>=0.85, F1>=0.80, FPR<=0.10",
                                       "ACCEPTABLE": "recall>=0.75, F1>=0.70, FPR<=0.15", "otherwise": "NOT PRODUCTION READY"},
        "confidence_intervals": "Wilson 95% (accuracy, precision, recall, FPR, per-defect recall)",
        "dataset_metadata_at_declaration": {
            "train_good": len(train), "train_good_png_headers(w,h,bit_depth,colour_type)": {str(k): v for k, v in headers.items()},
            "train_good_name_and_bytes_sha256": listing, "test_by_type_from_directory_entries": test_by_type,
            "test_total": sum(test_by_type.values()), "ground_truth_masks_by_type": masks,
            "note": "test image files were not opened (no headers, no bytes) before the lock"},
        "isolation": "artifacts under backend/ai_models/zipper/ only; no other category's artifacts or test data are read",
        "runner_sha256": hashlib.sha256((SCRATCH / "run_zipper_phase1.py").read_bytes()).hexdigest(),
        "model_implementation_sha256": {f: hashlib.sha256((BACKEND / f).read_bytes()).hexdigest()
                                        for f in ("app/ai/training/model.py", "app/ai/training/phase3_train.py", "app/ai/preprocessing/pipeline.py",
                                                  "app/ai/evaluation/evaluate.py", "app/ai/evaluation/threshold.py",
                                                  "app/ai/training/validation_split.py", "app/ai/evaluation/category_phase1.py")},
        "runner_lineage": "Wood (= Transistor = Toothbrush) Phase 1 runner = Screw/Grid/Capsule/Cable Phase 1 runner + hard STOP before the test if any pre-test "
                          "reproducibility check fails, one-shot final_test_sentinel.json (exclusive create) with a unique execution id, "
                          "final_test_consumed.json, model_config_sha256 in the lock, validation P97/P98, precision_defined flag, "
                          "lock-unchanged and scored-once audit entries. Only CATEGORY/EXPECTED changed for zipper.",
        "phase1_test_protocol": "the established Phase 1 workflow scores the test set once with the locked ConvAE after the lock and the "
                                "pre-test reproducibility retrain; that single pass is a consumed baseline evaluation and is never rescored",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(decl, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sha256": file_hashes(OUT)["sha256"], **decl["dataset_metadata_at_declaration"],
                      "runner_sha256": decl["runner_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
