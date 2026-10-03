import hashlib, json, sys, time
from pathlib import Path
B = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
out = B / "ai_models" / "toothbrush" / "phase1_declaration_addendum.json"
if out.exists() or (B / "ai_models" / "toothbrush" / "phase1_baseline").exists():
    sys.exit("STOP: addendum or phase1 outputs already exist")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
decl = B / "ai_models" / "toothbrush" / "phase1_declaration.json"
add = {"addendum_to": {"path": str(decl), "sha256": h(decl)},
       "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "written_before_any_toothbrush_image_decode": True,
       "reason": "fields intended for the declaration that were not included when it was written; the declaration itself is left unmodified",
       "dataset_path_note": "instruction mentioned dataset/mvtec_anomaly_detection/toothbrush/, which does not exist; the framework's established layout dataset/<category>/ (used for every prior category) is dataset/toothbrush/",
       "png_colour_type_2": "train/good are 1024x1024, 8-bit, colour type 2 (RGB, 3 channels)",
       "model_implementation_sha256": {f: h(B / f) for f in ("app/ai/training/model.py", "app/ai/training/phase3_train.py", "app/ai/preprocessing/pipeline.py",
                                                            "app/ai/evaluation/evaluate.py", "app/ai/evaluation/threshold.py")},
       "runner": {"path": str(Path(sys.argv[1]) / "run_toothbrush_phase1.py"), "sha256": h(Path(sys.argv[1]) / "run_toothbrush_phase1.py")},
       "runner_additions_vs_screw": ["hard STOP before the test if any pre-test reproducibility check fails",
                                     "one-shot final_test_sentinel.json (exclusive create) with a unique execution id before test access",
                                     "final_test_consumed.json after the report", "model_config + model_config_sha256 in the lock",
                                     "validation P97/P98", "precision_defined flag", "lock-unchanged and scored-once audit entries"],
       "known_limitation": "60 train/good images -> 48 training / 12 validation; the threshold is calibrated on only 12 images"}
out.write_text(json.dumps(add, indent=1), encoding="utf-8")
print(out, h(out)); print(json.dumps(add["model_implementation_sha256"], indent=1))
