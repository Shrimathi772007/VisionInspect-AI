import os

old = "scripts/run_leather_model_family.py"
new = "scripts/run_model_family_study.py"
s = open(old, encoding="utf-8").read()

s = s.replace('"""Controlled alternative-model-family study for Leather (8 candidates: frozen ResNet-18 + Gaussian / memory bank).',
              '"""Controlled alternative-model-family study for one MVTec category (8 candidates: frozen ResNet-18 + Gaussian / memory bank).')
s = s.replace("python scripts/run_leather_model_family.py study|lock|repro|final --category leather",
              "python scripts/run_model_family_study.py study|lock|repro|final --category leather|metal_nut")
s = s.replace('EXPECTED = {"train_good": 245, "test_good": 32, "test_defective": 92, "test_total": 124}\n',
              'EXPECTED_BY_CATEGORY = {\n'
              '    "leather": {"train_good": 245, "test_good": 32, "test_defective": 92, "test_total": 124},\n'
              '    "metal_nut": {"train_good": 220, "test_good": 22, "test_defective": 93, "test_total": 115},\n'
              '}\n'
              'BACKBONE_SIZE_BYTES = 46_830_571\n')

# study / repro use the category's expected counts and stop on a backbone size mismatch
s = s.replace("    t0 = time.perf_counter()\n    inputs = load_inputs(category, EXPECTED)\n",
              "    t0 = time.perf_counter()\n    _check_backbone()\n    inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])\n")
s = s.replace("        inputs = load_inputs(category, EXPECTED)\n",
              "        _check_backbone()\n        inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])\n")
s = s.replace("def _print_table(study: dict) -> None:",
              "def _check_backbone() -> None:\n"
              "    info = backbone_info()  # raises on a SHA-256 mismatch\n"
              "    if info[\"size_bytes\"] != BACKBONE_SIZE_BYTES:\n"
              "        sys.exit(f\"STOP: backbone size {info['size_bytes']} != {BACKBONE_SIZE_BYTES}\")\n\n\n"
              "def _print_table(study: dict) -> None:")
s = s.replace("from app.ai.evaluation.model_family_pipeline import (  # noqa: E402\n    candidates_dir,",
              "from app.ai.evaluation.model_family_pipeline import (  # noqa: E402\n    backbone_info,\n    candidates_dir,")
assert "EXPECTED)" not in s
open(new, "w", encoding="utf-8").write(s)
os.remove(old)
print("ok", "EXPECTED_BY_CATEGORY" in s, "_check_backbone" in s)
