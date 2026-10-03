"""Runs ONLY the unchanged `study` step through run_transistor_model_family.py (importing it installs all its test
guards) and records an execution record (execution id, start/end timestamps, environment) next to the study report.
No lock, no repro, no final test.
"""

import json
import platform
import sys
import time
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))

import run_transistor_model_family as wrapper  # noqa: E402  (installs guards)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402

EXECUTION_ID = sys.argv[1]
out = mfp.reports_dir("transistor") / "study_execution_record.json"
if out.exists():
    sys.exit("STOP: execution record exists; refusing to overwrite.")
started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
sys.argv = [sys.argv[0], "study"]
wrapper.main()
ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
decodes = json.loads((SCRATCH / "decodes_study.json").read_text(encoding="utf-8"))
record = {"study_id": "transistor-model-family-2026-09-28", "execution_id": EXECUTION_ID, "step": "study only",
          "started_utc": started, "ended_utc": ended,
          "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                          "device": "cpu", "torch_threads": torch.get_num_threads(), "platform": platform.platform()},
          "seeds": {"splits": "holdout 42-51, random k-fold 42-44, contiguous (none)", "coreset": 42, "synthetic": "20240607 + source index"},
          "study_report_sha256": file_hashes(mfp.study_report_path("transistor"))["sha256"],
          "declaration_sha256": file_hashes(mfp.study_root("transistor") / "study_declaration.json")["sha256"],
          "wrapper_sha256": file_hashes(SCRATCH / "run_transistor_model_family.py")["sha256"],
          "launcher_sha256": file_hashes(Path(__file__))["sha256"],
          "test_access": decodes,
          "lock_created": mfp.lock_path("transistor").exists(), "selected_candidate_dir_created": mfp.selected_dir("transistor").exists()}
out.write_text(json.dumps(record, indent=1), encoding="utf-8")
print(json.dumps(record, indent=1))
