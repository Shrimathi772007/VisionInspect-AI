"""Runs the UNCHANGED `repro` step through run_zipper_model_family.py (importing it installs all its test guards), with
one runtime-only addition: compare_studies is wrapped so the independently regenerated second study is saved to
this scratch folder (second_study.json) for the first-run/second-run value table. The comparison itself is the
framework's compare_studies, called unchanged.
"""

import json
import sys
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))

import run_zipper_model_family as wrapper  # noqa: E402  (installs guards)

_real_compare = wrapper.runner.compare_studies


def _capturing_compare(first, second):
    (SCRATCH / "second_study.json").write_text(json.dumps(second, indent=1, sort_keys=True), encoding="utf-8")
    return _real_compare(first, second)


wrapper.runner.compare_studies = _capturing_compare
sys.argv = [sys.argv[0], "repro"]
wrapper.main()
