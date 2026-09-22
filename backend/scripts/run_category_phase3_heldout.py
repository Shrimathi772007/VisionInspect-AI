"""Leather-style genuinely held-out ConvAutoencoder baseline (Phase 3) for one category.

  train   verify dataset counts, split the train/good images 196/49 (sorted names, seed 42), train ONLY on the 196,
          score the 49 validation images, apply the pre-declared threshold-selection rule (see
          app/ai/evaluation/category_phase3_heldout.py), write the model and threshold_lock.json, and the
          validation-stage report. Cannot reach the final test set.
  repro   repeat the complete train -> validate -> select -> lock procedure into a scratch directory and compare it
          with run 1. It does NOT score the final test, so the protected test set stays exactly-once.
  final   (only after repro passed) verify the lock, then score the protected test set exactly once with the locked
          threshold and write reports/final_test_result.json and reports/phase3_report.json.

Never overwrites a Phase 1 / Phase 2 artifact, registers nothing for serving.

Usage (from backend/, venv active):
    python scripts/run_category_phase3_heldout.py train --category leather --train-good 245 --test-good 32 --test-defective 92
    python scripts/run_category_phase3_heldout.py repro --category leather --train-good 245 --test-good 32 --test-defective 92
    python scripts/run_category_phase3_heldout.py final --category leather
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.evaluation.category_phase3_heldout import (  # noqa: E402
    NO_CANDIDATE_MESSAGE,
    compare_stages,
    lock_path,
    model_path,
    reports_dir,
    train_validate_lock,
    validation_report_path,
)
from app.ai.evaluation.category_phase3_heldout_final import evaluate_locked_on_final_test, result_path  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _expected(args) -> dict:
    return {"train_good": args.train_good, "test_good": args.test_good, "test_defective": args.test_defective,
            "test_total": args.test_good + args.test_defective}


def cmd_train(args) -> None:
    category = args.category
    if validation_report_path(category).exists():
        sys.exit(f"Refusing to overwrite {validation_report_path(category)}")
    stage = train_validate_lock(category, model_path(category), _expected(args), lock_path(category))
    stage["git_head"] = _git("rev-parse", "HEAD")
    reports_dir(category).mkdir(parents=True, exist_ok=True)
    validation_report_path(category).write_text(json.dumps(stage, indent=1, sort_keys=True), encoding="utf-8")
    hide = ("validation_errors", "training_errors", "training_files", "validation_files")
    print(json.dumps({k: v for k, v in stage.items() if k not in hide}, indent=1, sort_keys=True))
    if stage["selection"]["winner"] is None:
        sys.exit(NO_CANDIDATE_MESSAGE + " Nothing locked; the final test will NOT be run.")
    print(f"LOCKED {stage['lock']['threshold_method']} threshold {stage['lock']['threshold']!r}")


def cmd_repro(args) -> None:
    category = args.category
    out = reports_dir(category) / "reproducibility.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    first = json.loads(validation_report_path(category).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        second = train_validate_lock(category, Path(tmp) / "autoencoder.pt", _expected(args), None)
    second = json.loads(json.dumps(second, sort_keys=True))
    repro = compare_stages(first, second)
    repro["second_run_model_sha256"] = second["artifact"]["sha256"]
    repro["second_run_training_seconds"] = second["training"]["duration_seconds"]
    repro["final_test_scored_in_second_run"] = False
    print(json.dumps(repro, indent=1))
    out.write_text(json.dumps(repro, indent=1), encoding="utf-8")
    if not all(v for k, v in repro.items() if isinstance(v, bool) and k != "final_test_scored_in_second_run"):
        sys.exit("!!! Run 2 did not reproduce run 1; investigate before the final test. !!!")


def cmd_final(args) -> None:
    category = args.category
    repro_path = reports_dir(category) / "reproducibility.json"
    if not repro_path.is_file():
        sys.exit("Run `repro` first.")
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if not all(v for k, v in repro.items() if isinstance(v, bool) and k != "final_test_scored_in_second_run"):
        sys.exit("Reproducibility did not pass; not running the final test.")
    result = evaluate_locked_on_final_test(category)
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "comparison")}, indent=1))
    print(json.dumps(result["comparison"]["phase3"], indent=1))
    stage = json.loads(validation_report_path(category).read_text(encoding="utf-8"))
    report = {"git_head": _git("rev-parse", "HEAD"), "validation_stage": stage, "reproducibility": repro, "final_test": result,
              "artifact_hashes": {"model": file_hashes(model_path(category)), "threshold_lock": file_hashes(lock_path(category))}}
    out = reports_dir(category) / "phase3_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Wrote {result_path(category)}\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "repro", "final"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--train-good", type=int)
    parser.add_argument("--test-good", type=int)
    parser.add_argument("--test-defective", type=int)
    args = parser.parse_args()
    if args.command in ("train", "repro") and None in (args.train_good, args.test_good, args.test_defective):
        parser.error("train/repro need --train-good --test-good --test-defective")
    {"train": cmd_train, "repro": cmd_repro, "final": cmd_final}[args.command](args)


if __name__ == "__main__":
    main()
