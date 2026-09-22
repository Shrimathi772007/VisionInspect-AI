"""Phase 2 threshold robustness / selection study for one category, on the frozen Phase 1 ConvAutoencoder.

  study   verify the frozen model's hash, score train/good ONLY, run every normal-only calibration scheme and threshold
          policy, apply the pre-declared selection rule (see app/ai/evaluation/category_phase2.py), and write
          reports/calibration_study.json. Cannot reach the final test set.
  lock    repeat the ENTIRE calibration procedure (re-scoring included) and require it to reproduce the first run
          exactly, then write threshold_lock.json for the single selected threshold. If no candidate satisfies the
          calibration FPR requirement, nothing is locked and the final test is not run.
  final   score the protected test set exactly once with the locked threshold, compare with Phase 1, and write
          reports/final_test_result.json and reports/phase2_report.json.

Never retrains, never overwrites a Phase 1 artifact, registers nothing for serving.

Usage (from backend/, venv active):
    python scripts/run_category_phase2.py study --category leather --model-sha256 <sha>
    python scripts/run_category_phase2.py lock  --category leather --model-sha256 <sha>
    python scripts/run_category_phase2.py final --category leather --model-sha256 <sha>
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.evaluation.category_phase2 import (  # noqa: E402
    NO_CANDIDATE_MESSAGE,
    create_lock,
    lock_path,
    policy_ids,
    reports_dir,
    study_from_model,
)
from app.ai.evaluation.category_phase2_final import (  # noqa: E402
    evaluate_locked_on_final_test,
    phase1_paths,
    phase1_reference,
    result_path,
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _print(study: dict) -> None:
    print(f"pool: n={study['n']} mean {study['pool_score_distribution']['mean']:.6g} std {study['pool_score_distribution']['std']:.6g} "
          f"max {study['pool_score_distribution']['max']:.6g}   splits sha256 {study['splits_sha256'][:16]}")
    print(f"{'policy':16s} {'full-pool thr':>13s} {'pooled cv':>9s} {'thr min':>10s} {'thr max':>10s} {'mean FPR':>9s} {'worst FPR':>9s}  eligible")
    elig = set(study["selection"]["eligible"])
    for p in policy_ids():
        r = study["policies"][p]
        po = r["pooled"]
        print(f"{p:16s} {r['full_pool_threshold']:13.6g} {po['cv']:9.4f} {po['min']:10.6g} {po['max']:10.6g} "
              f"{po['mean_held_out_fpr']:9.2%} {po['worst_fold_fpr']:9.2%}  {p in elig}")
    print("SELECTION:", study["selection"]["winner"], "|", study["selection"]["reasoning"])


def cmd_study(category: str, sha: str) -> None:
    out = reports_dir(category) / "calibration_study.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    ref = phase1_reference(category)
    t0 = time.perf_counter()
    study = study_from_model(category, phase1_paths(category)["model"], sha)
    study["phase1_reference_at_study_time"] = ref
    study["git_head"] = _git("rev-parse", "HEAD")
    print(f"study done in {time.perf_counter() - t0:.1f}s")
    _print(study)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")


def cmd_lock(category: str, sha: str) -> None:
    study_path = reports_dir(category) / "calibration_study.json"
    if not study_path.is_file():
        sys.exit("Run `study` first.")
    if lock_path(category).exists():
        sys.exit(f"Refusing to overwrite {lock_path(category)}")
    first = json.loads(study_path.read_text(encoding="utf-8"))
    print("Repeating the complete calibration procedure (including re-scoring train/good) ...")
    second = study_from_model(category, phase1_paths(category)["model"], sha)
    outcome = create_lock(category, phase1_paths(category)["model"], sha, first, second, phase1_reference(category))
    print(json.dumps(outcome["reproducibility"], indent=1))
    if not outcome["locked"]:
        (reports_dir(category) / "no_candidate.json").write_text(json.dumps(outcome, indent=1), encoding="utf-8")
        sys.exit(NO_CANDIDATE_MESSAGE + " Nothing locked; the final test will NOT be run.")
    print(json.dumps(outcome["lock_reproducibility"], indent=1))
    print(f"LOCKED {outcome['lock']['threshold_method']} threshold {outcome['lock']['threshold']!r} -> {lock_path(category)}")


def cmd_final(category: str, sha: str) -> None:
    if not lock_path(category).is_file():
        sys.exit("No threshold lock: run `lock` first (or no candidate satisfied the requirement).")
    result = evaluate_locked_on_final_test(category, sha)
    hide = ("predictions",)
    print(json.dumps({k: v for k, v in result.items() if k not in hide}, indent=1))
    study = json.loads((reports_dir(category) / "calibration_study.json").read_text(encoding="utf-8"))
    repro = json.loads((reports_dir(category) / "lock_reproducibility.json").read_text(encoding="utf-8"))
    lock = json.loads(lock_path(category).read_text(encoding="utf-8"))
    report = {"git_head": _git("rev-parse", "HEAD"), "lock": lock, "calibration_study": study, "reproducibility": repro,
              "final_test": result, "artifact_hashes": {"threshold_lock": file_hashes(lock_path(category)),
                                                        "phase1_model": file_hashes(phase1_paths(category)["model"])}}
    out = reports_dir(category) / "phase2_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Wrote {result_path(category)}\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["study", "lock", "final"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()
    {"study": cmd_study, "lock": cmd_lock, "final": cmd_final}[args.command](args.category, args.model_sha256)


if __name__ == "__main__":
    main()
