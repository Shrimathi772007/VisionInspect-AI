"""Controlled alternative-model-family study for one MVTec category (8 candidates: frozen ResNet-18 + Gaussian / memory bank).

  study   verify dataset counts (directory-entry counts only), fit and evaluate every registered candidate on the
          normal-only splits and the synthetic diagnostic, save all 8 candidate artifacts, apply the pre-declared
          selection rule, and write reports/model_family_study.json. Cannot reach the final test set.
  lock    apply the selection rule again from the saved study and write selection_lock.json + selected_candidate/.
          If no configuration passes the gate, nothing is locked and the final test is not run.
  repro   repeat the COMPLETE study (features, splits, fitting, selection) into a scratch directory and require it to
          reproduce the first run; does not score the final test.
  final   verify the lock and every hash, then score the protected test set exactly once with the locked candidate.

Never touches Phase 1/2/3 artifacts; registers nothing for serving.

Usage (from backend/, venv active):
    python scripts/run_model_family_study.py study|lock|repro|final --category leather|metal_nut|pill
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation import model_family_study as mfs  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.evaluation.model_family_final import evaluate_locked_on_final_test, result_path  # noqa: E402
from app.ai.evaluation.model_family_pipeline import (  # noqa: E402
    backbone_info,
    candidates_dir,
    compare_studies,
    create_selection_lock,
    load_inputs,
    lock_path,
    reports_dir,
    run_study,
    selected_dir,
    study_report_path,
)

EXPECTED_BY_CATEGORY = {
    "leather": {"train_good": 245, "test_good": 32, "test_defective": 92, "test_total": 124},
    "metal_nut": {"train_good": 220, "test_good": 22, "test_defective": 93, "test_total": 115},
    # Pill: the enumerated test folders are part of the expectation - the run stops on a missing or unexpected defect folder
    "pill": {"train_good": 267, "test_good": 26, "test_defective": 141, "test_total": 167,
             "test_by_type": {"color": 25, "combined": 17, "contamination": 21, "crack": 26, "faulty_imprint": 19,
                              "good": 26, "pill_type": 9, "scratch": 24}},
}
BACKBONE_SIZE_BYTES = 46_830_571


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _check_backbone() -> None:
    info = backbone_info()  # raises on a SHA-256 mismatch
    if info["size_bytes"] != BACKBONE_SIZE_BYTES:
        sys.exit(f"STOP: backbone size {info['size_bytes']} != {BACKBONE_SIZE_BYTES}")


def _print_table(study: dict) -> None:
    print(f"{'candidate':14s} {'policy':13s} {'thr(full pool)':>14s} {'meanFPR':>8s} {'worstFPR':>8s} {'thrCV':>7s} {'synRec':>7s} gate")
    for r in study["selection"]["rows"]:
        print(f"{r['candidate']:14s} {r['policy']:13s} {r['full_pool_threshold']:14.5g} {r['mean_fpr']:8.2%} {r['worst_fold_fpr']:8.2%} "
              f"{r['threshold_cv']:7.4f} {r['synthetic_recall']:7.3f} {r['passes_gate']}")
    print("synthetic AUROC (diagnostic):", {c: round(r["summary"]["synthetic_auroc"], 3) for c, r in study["candidates"].items()})
    print("SELECTION:", study["selection"]["winner"], "|", study["selection"]["reasoning"])


def cmd_study(category: str) -> None:
    if study_report_path(category).exists() or candidates_dir(category).exists():
        sys.exit("STOP: study outputs already exist; refusing to overwrite.")
    t0 = time.perf_counter()
    _check_backbone()
    inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])
    print(f"inputs ready in {inputs.build_seconds:.0f}s; counts {json.dumps(inputs.counts)}", flush=True)
    study = run_study(inputs, candidates_dir(category), log=lambda m: print(m, flush=True))
    study["git_head"] = _git("rev-parse", "HEAD")
    reports_dir(category).mkdir(parents=True, exist_ok=True)
    study_report_path(category).write_text(json.dumps(study, indent=1, sort_keys=True), encoding="utf-8")
    print(f"study finished in {time.perf_counter() - t0:.0f}s")
    _print_table(study)


def cmd_lock(category: str) -> None:
    study = json.loads(study_report_path(category).read_text(encoding="utf-8"))
    recomputed = mfs.select_configuration({c: r["summary"] for c, r in study["candidates"].items()}, mfs.registry())
    if recomputed["winner"] != study["selection"]["winner"]:
        sys.exit("STOP: the selection recomputed from the saved study differs from the recorded selection.")
    outcome = create_selection_lock(category, study)
    if not outcome["locked"]:
        (reports_dir(category) / "no_candidate.json").write_text(json.dumps(outcome, indent=1), encoding="utf-8")
        sys.exit(outcome["message"] + " Nothing locked; the final test will NOT be run.")
    lock = outcome["lock"]
    print(f"LOCKED {lock['selected_candidate']} | {lock['selected_policy']} threshold {lock['selected_threshold']!r}\n"
          f"artifact sha256 {lock['selected_artifact']['sha256']}\nlock digest {lock['lock_digest']}\n{outcome['lock_file']}")


def cmd_repro(category: str) -> None:
    out = reports_dir(category) / "reproducibility.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    first = json.loads(study_report_path(category).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        _check_backbone()
        inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])
        second = json.loads(json.dumps(run_study(inputs, Path(tmp) / "candidates", log=lambda m: print(m, flush=True)), sort_keys=True))
    repro = compare_studies(first, second)
    repro["second_run_seconds"] = second["study_seconds"]
    repro["final_test_scored_in_second_run"] = False
    print(json.dumps(repro, indent=1))
    out.write_text(json.dumps(repro, indent=1), encoding="utf-8")
    if not all(v for k, v in repro.items() if isinstance(v, bool) and k != "final_test_scored_in_second_run"):
        sys.exit("!!! Run 2 did not reproduce run 1; STOP and investigate. !!!")


def cmd_final(category: str) -> None:
    repro_path = reports_dir(category) / "reproducibility.json"
    if not repro_path.is_file():
        sys.exit("Run `repro` first.")
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if not all(v for k, v in repro.items() if isinstance(v, bool) and k != "final_test_scored_in_second_run"):
        sys.exit("Reproducibility did not pass; not running the final test.")
    result = evaluate_locked_on_final_test(category)
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "comparison")}, indent=1))
    print(json.dumps(result["comparison"], indent=1))
    report = {"git_head": _git("rev-parse", "HEAD"), "study": json.loads(study_report_path(category).read_text(encoding="utf-8")),
              "lock": json.loads(lock_path(category).read_text(encoding="utf-8")), "reproducibility": repro, "final_test": result,
              "artifact_hashes": {"selection_lock": file_hashes(lock_path(category)), "selected_state": file_hashes(selected_dir(category) / "model_state.pt")}}
    (reports_dir(category) / "model_family_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Wrote {result_path(category)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["study", "lock", "repro", "final"])
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    {"study": cmd_study, "lock": cmd_lock, "repro": cmd_repro, "final": cmd_final}[args.command](args.category)


if __name__ == "__main__":
    main()
