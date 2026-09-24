"""Controlled Phase 4 optimization experiment for ONE category, against its protected Phase 3 baseline.

Order of operations (each step uses only what earlier steps allow):

  1. Verify the protected Phase 3 artifact (hash from its own report) and refuse to run if a
     phase4_optimization directory already exists (never overwrites).
  2. Experiment run 1 - Stages 1-3 on the 313 training / 78 validation images, ending in a frozen
     selection. NO test-set access is possible in this step.
  3. Experiment run 2 - the complete experiment again into a scratch directory (validation only).
  4. Compare the two runs (configs, hashes, losses, validation errors, thresholds, winners). If
     anything differs the script STOPS here, before the final test is ever touched.
  5. If the frozen selection is a non-baseline candidate: score the final test EXACTLY ONCE with
     it, compare against the protected baseline and apply the pre-declared adoption rule. If the
     frozen selection IS the baseline configuration there is nothing new to score: the baseline
     is retained and its already-recorded Phase 3 result stands.
  6. Leakage audit, then write backend/ai_models/<category>/phase4_optimization/reports/.

Trains only the category given; registers nothing for serving.

Usage (from backend/, venv active):
    python scripts/optimize_category_phase4.py --category hazelnut
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION, resolve_outputs  # noqa: E402
from app.ai.evaluation.category_phase4 import Phase3Reference, compare_experiments, run_optimization_experiment  # noqa: E402
from app.ai.evaluation.category_phase4_final import (  # noqa: E402
    audit_phase4_leakage,
    compare_to_baseline,
    evaluate_locked_on_final_test,
)
from app.ai.evaluation.category_phase4_report import (  # noqa: E402
    build_category_phase4_report,
    ensure_no_phase4_overwrite,
    phase4_report_path,
    phase4_root,
)
from app.ai.evaluation.phase3_report import save_phase3_report  # noqa: E402
from app.ai.training import discover_train_samples  # noqa: E402
from app.ai.training.validation_split import split_train_validation  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _table(experiment) -> None:
    header = ("Candidate", "Input", "Ep", "LR", "TrainS", "FinalLoss", "ValCoV", "ValFP@sel", "SelThreshold", "ValMs/img", "Size")
    rows = []
    for o in experiment.outcomes.values():
        tc = o.config.training_config
        sel = o.selection.selected
        fp = next((s for s in o.selection.stability_by_candidate
                   if sel and s.candidate.method == sel.method and s.candidate.parameter == sel.parameter), None)
        rows.append((
            o.candidate_id, tc.image_size[0], tc.epochs, f"{tc.learning_rate:g}",
            f"{o.training_duration_s:.0f}{'*' if o.reused_phase3_artifact else ''}", f"{o.final_training_loss:.6f}",
            f"{o.coefficient_of_variation:.4f}",
            f"{fp.validation_false_positive_count}/{len(o.validation_errors)}" if fp else "none",
            f"{sel.threshold:.6f}" if sel else "none",
            f"{o.validation_evaluation_ms / len(o.validation_errors):.1f}", o.model_metadata.artifact_size_bytes,
        ))
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header, None] + rows:
        print("-+-".join("-" * x for x in w) if r is None else " | ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))
    print("  (* = protected Phase 3 baseline artifact, training time/loss from its Phase 3 report)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    category = parser.parse_args().category

    branch, head = _git("rev-parse", "--abbrev-ref", "HEAD"), _git("rev-parse", "HEAD")
    print(f"Git branch: {branch}  HEAD: {head}\nCategory: {category}")

    p3 = resolve_outputs(category)
    if not (p3.model_path.is_file() and p3.report_path.is_file()):
        sys.exit(f"Protected Phase 3 baseline for '{category}' not found ({p3.model_path}).")
    p3_report = json.loads(p3.report_path.read_text(encoding="utf-8"))
    baseline_md5 = _md5(p3.model_path)
    if baseline_md5 != p3_report["model"]["md5"]:
        sys.exit("!!! STOP: Phase 3 baseline artifact does not match the hash in its own report. !!!")
    if p3_report["validation_protocol_status"] != "PHASE 3 PROTOCOL PASSED":
        sys.exit("!!! STOP: the Phase 3 baseline did not pass its protocol. !!!")
    ensure_no_phase4_overwrite(category)  # raises FileExistsError - never overwrite
    root = phase4_root(category)
    print(f"Protected Phase 3 baseline MD5 {baseline_md5} (verified against its report)\n")

    split = split_train_validation(discover_train_samples(category), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
    print(f"Split: training={len(split.training)} validation={len(split.validation)}\n")
    reference = Phase3Reference(
        model_path=p3.model_path,
        training_duration_s=p3_report["training_result"]["duration_seconds"],
        final_loss=p3_report["training_result"]["final_loss"],
        num_training_images=p3_report["training_result"]["num_training_images"],
        loss_history=p3_report["training_result"]["loss_history"],
    )

    print("################ EXPERIMENT RUN 1 (validation only) ################")
    exp1 = run_optimization_experiment(category, split, root / "candidates", reference)
    print("\nRun 1 candidate table:")
    _table(exp1)
    print(f"\nFROZEN: {exp1.locked.candidate_id} threshold {exp1.locked.threshold!r} "
          f"({exp1.locked.threshold_method} {exp1.locked.threshold_parameter}) digest {exp1.locked.digest[:16]}...\n")

    print("################ EXPERIMENT RUN 2 (independent full re-run, scratch dir) ################")
    with tempfile.TemporaryDirectory() as tmp:
        exp2 = run_optimization_experiment(category, split, Path(tmp) / "candidates", reference)
    repro = compare_experiments(exp1, exp2)
    print("\nReproducibility:")
    for k, v in repro.items():
        print(f"  {k}: {v}")
    if not repro["all_identical"]:
        print("\n!!! STOP: the two experiment runs differ. The final test set has NOT been touched. "
              "Investigate before proceeding. !!!")
        sys.exit(2)

    locked = exp1.locked
    final = comparison = None
    baseline_final = p3_report["final_test_at_selected_threshold"]
    if locked.is_baseline_configuration:
        decision = "PHASE 3 BASELINE RETAINED (the frozen selection is the baseline configuration itself)"
        print(f"\n{decision}. No new final-test scoring is needed or performed.")
    else:
        selected_dir = root / "selected_candidate"
        selected_dir.mkdir(parents=True)
        shutil.copyfile(locked.model_path, selected_dir / "autoencoder.pt")
        print("\n################ FINAL TEST (scored exactly once, selection frozen) ################")
        final = evaluate_locked_on_final_test(category, locked, exp1.events)
        comparison = compare_to_baseline(p3_report["final_test_predictions_at_selected_threshold"], baseline_final, final)
        decision = comparison["decision"]
        print(json.dumps({k: v for k, v in final.metrics.items()}, indent=1))
        print("recall by defect type:", json.dumps(final.recall_by_defect_type))
        print("comparison:", json.dumps(comparison, indent=1))

    leakage = audit_phase4_leakage(category, split, [exp1, exp2], final, p3_report["validation_statistics"])
    print("\nLeakage audit:")
    for k, v in leakage.items():
        print(f"  {k}: {v}")

    protected = {
        "hazelnut_phase3_model_md5_before": baseline_md5,
        "hazelnut_phase3_model_md5_after": _md5(p3.model_path),
        "phase3_report_unchanged_hash_note": "Phase 3 report file is never written by this script.",
    }
    protected["unchanged"] = protected["hazelnut_phase3_model_md5_before"] == protected["hazelnut_phase3_model_md5_after"]

    baseline = {
        "candidate": "phase3 (input128_epochs15_lr0.001)",
        "model_md5": baseline_md5,
        "threshold": p3_report["threshold_selection"]["selected_threshold"],
        "final_test": baseline_final,
        "final_test_ms_per_image": p3_report["timing"]["final_test_mean_ms_per_image"],
    }
    report = build_category_phase4_report(
        category=category, git_branch=branch, git_head=head, baseline=baseline, experiment=exp1,
        reproducibility=repro, final=final, comparison=comparison, decision=decision, leakage=leakage,
        experiment2_total_ms=exp2.total_ms, protected_artifacts=protected,
    )
    path = save_phase3_report(report, phase4_report_path(category))
    print(f"\nDECISION: {decision}\nReport: {path}\nProtected baseline unchanged: {protected['unchanged']}")
    if not leakage["all_passed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
