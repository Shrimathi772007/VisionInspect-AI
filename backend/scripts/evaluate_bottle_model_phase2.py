"""Milestone 4 Phase 2: leakage-controlled threshold-selection experiment.

Order of operations (matches the Phase 2 spec exactly):

  1. Reproduce the Phase 1 baseline with the existing, UNMODIFIED Phase 1
     code (evaluate_model_validated) and verify it still matches the
     documented Phase 1 numbers. STOP if it does not.
  2. Compute calibration/held-out reconstruction errors (167 / 42 of the 209
     train/good images - Phase 1's own split, fraction=0.8, seed=42).
  3. Generate 9 threshold candidates (5 mean+K*std, 4 percentile) from the
     calibration errors ONLY.
  4. Lock in a selection using ONLY calibration + held-out-42 data
     (app.ai.evaluation.threshold_selection.select_threshold). This is
     printed and recorded BEFORE step 5 runs.
  5. Only now: compute final-test (83 image) reconstruction errors and score
     every candidate against them, purely for reporting/comparison - this
     cannot change the step-4 decision, which already happened.
  6. Repeat steps 2-5 a second time and diff every number for reproducibility.
  7. Write backend/ai_models/bottle/evaluation_reports/phase2_threshold_experiment_report.json
     (Phase 1's report file is untouched).

Does not train, retrain, or modify the model, and does not change the
production threshold used by app.ai.inference.predict.predict_image.

Usage (from backend/, with the venv activated):
    python scripts/evaluate_bottle_model_phase2.py
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation import compute_model_metadata, evaluate_model_validated  # noqa: E402
from app.ai.evaluation.calibration import split_for_calibration  # noqa: E402
from app.ai.evaluation.phase2_report import build_phase2_report, default_phase2_report_path, save_phase2_report  # noqa: E402
from app.ai.evaluation.threshold_experiments import (  # noqa: E402
    ThresholdCandidate,
    compute_errors_for_samples,
    evaluate_candidate_on_final_test,
    generate_candidates,
)
from app.ai.evaluation.threshold_selection import SelectionResult, select_threshold  # noqa: E402
from app.ai.training import compute_statistics, discover_test_samples, discover_train_samples, load_model_for_category  # noqa: E402
from app.ai.training.artifacts import get_model_path  # noqa: E402

CATEGORY = "bottle"
IMAGE_SIZE = (128, 128)
CALIBRATION_FRACTION = 0.8
CALIBRATION_SEED = 42

# Documented Phase 1 validated baseline (backend/ai_models/bottle/evaluation_reports/phase1_validation_report.json)
EXPECTED_PHASE1_THRESHOLD = 0.003208
EXPECTED_PHASE1_CONFUSION_MATRIX = [[19, 1], [33, 30]]
PHASE1_THRESHOLD_TOLERANCE = 1e-4


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"<unavailable: {exc}>"


@dataclass
class PipelineRun:
    calibration_errors: list[float]
    held_out_errors: list[float]
    candidates: list[ThresholdCandidate]
    selection: SelectionResult
    test_labels: list[int]
    test_errors: list[float]
    candidate_results: list


def run_pipeline(model) -> PipelineRun:
    train_samples = discover_train_samples(CATEGORY)
    split = split_for_calibration(train_samples, fraction=CALIBRATION_FRACTION, seed=CALIBRATION_SEED)

    calibration_errors = compute_errors_for_samples(model, split.calibration, IMAGE_SIZE)
    held_out_errors = compute_errors_for_samples(model, split.held_out, IMAGE_SIZE)

    candidates = generate_candidates(calibration_errors)

    # --- Selection is LOCKED here, using only calibration + held-out data. ---
    selection = select_threshold(candidates, held_out_errors)

    # --- Only now does the final test set get touched, for reporting only. ---
    test_samples = discover_test_samples(CATEGORY)
    test_labels = [s.label for s in test_samples]
    test_errors = compute_errors_for_samples(model, test_samples, IMAGE_SIZE)
    candidate_results = [evaluate_candidate_on_final_test(c, test_labels, test_errors) for c in candidates]

    return PipelineRun(
        calibration_errors=calibration_errors,
        held_out_errors=held_out_errors,
        candidates=candidates,
        selection=selection,
        test_labels=test_labels,
        test_errors=test_errors,
        candidate_results=candidate_results,
    )


def _print_table(rows: list[tuple]) -> None:
    header = ("Method", "Parameter", "Threshold", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP")
    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    def fmt(row):
        return " | ".join(str(v).rjust(widths[i]) for i, v in enumerate(row))
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    head = _git("rev-parse", "HEAD")
    print(f"Git branch: {branch}  HEAD: {head}")
    print()

    model_path = get_model_path(CATEGORY)
    load_start = time.perf_counter()
    model = load_model_for_category(CATEGORY)
    model_load_ms = (time.perf_counter() - load_start) * 1000
    print(f"Loaded trained model for category={CATEGORY!r} in {model_load_ms:.2f} ms")

    # --- Step 2: reproduce the Phase 1 baseline with the UNMODIFIED Phase 1 code. ---
    print()
    print("=== STEP: Reproducing Phase 1 baseline (evaluate_model_validated, unmodified) ===")
    phase1 = evaluate_model_validated(CATEGORY, model, image_size=IMAGE_SIZE)
    threshold_diff = abs(phase1.threshold - EXPECTED_PHASE1_THRESHOLD)
    matches_threshold = threshold_diff <= PHASE1_THRESHOLD_TOLERANCE
    matches_matrix = phase1.confusion_matrix == EXPECTED_PHASE1_CONFUSION_MATRIX

    print(f"  Reproduced threshold: {phase1.threshold:.6f}  (expected ~{EXPECTED_PHASE1_THRESHOLD:.6f}, "
          f"diff={threshold_diff:.6f}, within tolerance={matches_threshold})")
    print(f"  Reproduced confusion matrix: {phase1.confusion_matrix}  (expected {EXPECTED_PHASE1_CONFUSION_MATRIX}, "
          f"matches={matches_matrix})")

    if not (matches_threshold and matches_matrix):
        print()
        print("!!! STOP: Phase 1 baseline did NOT reproduce as documented. !!!")
        print("Refusing to proceed with Phase 2 threshold selection until this is investigated.")
        sys.exit(1)

    print("  Phase 1 baseline reproduced successfully. Proceeding.")
    print()

    # --- Steps 3-5, run twice for reproducibility (Step 10). ---
    print("=== Running the full threshold experiment pipeline twice (reproducibility check) ===")
    experiment_start = time.perf_counter()
    run1 = run_pipeline(model)
    run1_ms = (time.perf_counter() - experiment_start) * 1000
    run2_start = time.perf_counter()
    run2 = run_pipeline(model)
    run2_ms = (time.perf_counter() - run2_start) * 1000

    thresholds_match = [c1.threshold == c2.threshold for c1, c2 in zip(run1.candidates, run2.candidates)]
    predictions_match = (
        [r1.true_negatives, r1.false_positives, r1.false_negatives, r1.true_positives]
        == [r2.true_negatives, r2.false_positives, r2.false_negatives, r2.true_positives]
        for r1, r2 in zip(run1.candidate_results, run2.candidate_results)
    )
    predictions_match = list(predictions_match)
    metrics_match = [
        (r1.accuracy, r1.precision, r1.recall, r1.f1_score) == (r2.accuracy, r2.precision, r2.recall, r2.f1_score)
        for r1, r2 in zip(run1.candidate_results, run2.candidate_results)
    ]
    selection_matches = (run1.selection.status == run2.selection.status) and (
        (run1.selection.selected is None and run2.selection.selected is None)
        or (
            run1.selection.selected is not None
            and run2.selection.selected is not None
            and run1.selection.selected.method == run2.selection.selected.method
            and run1.selection.selected.parameter == run2.selection.selected.parameter
            and run1.selection.selected.threshold == run2.selection.selected.threshold
        )
    )

    reproducible = all(thresholds_match) and all(predictions_match) and all(metrics_match) and selection_matches
    print(f"  Run 1: {run1_ms:.1f} ms   Run 2: {run2_ms:.1f} ms")
    print(f"  Thresholds identical across runs: {all(thresholds_match)}")
    print(f"  Predictions (confusion matrices) identical across runs: {all(predictions_match)}")
    print(f"  Metrics identical across runs: {all(metrics_match)}")
    print(f"  Selection identical across runs: {selection_matches}")

    if not reproducible:
        print()
        print("!!! STOP: threshold experiment is NOT reproducible across two runs. !!!")
        sys.exit(1)

    print("  Reproducibility verified. Using run 1 for the report.")
    print()

    run = run1

    # --- Cross-check: the mean_std K=3.0 candidate must equal the Phase 1 baseline (same formula, same data). ---
    k3_candidate = next(c for c in run.candidates if c.method == "mean_std" and c.parameter == 3.0)
    print(f"Consistency check - mean_std K=3.0 candidate threshold: {k3_candidate.threshold:.6f} "
          f"vs Phase 1 baseline: {phase1.threshold:.6f} "
          f"(equal={k3_candidate.threshold == phase1.threshold})")
    print()

    print("=== Selection locked from calibration + held-out-42 data (BEFORE final-test scoring) ===")
    print(f"  Status: {run.selection.status}")
    if run.selection.selected is not None:
        print(f"  Selected: {run.selection.selected.method} parameter={run.selection.selected.parameter} "
              f"threshold={run.selection.selected.threshold:.6f}")
    print(f"  Reasoning: {run.selection.reasoning}")
    print()

    print("=== Threshold candidates vs. final 83-image test set (reporting only - not used to select) ===")
    rows = []
    selected_markers = []
    for result in run.candidate_results:
        c = result.candidate
        is_selected = (
            run.selection.selected is not None
            and c.method == run.selection.selected.method
            and c.parameter == run.selection.selected.parameter
        )
        rows.append((
            c.method, c.parameter, f"{c.threshold:.6f}",
            f"{result.precision:.4f}", f"{result.recall:.4f}", f"{result.f1_score:.4f}",
            result.true_negatives, result.false_positives, result.false_negatives, result.true_positives,
        ))
        if is_selected:
            selected_markers.append(f"{c.method} parameter={c.parameter}")
    _print_table(rows)
    for marker in selected_markers:
        print(f"  -> {marker}  <== SELECTED (provisional)")
    print()
    print(f"  (Phase 1 validated baseline for comparison: threshold={phase1.threshold:.6f}, "
          f"precision={phase1.precision:.4f}, recall={phase1.recall:.4f}, f1={phase1.f1_score:.4f}, "
          f"confusion_matrix={phase1.confusion_matrix})")
    print()

    model_metadata = compute_model_metadata(model_path, category=CATEGORY, latent_channels=model.latent_channels)
    dataset_stats = compute_statistics(CATEGORY)

    phase1_baseline_dict = {
        "threshold": phase1.threshold,
        "accuracy": phase1.accuracy,
        "precision": phase1.precision,
        "recall": phase1.recall,
        "f1_score": phase1.f1_score,
        "confusion_matrix": phase1.confusion_matrix,
        "calibration_sample_count": phase1.calibration.calibration_sample_count,
        "held_out_sample_count": phase1.calibration.held_out_sample_count,
    }

    reproducibility = {
        "run_count": 2,
        "run1_duration_ms": run1_ms,
        "run2_duration_ms": run2_ms,
        "thresholds_identical": all(thresholds_match),
        "predictions_identical": all(predictions_match),
        "metrics_identical": all(metrics_match),
        "selection_identical": selection_matches,
        "overall_reproducible": reproducible,
    }

    timing_ms = {
        "model_load_ms": model_load_ms,
        "pipeline_run1_ms": run1_ms,
        "pipeline_run2_ms": run2_ms,
        "calibration_sample_count": len(run.calibration_errors),
        "held_out_sample_count": len(run.held_out_errors),
        "final_test_sample_count": len(run.test_errors),
    }

    report = build_phase2_report(
        model_metadata=model_metadata,
        dataset_stats=dataset_stats,
        git_branch=branch,
        git_head=head,
        calibration_fraction=CALIBRATION_FRACTION,
        calibration_seed=CALIBRATION_SEED,
        calibration_sample_count=len(run.calibration_errors),
        held_out_sample_count=len(run.held_out_errors),
        phase1_baseline=phase1_baseline_dict,
        candidate_results=run.candidate_results,
        selection=run.selection,
        reproducibility=reproducibility,
        timing_ms=timing_ms,
    )
    report_path = save_phase2_report(report, default_phase2_report_path(CATEGORY))
    print(f"Phase 2 threshold experiment report written to: {report_path}")


if __name__ == "__main__":
    main()
