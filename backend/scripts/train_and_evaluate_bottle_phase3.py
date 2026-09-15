"""Milestone 4 Phase 3: train a NEW autoencoder on a true train/validation
split, select a threshold from genuine (unseen) validation reconstruction
errors, and evaluate once on the untouched final 83-image test set.

Order of operations (matches the Phase 3 spec exactly):

  1. Verify the dataset (209 train/good, 83 final test) and the original
     Phase 1/2 production model's MD5 (must stay unchanged throughout).
  2. Deterministically split the 209 train/good samples into a training
     subset (167) and a validation subset (42) - seed=42, fraction=0.8
     (app.ai.training.validation_split).
  3. Train a NEW autoencoder on ONLY the training subset - identical
     architecture/hyperparameters to the Phase 1 production model, saved to
     a separate path (backend/ai_models/bottle/phase3_validation/autoencoder.pt).
     The validation subset never reaches the DataLoader.
  4. Compute reconstruction errors for the validation subset with the new,
     trained model - genuinely unseen data.
  5. Generate 9 threshold candidates from validation errors only, and LOCK
     IN a selection (app.ai.evaluation.phase3_threshold_selection) using
     only validation false-positive behavior - before the final test set is
     touched at all.
  6. Only now: evaluate every candidate (and note the selected one) against
     the untouched final 83-image test set, for reporting.
  7. Repeat steps 3-6 a second time (a fresh, separately-trained model) and
     diff every number for reproducibility.
  8. Reproduce the Phase 1 baseline (existing, unmodified code) and load the
     already-persisted Phase 2 provisional result (never recomputed/altered)
     for the final comparison.
  9. Write backend/ai_models/bottle/evaluation_reports/phase3_validated_model_report.json.
     Verify the original model artifact's MD5 is still unchanged.

Does not modify the original Phase 1 model, Phase 1/2 code, or any
production inference behavior.

Usage (from backend/, with the venv activated):
    python scripts/train_and_evaluate_bottle_phase3.py
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation import compute_model_metadata, evaluate_model_validated  # noqa: E402
from app.ai.evaluation.phase3_report import build_phase3_report, default_phase3_report_path, save_phase3_report  # noqa: E402
from app.ai.evaluation.phase3_threshold_selection import Phase3SelectionResult, select_threshold_from_validation  # noqa: E402
from app.ai.evaluation.report import default_report_path  # noqa: E402
from app.ai.evaluation.threshold_experiments import (  # noqa: E402
    ThresholdCandidate,
    compute_errors_for_samples,
    evaluate_candidate_on_final_test,
    generate_candidates,
)
from app.ai.training import discover_test_samples, discover_train_samples  # noqa: E402
from app.ai.training.artifacts import get_model_path  # noqa: E402
from app.ai.training.phase3_train import train_anomaly_model_on_samples  # noqa: E402
from app.ai.training.schemas import TrainingConfig  # noqa: E402
from app.ai.training.validation_split import split_train_validation  # noqa: E402

CATEGORY = "bottle"
IMAGE_SIZE = (128, 128)
PHASE3_MODEL_NAME = "phase3_validation/autoencoder"


def _phase2_report_path() -> Path:
    from app.ai.evaluation.phase2_report import default_phase2_report_path

    return default_phase2_report_path(CATEGORY)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"<unavailable: {exc}>"


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class PipelineRun:
    model_path: Path
    model_md5: str
    training_duration_s: float
    final_loss: float
    training_count: int
    validation_count: int
    validation_errors: list[float]
    validation_mean: float
    validation_std: float
    validation_min: float
    validation_max: float
    candidates: list[ThresholdCandidate]
    selection: Phase3SelectionResult
    test_labels: list[int]
    test_errors: list[float]
    candidate_results: list
    validation_inference_ms: float
    final_test_inference_ms: float


def run_pipeline(model_path: Path, config: TrainingConfig) -> PipelineRun:
    train_samples = discover_train_samples(CATEGORY)
    split = split_train_validation(train_samples, training_fraction=0.8, seed=42)

    # --- Defensive, in-script leakage/overlap guard (the formal proof lives in tests). ---
    training_paths = {s.path for s in split.training}
    validation_paths = {s.path for s in split.validation}
    assert not (training_paths & validation_paths), "training/validation overlap detected"

    training_result, model = train_anomaly_model_on_samples(split.training, config, model_path)
    model_md5 = _file_md5(training_result.model_path)

    val_start = time.perf_counter()
    validation_errors = compute_errors_for_samples(model, split.validation, IMAGE_SIZE)
    validation_inference_ms = (time.perf_counter() - val_start) * 1000

    import statistics as stats_mod
    validation_mean = stats_mod.fmean(validation_errors)
    validation_std = (sum((e - validation_mean) ** 2 for e in validation_errors) / len(validation_errors)) ** 0.5

    candidates = generate_candidates(validation_errors)

    # --- Selection LOCKED here, using only genuine validation data. ---
    selection = select_threshold_from_validation(candidates, validation_errors)

    # --- Only now does the final test set get touched, for reporting only. ---
    test_samples = discover_test_samples(CATEGORY)
    test_labels = [s.label for s in test_samples]
    test_start = time.perf_counter()
    test_errors = compute_errors_for_samples(model, test_samples, IMAGE_SIZE)
    final_test_inference_ms = (time.perf_counter() - test_start) * 1000
    candidate_results = [evaluate_candidate_on_final_test(c, test_labels, test_errors) for c in candidates]

    return PipelineRun(
        model_path=training_result.model_path,
        model_md5=model_md5,
        training_duration_s=training_result.duration_seconds,
        final_loss=training_result.final_loss,
        training_count=len(split.training),
        validation_count=len(split.validation),
        validation_errors=validation_errors,
        validation_mean=validation_mean,
        validation_std=validation_std,
        validation_min=min(validation_errors),
        validation_max=max(validation_errors),
        candidates=candidates,
        selection=selection,
        test_labels=test_labels,
        test_errors=test_errors,
        candidate_results=candidate_results,
        validation_inference_ms=validation_inference_ms,
        final_test_inference_ms=final_test_inference_ms,
    )


def _print_table(rows: list[tuple], header: tuple) -> None:
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

    original_model_path = get_model_path(CATEGORY)
    original_md5_before = _file_md5(original_model_path)
    print(f"Original production model: {original_model_path}")
    print(f"  MD5 before Phase 3: {original_md5_before}")
    if original_md5_before != "2f470c30834baf80252f1d352c7fb37f":
        print("!!! STOP: original model MD5 does not match the documented reference. !!!")
        sys.exit(1)
    print()

    config = TrainingConfig(category=CATEGORY, image_size=IMAGE_SIZE, batch_size=8, epochs=15,
                             learning_rate=1e-3, seed=42)
    print(f"Phase 3 training config: image_size={config.image_size} batch_size={config.batch_size} "
          f"epochs={config.epochs} lr={config.learning_rate} seed={config.seed}")
    print()

    canonical_model_path = get_model_path(CATEGORY, model_name=PHASE3_MODEL_NAME)

    print("=== Run 1: training + validation + selection + final-test scoring ===")
    experiment_start = time.perf_counter()
    run1 = run_pipeline(canonical_model_path, config)
    run1_total_ms = (time.perf_counter() - experiment_start) * 1000
    print(f"  Trained on {run1.training_count} images, validated on {run1.validation_count} images.")
    print(f"  Training duration: {run1.training_duration_s:.2f} s   Final training loss: {run1.final_loss:.6f}")
    print(f"  Phase 3 model MD5 (run 1): {run1.model_md5}")
    print(f"  Validation errors: mean={run1.validation_mean:.6f} std={run1.validation_std:.6f} "
          f"min={run1.validation_min:.6f} max={run1.validation_max:.6f}")
    print()

    print("=== Run 2: independent retraining, for reproducibility comparison ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        run2_model_path = Path(tmpdir) / "phase3_repro_check" / "autoencoder.pt"
        run2_start = time.perf_counter()
        run2 = run_pipeline(run2_model_path, config)
        run2_total_ms = (time.perf_counter() - run2_start) * 1000
    print(f"  Training duration: {run2.training_duration_s:.2f} s   Final training loss: {run2.final_loss:.6f}")
    print(f"  Phase 3 model MD5 (run 2): {run2.model_md5}")
    print()

    split_identical = run1.training_count == run2.training_count and run1.validation_count == run2.validation_count
    model_hash_identical = run1.model_md5 == run2.model_md5
    validation_errors_identical = run1.validation_errors == run2.validation_errors
    thresholds_identical = [c1.threshold for c1 in run1.candidates] == [c2.threshold for c2 in run2.candidates]
    predictions_identical = all(
        (r1.true_negatives, r1.false_positives, r1.false_negatives, r1.true_positives)
        == (r2.true_negatives, r2.false_positives, r2.false_negatives, r2.true_positives)
        for r1, r2 in zip(run1.candidate_results, run2.candidate_results)
    )
    metrics_identical = all(
        (r1.accuracy, r1.precision, r1.recall, r1.f1_score) == (r2.accuracy, r2.precision, r2.recall, r2.f1_score)
        for r1, r2 in zip(run1.candidate_results, run2.candidate_results)
    )
    selection_identical = (run1.selection.status == run2.selection.status) and (
        (run1.selection.selected is None and run2.selection.selected is None)
        or (
            run1.selection.selected is not None and run2.selection.selected is not None
            and run1.selection.selected.method == run2.selection.selected.method
            and run1.selection.selected.parameter == run2.selection.selected.parameter
            and run1.selection.selected.threshold == run2.selection.selected.threshold
        )
    )

    print("=== Reproducibility ===")
    print(f"  Split sizes identical: {split_identical}")
    print(f"  Model weight hash (MD5) identical: {model_hash_identical}  "
          f"(bit-identical weights are not required - predictions/metrics below are the real check)")
    print(f"  Validation reconstruction errors identical: {validation_errors_identical}")
    print(f"  Candidate thresholds identical: {thresholds_identical}")
    print(f"  Final-test predictions (confusion matrices) identical: {predictions_identical}")
    print(f"  Final-test metrics identical: {metrics_identical}")
    print(f"  Selection identical: {selection_identical}")

    scientifically_reproducible = (
        split_identical and validation_errors_identical and thresholds_identical
        and predictions_identical and metrics_identical and selection_identical
    )
    if not scientifically_reproducible:
        print()
        print("!!! STOP: Phase 3 scientific result (predictions/metrics/selection) is NOT "
              "reproducible across two independent training runs. !!!")
        sys.exit(1)
    print("  Scientific result reproducible (predictions/metrics/selection identical). Using run 1 for the report.")
    print()

    run = run1

    print("=== Selection locked from genuine validation data (BEFORE final-test scoring) ===")
    print(f"  Status: {run.selection.status}")
    if run.selection.selected is not None:
        print(f"  Selected: {run.selection.selected.method} parameter={run.selection.selected.parameter} "
              f"threshold={run.selection.selected.threshold:.6f}")
    print(f"  Reasoning: {run.selection.reasoning}")
    print()

    print("=== Validation threshold candidates (false-positive behavior only - no recall/F1 here) ===")
    val_rows = []
    for stability in run.selection.stability_by_candidate:
        c = stability.candidate
        selected_flag = "YES" if (
            run.selection.selected is not None and c.method == run.selection.selected.method
            and c.parameter == run.selection.selected.parameter
        ) else "no"
        val_rows.append((
            c.method, c.parameter, f"{c.threshold:.6f}",
            stability.validation_false_positive_count, f"{stability.validation_false_positive_rate:.1%}",
            selected_flag,
        ))
    _print_table(val_rows, ("Method", "Parameter", "Threshold", "ValFP", "ValFPRate", "Selected"))
    print()

    print("=== Threshold candidates vs. final 83-image test set (reporting only - not used to select) ===")
    rows = []
    for result in run.candidate_results:
        c = result.candidate
        rows.append((
            c.method, c.parameter, f"{c.threshold:.6f}",
            f"{result.precision:.4f}", f"{result.recall:.4f}", f"{result.f1_score:.4f}",
            result.true_negatives, result.false_positives, result.false_negatives, result.true_positives,
        ))
    _print_table(rows, ("Method", "Parameter", "Threshold", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"))
    print()

    # --- Phase 1 baseline: reproduced live with the existing, unmodified code. ---
    print("=== Reproducing Phase 1 baseline (unmodified evaluate_model_validated) ===")
    from app.ai.training import load_model_for_category
    phase1_model = load_model_for_category(CATEGORY)
    phase1 = evaluate_model_validated(CATEGORY, phase1_model, image_size=IMAGE_SIZE)
    print(f"  threshold={phase1.threshold:.6f} precision={phase1.precision:.4f} recall={phase1.recall:.4f} "
          f"f1={phase1.f1_score:.4f} confusion_matrix={phase1.confusion_matrix}")
    print()

    # --- Phase 2 provisional result: loaded from its already-persisted report, never recomputed. ---
    phase2_provisional = None
    phase2_path = _phase2_report_path()
    if phase2_path.is_file():
        phase2_data = json.loads(phase2_path.read_text(encoding="utf-8"))
        phase2_provisional = phase2_data.get("selection", {}).get("selected_final_test_result")
        print(f"Loaded Phase 2 provisional result from: {phase2_path}")
    else:
        print(f"Phase 2 report not found at {phase2_path} - proceeding without it.")
    print()

    # --- Original model integrity check. ---
    original_md5_after = _file_md5(original_model_path)
    print(f"Original production model MD5 after Phase 3: {original_md5_after}  "
          f"(unchanged={original_md5_after == original_md5_before})")
    if original_md5_after != original_md5_before:
        print("!!! STOP: original production model artifact was modified. !!!")
        sys.exit(1)
    print()

    original_metadata = compute_model_metadata(original_model_path, category=CATEGORY)
    phase3_metadata = compute_model_metadata(
        run.model_path, category=CATEGORY, model_name=PHASE3_MODEL_NAME, latent_channels=128
    )

    report = build_phase3_report(
        git_branch=branch,
        git_head=head,
        original_model_metadata=original_metadata,
        phase3_model_metadata=phase3_metadata,
        training_config={
            "architecture": "conv_autoencoder",
            "image_size": list(config.image_size),
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
        },
        split={
            "training_fraction": 0.8,
            "seed": 42,
            "training_count": run.training_count,
            "validation_count": run.validation_count,
            "final_test_count": len(run.test_labels),
        },
        training_result={
            "duration_seconds": run.training_duration_s,
            "final_loss": run.final_loss,
        },
        validation_statistics={
            "sample_count": run.validation_count,
            "mean_error": run.validation_mean,
            "std_error": run.validation_std,
            "min_error": run.validation_min,
            "max_error": run.validation_max,
        },
        candidate_results=run.candidate_results,
        selection=run.selection,
        phase1_baseline={
            "threshold": phase1.threshold,
            "accuracy": phase1.accuracy,
            "precision": phase1.precision,
            "recall": phase1.recall,
            "f1_score": phase1.f1_score,
            "confusion_matrix": phase1.confusion_matrix,
        },
        phase2_provisional=phase2_provisional,
        reproducibility={
            "run_count": 2,
            "split_identical": split_identical,
            "model_hash_identical": model_hash_identical,
            "validation_errors_identical": validation_errors_identical,
            "thresholds_identical": thresholds_identical,
            "predictions_identical": predictions_identical,
            "metrics_identical": metrics_identical,
            "selection_identical": selection_identical,
            "overall_scientifically_reproducible": scientifically_reproducible,
        },
        timing_ms={
            "run1_total_ms": run1_total_ms,
            "run2_total_ms": run2_total_ms,
            "run1_training_duration_s": run1.training_duration_s,
            "run1_validation_inference_ms": run1.validation_inference_ms,
            "run1_final_test_inference_ms": run1.final_test_inference_ms,
            "run1_validation_mean_ms_per_image": run1.validation_inference_ms / run1.validation_count,
            "run1_final_test_mean_ms_per_image": run1.final_test_inference_ms / len(run1.test_labels),
        },
    )
    report_path = save_phase3_report(report, default_phase3_report_path(CATEGORY))
    print(f"Phase 3 report written to: {report_path}")
    print(f"Phase 3 model saved to: {run.model_path}")


if __name__ == "__main__":
    main()
