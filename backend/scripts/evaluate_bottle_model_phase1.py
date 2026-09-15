"""Milestone 4 Phase 1: reproducible, leakage-free evaluation of the trained
MVTec `bottle` autoencoder, with a persisted JSON report.

Runs two evaluations against the identical, untouched test set
(bottle/test/good + every bottle/test/<defect_type>/*, 83 images):

1. BASELINE REPRODUCTION - app.ai.evaluation.evaluate_model, unchanged,
   using the original train-good-based threshold methodology. This should
   reproduce the documented Milestone 2 baseline (threshold ~0.003212,
   accuracy 57.83%, precision 96.67%, recall 46.03%, F1 62.37%,
   confusion matrix [[19,1],[34,29]]).

2. VALIDATED PHASE 1 EVALUATION - app.ai.evaluation.evaluate_model_validated,
   using a threshold calibrated on a deterministic subset of train/good
   reconstruction errors (see app.ai.evaluation.calibration), never touching
   the final test set for calibration.

Does not train, retrain, or modify the model in any way. Writes
backend/ai_models/bottle/evaluation_reports/phase1_validation_report.json
(backend/ai_models/ is already gitignored - this is not committed).

Usage (from backend/, with the venv activated):
    python scripts/evaluate_bottle_model_phase1.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation import (  # noqa: E402
    build_phase1_report,
    compute_model_metadata,
    default_report_path,
    evaluate_model,
    evaluate_model_validated,
    save_report,
)
from app.ai.training import compute_statistics, load_model_for_category  # noqa: E402
from app.ai.training.artifacts import get_model_path  # noqa: E402
from app.ai.training.schemas import TrainingConfig  # noqa: E402

CATEGORY = "bottle"


def _print_evaluation(label: str, result) -> None:
    print(f"--- {label} ---")
    print(f"Threshold: {result.threshold:.6f}")
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1-score:  {result.f1_score:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    for row in result.confusion_matrix:
        print(f"  {row}")
    print(f"Correct: {result.num_correct}/{result.total_test_samples}  "
          f"Incorrect: {result.num_incorrect}/{result.total_test_samples}")
    print()


def main() -> None:
    model_path = get_model_path(CATEGORY)

    load_start = time.perf_counter()
    model = load_model_for_category(CATEGORY)
    model_load_ms = (time.perf_counter() - load_start) * 1000
    print(f"Loaded trained model for category={CATEGORY!r} in {model_load_ms:.2f} ms")
    print()

    dataset_stats = compute_statistics(CATEGORY)
    print(f"Dataset: train/good={dataset_stats.train_good_count}  "
          f"test/good={dataset_stats.test_good_count}  "
          f"test/defective={dataset_stats.test_defective_count}  "
          f"test/total={dataset_stats.test_total_count}")
    print(f"Test defect-type breakdown: {dataset_stats.test_defect_type_counts}")
    print()

    print("Reproducing the original baseline methodology (train-good-based threshold)...")
    baseline = evaluate_model(CATEGORY, model)
    _print_evaluation("BASELINE REPRODUCTION (original methodology, unchanged)", baseline)

    print("Running the Phase 1 validated protocol (calibration-split threshold)...")
    validated = evaluate_model_validated(CATEGORY, model, model_load_ms=model_load_ms)
    _print_evaluation("VALIDATED PHASE 1 EVALUATION (calibration-split threshold)", validated)

    print(
        f"Calibration: {validated.calibration.calibration_sample_count} of "
        f"{dataset_stats.train_good_count} train/good errors used for threshold "
        f"(fraction={validated.calibration.calibration_fraction}, seed={validated.calibration.calibration_seed}); "
        f"{validated.calibration.held_out_sample_count} held out, "
        f"{validated.calibration.held_out_false_positive_count} of which exceed the threshold "
        "(calibration-stability check only, not a generalization guarantee - see calibration.py docstring)."
    )
    print()

    if validated.timing is not None:
        t = validated.timing
        print(
            f"Timing over {t.final_test_sample_count} final-test images: "
            f"preprocessing mean={t.mean_preprocessing_ms_per_image:.3f} ms/image, "
            f"inference mean={t.mean_inference_ms_per_image:.3f} ms/image, "
            f"total mean={t.mean_total_ms_per_image:.3f} ms/image "
            f"(model load, one-time: {t.model_load_ms:.2f} ms)."
        )
        print()

    model_metadata = compute_model_metadata(
        model_path,
        category=CATEGORY,
        latent_channels=model.latent_channels,
    )
    print(f"Model artifact: {model_metadata.artifact_path}")
    print(f"  SHA-256: {model_metadata.sha256}")
    print(f"  MD5:     {model_metadata.md5}")
    print(f"  torch={model_metadata.torch_version}  numpy={model_metadata.numpy_version}  "
          f"scikit-learn={model_metadata.scikit_learn_version}  opencv={model_metadata.opencv_version}")
    print()

    report = build_phase1_report(
        model_metadata=model_metadata,
        dataset_stats=dataset_stats,
        training_seed=TrainingConfig(category=CATEGORY).seed,
        baseline=baseline,
        validated=validated,
    )
    report_path = save_report(report, default_report_path(CATEGORY))
    print(f"Phase 1 evaluation report written to: {report_path}")


if __name__ == "__main__":
    main()
