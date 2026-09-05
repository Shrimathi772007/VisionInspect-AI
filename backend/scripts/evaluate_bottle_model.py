"""Development script: evaluate the already-trained MVTec `bottle` autoencoder
on the complete, unseen bottle/test/ dataset.

Not part of the automated test suite (deliberately) - this runs a real
evaluation over all 209 train/good images (to derive the threshold) and all
83 test images (good + defective), and prints the real, measured
EvaluationResult. Does not retrain or modify the model in any way.

Usage (from backend/, with the venv activated):
    python scripts/evaluate_bottle_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation import evaluate_model  # noqa: E402
from app.ai.training import load_model_for_category  # noqa: E402

CATEGORY = "bottle"


def main() -> None:
    model = load_model_for_category(CATEGORY)
    print(f"Loaded trained model for category={CATEGORY!r}")

    result = evaluate_model(CATEGORY, model)

    print()
    print(f"Category: {result.category}")
    print(f"Training images: {result.num_training_normal_samples}")
    print(f"Test images: {result.total_test_samples}")
    print(f"Good test images: {result.good_test_count}")
    print(f"Defective test images: {result.defective_test_count}")
    print()
    print(f"Threshold (mean + {result.threshold_k}*std of training-normal error): {result.threshold:.6f}")
    print()
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1-score:  {result.f1_score:.4f}")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]] (rows=actual good/defective, cols=predicted good/defective):")
    for row in result.confusion_matrix:
        print(f"  {row}")
    print()
    print(f"Correctly classified:   {result.num_correct} / {result.total_test_samples}")
    print(f"Incorrectly classified: {result.num_incorrect} / {result.total_test_samples}")
    print()
    print(f"Min reconstruction error:              {result.min_error:.6f}")
    print(f"Max reconstruction error:              {result.max_error:.6f}")
    print(f"Mean reconstruction error (all test):   {result.mean_error:.6f}")
    print(f"Mean reconstruction error (actual good): {result.mean_good_error:.6f}")
    print(f"Mean reconstruction error (actual defective): {result.mean_defective_error:.6f}")


if __name__ == "__main__":
    main()
