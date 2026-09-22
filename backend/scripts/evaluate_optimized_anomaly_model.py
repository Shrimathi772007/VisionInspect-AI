"""Score a category's optimized frozen anomaly detector on its final test set - exactly once.

Run only after scripts/optimize_anomaly_calibration.py has run `study` and `freeze`. Refuses to run twice, or if
the frozen model or the original baseline artifact no longer match their recorded hashes. Registers nothing for
serving.

Usage (from backend/, venv active):
    python scripts/evaluate_optimized_anomaly_model.py --category carpet
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.calibration_final_test import evaluate_optimized_on_final_test, result_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    category = parser.parse_args().category
    result = evaluate_optimized_on_final_test(category)
    hide = ("predictions", "detected_files", "missed_files", "false_positive_files")
    print(json.dumps({k: v for k, v in result.items() if k not in hide}, indent=1))
    print(f"\nCLASSIFICATION: {result['classification']}\nWrote {result_path(category)}")


if __name__ == "__main__":
    main()
