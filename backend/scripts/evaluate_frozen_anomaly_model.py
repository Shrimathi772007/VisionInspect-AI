"""Evaluate a category's FROZEN anomaly detector on its final test set - exactly once.

Run only after scripts/select_category_anomaly_model.py has selected and frozen a model. Refuses to run if a
final-test result already exists for the category, or if the frozen model's hashes no longer match its lock
record. Registers nothing for serving.

Usage (from backend/, venv active):
    python scripts/evaluate_frozen_anomaly_model.py --category carpet
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.anomaly_final_test import evaluate_frozen_on_final_test, final_result_path  # noqa: E402
from app.ai.training import artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    category = parser.parse_args().category
    frozen_dir = artifacts.ARTIFACTS_ROOT / category / "selected_model"
    if not (frozen_dir / "frozen_model.json").is_file():
        sys.exit(f"No frozen model at {frozen_dir}")
    result = evaluate_frozen_on_final_test(category, frozen_dir)
    printable = {k: v for k, v in result.items() if k not in ("predictions", "detected_files", "missed_files",
                                                              "false_positive_files")}
    print(json.dumps(printable, indent=1))
    print(f"\nCLASSIFICATION: {result['classification']}\nWrote {final_result_path(category)}")


if __name__ == "__main__":
    main()
