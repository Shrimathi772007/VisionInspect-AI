"""Development script: run the actual anomaly-detection training for MVTec `bottle`.

Not part of the automated test suite (deliberately) - this performs a real
training run over all 209 bottle/train/good images and prints the real,
measured TrainingResult.

Usage (from backend/, with the venv activated):
    python scripts/train_bottle_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.training import TrainingConfig, train_anomaly_model  # noqa: E402


def main() -> None:
    config = TrainingConfig(category="bottle")
    print(f"Training config: {config}")

    result = train_anomaly_model(config)

    print()
    print("Training complete.")
    print(f"  category:            {result.category}")
    print(f"  device:               {result.device}")
    print(f"  epochs:               {result.epochs}")
    print(f"  num_training_images:  {result.num_training_images}")
    print(f"  final_loss:           {result.final_loss:.6f}")
    print(f"  duration_seconds:     {result.duration_seconds:.2f}")
    print(f"  model_path:           {result.model_path}")
    print(f"  loss_history:         {[round(v, 6) for v in result.loss_history]}")


if __name__ == "__main__":
    main()
