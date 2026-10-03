# Diagnostic only: ConvAutoencoder at 256x256 on the 224 Carpet TRAINING images (never validation/test).
import sys; sys.path.insert(0, ".")
from pathlib import Path
from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION
from app.ai.training import discover_train_samples
from app.ai.training.artifacts import get_model_path
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.schemas import TrainingConfig
from app.ai.training.validation_split import split_train_validation
split = split_train_validation(discover_train_samples("carpet"), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
path = get_model_path("carpet", "model_selection/convae_input256/autoencoder")
assert not path.exists(), "refusing to overwrite"
cfg = TrainingConfig(category="carpet", image_size=(256, 256), batch_size=8, epochs=15, learning_rate=1e-3, seed=42, device="cpu")
res, _ = train_anomaly_model_on_samples(split.training, cfg, path)
print("trained on", res.num_training_images, "images; seconds", res.duration_seconds, "final loss", res.final_loss, "->", path)
