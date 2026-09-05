"""Data structures describing discovered MVTec AD dataset samples, training
configuration, and training results."""

from dataclasses import dataclass, field
from pathlib import Path

from app.ai.config import DEVICE
from app.ai.training.model import DEFAULT_LATENT_CHANNELS


@dataclass(frozen=True)
class DatasetSample:
    """Metadata for one discovered image - no pixel data, just where it is and what it is."""

    path: Path
    category: str
    split: str  # "train" | "test"
    defect_type: str  # e.g. "good", "broken_large"
    label: int  # 0 = good, 1 = defective (see app.ai.training.labels)


@dataclass
class DatasetStatistics:
    """Real counts computed from the actual dataset - nothing here is hard-coded."""

    category: str
    train_good_count: int
    test_good_count: int
    test_defective_count: int
    test_total_count: int
    test_defect_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model architecture configuration (kept separate so it's easy to log/reproduce)."""

    architecture: str = "conv_autoencoder"
    latent_channels: int = DEFAULT_LATENT_CHANNELS


@dataclass
class TrainingConfig:
    """A complete, reproducible description of one training run."""

    category: str
    image_size: tuple[int, int] = (128, 128)  # (width, height); must be a multiple of 16
    batch_size: int = 8
    epochs: int = 15
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = field(default_factory=lambda: str(DEVICE))
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class TrainingResult:
    """Real, measured outcome of one training run - every field is an actual observed value."""

    category: str
    device: str
    epochs: int
    final_loss: float
    duration_seconds: float
    num_training_images: int
    model_path: Path
    loss_history: list[float] = field(default_factory=list)
