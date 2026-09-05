"""MVTec AD dataset discovery and anomaly-detection model training.

    train/good images -> GoodImageDataset -> ConvAutoencoder -> training loop
    -> saved model artifact -> load_model_for_category (for later inference)

Trains a convolutional autoencoder to reconstruct NORMAL images only - no
labels are used for training. Ground-truth labels (good=0/defective=1) exist
here solely to build the held-out, unseen test set for future evaluation.

Public API:
    discover_train_samples(category) -> list[DatasetSample]
    discover_test_samples(category)  -> list[DatasetSample]
    compute_statistics(category)     -> DatasetStatistics
    load_sample(sample, target_size) -> ImageProcessingResult

    build_model(latent_channels)     -> ConvAutoencoder
    train_anomaly_model(config)      -> TrainingResult
    save_model(model, path) / load_model(path, model)
    load_model_for_category(category) -> ConvAutoencoder
"""

from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path, load_model, load_model_for_category, save_model
from app.ai.training.dataset import (
    compute_statistics,
    discover_test_samples,
    discover_train_samples,
    get_category_dir,
    load_sample,
)
from app.ai.training.errors import CategoryNotFoundError, DatasetRootNotFoundError, TrainingDataError
from app.ai.training.labels import DEFECTIVE_LABEL, GOOD_DEFECT_TYPE, GOOD_LABEL, label_for_defect_type
from app.ai.training.model import ConvAutoencoder, build_model
from app.ai.training.schemas import DatasetSample, DatasetStatistics, ModelConfig, TrainingConfig, TrainingResult
from app.ai.training.torch_dataset import GoodImageDataset
from app.ai.training.train import set_seed, train_anomaly_model

__all__ = [
    "discover_train_samples",
    "discover_test_samples",
    "compute_statistics",
    "load_sample",
    "get_category_dir",
    "DatasetSample",
    "DatasetStatistics",
    "GOOD_LABEL",
    "DEFECTIVE_LABEL",
    "GOOD_DEFECT_TYPE",
    "label_for_defect_type",
    "TrainingDataError",
    "DatasetRootNotFoundError",
    "CategoryNotFoundError",
    "ConvAutoencoder",
    "build_model",
    "GoodImageDataset",
    "ModelConfig",
    "TrainingConfig",
    "TrainingResult",
    "train_anomaly_model",
    "set_seed",
    "ARTIFACTS_ROOT",
    "get_model_path",
    "save_model",
    "load_model",
    "load_model_for_category",
]
