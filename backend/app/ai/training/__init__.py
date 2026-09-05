"""MVTec AD dataset discovery for anomaly detection (data preparation only).

    MVTec images -> Dataset loader -> Metadata + labels -> Existing
    preprocessing -> Model-ready samples

No model training, model architecture, or predictions live here yet.

Public API:
    discover_train_samples(category) -> list[DatasetSample]
    discover_test_samples(category)  -> list[DatasetSample]
    compute_statistics(category)     -> DatasetStatistics
    load_sample(sample, target_size) -> ImageProcessingResult
"""

from app.ai.training.dataset import (
    compute_statistics,
    discover_test_samples,
    discover_train_samples,
    get_category_dir,
    load_sample,
)
from app.ai.training.errors import CategoryNotFoundError, DatasetRootNotFoundError, TrainingDataError
from app.ai.training.labels import DEFECTIVE_LABEL, GOOD_DEFECT_TYPE, GOOD_LABEL, label_for_defect_type
from app.ai.training.schemas import DatasetSample, DatasetStatistics

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
]
