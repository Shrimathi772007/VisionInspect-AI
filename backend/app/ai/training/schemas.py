"""Data structures describing discovered MVTec AD dataset samples."""

from dataclasses import dataclass, field
from pathlib import Path


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
