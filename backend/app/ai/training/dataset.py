"""MVTec AD dataset discovery for anomaly detection.

    MVTec images -> Dataset loader -> Metadata + labels -> Existing
    preprocessing -> Model-ready samples

Discovery only lists and labels files (fast, no image decoding); actual
decode/resize/normalize/quality validation is delegated to the existing
Phase 2 preprocessing pipeline via `load_sample`, so that logic is not
duplicated here.

Reuses DATASET_ROOT from app.ai.config (itself sourced from
app.inspections.storage) - the single source of truth for the dataset
location - so no path is ever hard-coded here.
"""

from pathlib import Path

from app.ai.config import DATASET_ROOT
from app.ai.preprocessing import DEFAULT_TARGET_SIZE, ImageProcessingResult, process_image
from app.ai.training.errors import CategoryNotFoundError, DatasetRootNotFoundError
from app.ai.training.labels import GOOD_DEFECT_TYPE, label_for_defect_type
from app.ai.training.schemas import DatasetSample, DatasetStatistics
from app.inspections.storage import ALLOWED_EXTENSIONS

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def get_category_dir(category: str) -> Path:
    """Resolve dataset/<category>, validating the dataset root and the category both exist."""
    if not DATASET_ROOT.is_dir():
        raise DatasetRootNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    category_dir = DATASET_ROOT / category
    if not category_dir.is_dir():
        raise CategoryNotFoundError(f"MVTec category not found: {category!r} (looked in {DATASET_ROOT})")

    return category_dir


def _list_image_files(directory: Path) -> list[Path]:
    """Deterministically list readable image files directly inside `directory`.

    "Readable" here means a real, non-empty file with a supported extension -
    unrelated files (readme.txt, ground-truth masks living elsewhere, etc.)
    are safely ignored. Full decodability is validated later, per-sample, by
    the existing preprocessing pipeline (see `load_sample`) rather than
    re-implemented here.
    """
    if not directory.is_dir():
        return []

    files = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in ALLOWED_EXTENSIONS and entry.stat().st_size > 0
    ]
    return sorted(files, key=lambda p: p.name)


def discover_train_samples(category: str) -> list[DatasetSample]:
    """Training data for `category`: ONLY <category>/train/good, labeled good (0)."""
    category_dir = get_category_dir(category)
    good_dir = category_dir / TRAIN_SPLIT / GOOD_DEFECT_TYPE

    return [
        DatasetSample(path=path, category=category, split=TRAIN_SPLIT, defect_type=GOOD_DEFECT_TYPE, label=0)
        for path in _list_image_files(good_dir)
    ]


def discover_test_samples(category: str) -> list[DatasetSample]:
    """Test/evaluation data for `category`: every defect-type folder under test/.

    Any folder other than "good" is treated as defective (label 1) - this is
    generic across MVTec categories, not a hard-coded list of defect names.
    """
    category_dir = get_category_dir(category)
    test_dir = category_dir / TEST_SPLIT
    if not test_dir.is_dir():
        return []

    defect_dirs = sorted((entry for entry in test_dir.iterdir() if entry.is_dir()), key=lambda p: p.name)

    samples = []
    for defect_dir in defect_dirs:
        defect_type = defect_dir.name
        label = label_for_defect_type(defect_type)
        for path in _list_image_files(defect_dir):
            samples.append(
                DatasetSample(path=path, category=category, split=TEST_SPLIT, defect_type=defect_type, label=label)
            )
    return samples


def compute_statistics(category: str) -> DatasetStatistics:
    """Real counts derived from discover_train_samples/discover_test_samples - nothing hard-coded."""
    train_samples = discover_train_samples(category)
    test_samples = discover_test_samples(category)

    test_defect_type_counts: dict[str, int] = {}
    for sample in test_samples:
        test_defect_type_counts[sample.defect_type] = test_defect_type_counts.get(sample.defect_type, 0) + 1

    test_good_count = test_defect_type_counts.get(GOOD_DEFECT_TYPE, 0)
    test_defective_count = len(test_samples) - test_good_count

    return DatasetStatistics(
        category=category,
        train_good_count=len(train_samples),
        test_good_count=test_good_count,
        test_defective_count=test_defective_count,
        test_total_count=len(test_samples),
        test_defect_type_counts=test_defect_type_counts,
    )


def load_sample(sample: DatasetSample, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE) -> ImageProcessingResult:
    """Produce a model-ready sample by running the existing preprocessing pipeline on it.

    This is where real decode/readability validation happens (reusing
    app.ai.preprocessing, not duplicating it) - raises the same
    ImageNotFoundError / UnsupportedImageFormatError / ImageDecodeError as
    the preprocessing pipeline for an unreadable or invalid file.
    """
    return process_image(sample.path, target_size=target_size)
