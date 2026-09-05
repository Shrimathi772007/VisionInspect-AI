import pytest

from app.ai.preprocessing import ImageDecodeError, ImageProcessingResult
from app.ai.training import (
    CategoryNotFoundError,
    DatasetRootNotFoundError,
    compute_statistics,
    discover_test_samples,
    discover_train_samples,
    load_sample,
)
from app.ai.training.dataset import get_category_dir
from app.inspections.storage import ALLOWED_EXTENSIONS

CATEGORY = "bottle"


def _real_file_count(directory) -> int:
    """Independent, from-scratch count of real image files, to cross-check the loader's own numbers."""
    if not directory.is_dir():
        return 0
    return sum(
        1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS and f.stat().st_size > 0
    )


# ---------------------------------------------------------------------------
# Bottle category discovery
# ---------------------------------------------------------------------------

def test_bottle_category_resolves_under_dataset_root():
    category_dir = get_category_dir(CATEGORY)
    assert category_dir.name == CATEGORY
    assert category_dir.is_dir()
    assert (category_dir / "train" / "good").is_dir()
    assert (category_dir / "test").is_dir()


def test_train_samples_are_only_from_train_good():
    samples = discover_train_samples(CATEGORY)
    assert len(samples) > 0

    category_dir = get_category_dir(CATEGORY)
    expected_dir = category_dir / "train" / "good"

    for sample in samples:
        assert sample.split == "train"
        assert sample.defect_type == "good"
        assert sample.label == 0
        assert sample.category == CATEGORY
        assert sample.path.parent == expected_dir


def test_train_sample_count_matches_real_directory():
    samples = discover_train_samples(CATEGORY)
    category_dir = get_category_dir(CATEGORY)
    assert len(samples) == _real_file_count(category_dir / "train" / "good")


def test_all_expected_test_defect_categories_discovered():
    samples = discover_test_samples(CATEGORY)
    discovered_defect_types = {s.defect_type for s in samples}

    # These are the actual defect folders that exist for bottle - verified
    # from the real filesystem, not assumed.
    category_dir = get_category_dir(CATEGORY)
    expected_defect_types = {d.name for d in (category_dir / "test").iterdir() if d.is_dir()}

    assert discovered_defect_types == expected_defect_types
    assert {"good", "broken_large", "broken_small", "contamination"} <= discovered_defect_types


def test_labels_are_correct_for_every_test_sample():
    for sample in discover_test_samples(CATEGORY):
        if sample.defect_type == "good":
            assert sample.label == 0
        else:
            assert sample.label == 1


def test_deterministic_ordering():
    first_train = [s.path for s in discover_train_samples(CATEGORY)]
    second_train = [s.path for s in discover_train_samples(CATEGORY)]
    assert first_train == second_train
    assert first_train == sorted(first_train)

    first_test = [s.path for s in discover_test_samples(CATEGORY)]
    second_test = [s.path for s in discover_test_samples(CATEGORY)]
    assert first_test == second_test


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_category_raises():
    with pytest.raises(CategoryNotFoundError):
        discover_train_samples("not_a_real_mvtec_category")

    with pytest.raises(CategoryNotFoundError):
        discover_test_samples("not_a_real_mvtec_category")


def test_missing_dataset_root_raises(monkeypatch, tmp_path):
    missing_root = tmp_path / "does_not_exist"
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", missing_root)

    with pytest.raises(DatasetRootNotFoundError):
        discover_train_samples(CATEGORY)


def test_unreadable_image_raises_via_existing_pipeline(monkeypatch, tmp_path):
    # Build a minimal fake category under a temp "dataset root" to isolate
    # this from the real (always-valid) MVTec files.
    fake_root = tmp_path / "dataset"
    good_dir = fake_root / "widget" / "train" / "good"
    good_dir.mkdir(parents=True)
    corrupted = good_dir / "broken.png"
    corrupted.write_bytes(b"not actually a png file")

    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    samples = discover_train_samples("widget")
    assert len(samples) == 1  # discovery lists it - it's a real, non-empty file

    with pytest.raises(ImageDecodeError):
        load_sample(samples[0])  # loading it decodes it, via the existing pipeline


def test_discovery_ignores_unrelated_and_empty_files(monkeypatch, tmp_path):
    fake_root = tmp_path / "dataset"
    good_dir = fake_root / "widget" / "train" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "readme.txt").write_text("not an image")
    (good_dir / "empty.png").write_bytes(b"")
    (good_dir / "real.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-but-nonzero")

    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_root)

    samples = discover_train_samples("widget")
    assert [s.path.name for s in samples] == ["real.png"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_statistics_match_actual_dataset():
    category_dir = get_category_dir(CATEGORY)
    test_dir = category_dir / "test"

    expected_train_good = _real_file_count(category_dir / "train" / "good")
    expected_per_defect = {
        d.name: _real_file_count(d) for d in test_dir.iterdir() if d.is_dir()
    }
    expected_test_total = sum(expected_per_defect.values())
    expected_test_good = expected_per_defect.get("good", 0)
    expected_test_defective = expected_test_total - expected_test_good

    stats = compute_statistics(CATEGORY)

    assert stats.category == CATEGORY
    assert stats.train_good_count == expected_train_good
    assert stats.test_total_count == expected_test_total
    assert stats.test_good_count == expected_test_good
    assert stats.test_defective_count == expected_test_defective
    assert stats.test_defect_type_counts == expected_per_defect
    assert stats.test_good_count + stats.test_defective_count == stats.test_total_count


# ---------------------------------------------------------------------------
# Integration with the existing preprocessing pipeline
# ---------------------------------------------------------------------------

def test_load_sample_produces_model_ready_result():
    sample = discover_train_samples(CATEGORY)[0]

    result = load_sample(sample, target_size=(128, 128))

    assert isinstance(result, ImageProcessingResult)
    assert result.source_path == sample.path
    assert result.preprocessing.resized_image.shape == (128, 128, 3)
    assert result.preprocessing.normalized_image.shape == (128, 128, 3)
    assert result.preprocessing.color_space == "RGB"
    assert result.quality.width > 0
    assert result.quality.height > 0


def test_load_sample_default_target_size_matches_preprocessing_default():
    from app.ai.preprocessing import DEFAULT_TARGET_SIZE

    sample = discover_test_samples(CATEGORY)[0]
    result = load_sample(sample)

    assert result.preprocessing.resized_image.shape == (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0], 3)
