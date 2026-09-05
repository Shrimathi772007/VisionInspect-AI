import math

import torch

from app.ai.training import (
    ARTIFACTS_ROOT,
    ConvAutoencoder,
    GoodImageDataset,
    ModelConfig,
    TrainingConfig,
    TrainingResult,
    build_model,
    discover_train_samples,
    get_model_path,
    load_model,
    save_model,
    set_seed,
    train_anomaly_model,
)
from tests.conftest import make_image_bytes

CATEGORY = "bottle"


def _make_fake_category(dataset_root, category, good_count=4, extra_dirs=None):
    """Build a minimal fake MVTec-style category tree under a temp dataset root."""
    good_dir = dataset_root / category / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(good_count):
        (good_dir / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=(i * 10, 50, 100)))

    for relative_dir, count in (extra_dirs or {}).items():
        d = dataset_root / category / relative_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (d / f"{i:03d}.png").write_bytes(make_image_bytes("PNG", size=(32, 32), color=(200, i * 10, 10)))


# ---------------------------------------------------------------------------
# Model construction / forward pass
# ---------------------------------------------------------------------------

def test_build_model_returns_conv_autoencoder():
    model = build_model()
    assert isinstance(model, ConvAutoencoder)
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")


def test_forward_pass_shapes_match_input():
    model = build_model()
    for size in (32, 64, 96):
        for batch_size in (1, 3):
            x = torch.randn(batch_size, 3, size, size)
            y = model(x)
            assert y.shape == x.shape


def test_forward_output_is_bounded_zero_to_one():
    model = build_model()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_training_config_defaults_are_sane():
    config = TrainingConfig(category=CATEGORY)
    assert config.category == CATEGORY
    assert config.batch_size > 0
    assert config.epochs > 0
    assert config.learning_rate > 0
    assert config.seed is not None
    assert isinstance(config.device, str)
    assert isinstance(config.model, ModelConfig)
    width, height = config.image_size
    assert width % 16 == 0 and height % 16 == 0, "image_size must be a multiple of 16 for this architecture"


def test_training_config_is_overridable():
    config = TrainingConfig(category=CATEGORY, batch_size=4, epochs=2, learning_rate=5e-4, seed=7)
    assert config.batch_size == 4
    assert config.epochs == 2
    assert config.learning_rate == 5e-4
    assert config.seed == 7


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def test_seed_gives_deterministic_model_initialization():
    set_seed(123)
    model_a = build_model()
    set_seed(123)
    model_b = build_model()

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)


def test_different_seeds_give_different_initialization():
    set_seed(1)
    model_a = build_model()
    set_seed(2)
    model_b = build_model()

    params_equal = all(
        torch.equal(param_a, param_b) for param_a, param_b in zip(model_a.parameters(), model_b.parameters())
    )
    assert not params_equal


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_model_save_and_load_roundtrip(tmp_path):
    model = build_model()
    path = save_model(model, tmp_path / "nested" / "autoencoder.pt")
    assert path.is_file()

    fresh_model = build_model()
    loaded = load_model(path, fresh_model)

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model.eval()(x)
        loaded_output = loaded(x)
    torch.testing.assert_close(original_output, loaded_output)


def test_load_model_missing_file_raises(tmp_path):
    model = build_model()
    try:
        load_model(tmp_path / "does_not_exist.pt", model)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Artifact path handling
# ---------------------------------------------------------------------------

def test_artifacts_root_is_outside_source_and_dataset():
    assert ARTIFACTS_ROOT.name == "ai_models"
    assert "dataset" not in ARTIFACTS_ROOT.parts
    assert "app" not in ARTIFACTS_ROOT.parts  # not inside the app/ source package


def test_get_model_path_is_scoped_by_category():
    bottle_path = get_model_path("bottle")
    cable_path = get_model_path("cable")
    assert bottle_path != cable_path
    assert bottle_path.parent.name == "bottle"
    assert bottle_path.suffix == ".pt"
    assert bottle_path.is_relative_to(ARTIFACTS_ROOT)


# ---------------------------------------------------------------------------
# Training-data isolation (reused/re-verified at the training-dataset level)
# ---------------------------------------------------------------------------

def test_good_image_dataset_only_contains_train_good_samples():
    dataset = GoodImageDataset(CATEGORY, (64, 64))
    assert len(dataset.samples) == len(discover_train_samples(CATEGORY))
    for sample in dataset.samples:
        assert sample.split == "train"
        assert sample.defect_type == "good"
        assert "test" not in sample.path.parts


def test_good_image_dataset_yields_expected_tensor_shape():
    dataset = GoodImageDataset(CATEGORY, (64, 64))
    tensor = dataset[0]
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


# ---------------------------------------------------------------------------
# Lightweight smoke test of the full training loop (NOT the real 209-image run)
# ---------------------------------------------------------------------------

def test_training_loop_smoke_test_on_tiny_synthetic_dataset(tmp_path, monkeypatch):
    fake_dataset_root = tmp_path / "dataset"
    _make_fake_category(
        fake_dataset_root,
        "widget",
        good_count=4,
        extra_dirs={"train/bad": 2, "test/good": 2, "test/broken": 2},
    )
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_dataset_root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")

    config = TrainingConfig(category="widget", image_size=(32, 32), batch_size=2, epochs=1, seed=0)
    result = train_anomaly_model(config)

    assert isinstance(result, TrainingResult)
    # Only the 4 train/good images were used - not train/bad, not any test/ images.
    assert result.num_training_images == 4
    assert result.epochs == 1
    assert math.isfinite(result.final_loss)
    assert result.final_loss >= 0.0
    assert result.duration_seconds >= 0.0
    assert result.model_path.is_file()
    assert len(result.loss_history) == 1


def test_training_never_touches_defective_or_test_images(tmp_path, monkeypatch):
    fake_dataset_root = tmp_path / "dataset"
    _make_fake_category(
        fake_dataset_root,
        "widget",
        good_count=2,
        extra_dirs={"test/good": 5, "test/broken_large": 5},
    )
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", fake_dataset_root)
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")

    config = TrainingConfig(category="widget", image_size=(32, 32), batch_size=2, epochs=1, seed=0)
    result = train_anomaly_model(config)

    # 2 train/good images exist; the 10 test images (good and defective) must
    # never be counted as training data.
    assert result.num_training_images == 2
