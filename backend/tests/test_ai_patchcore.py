"""PatchCore detector (app.ai.models.patchcore) and its preprocessing (app.ai.preprocessing.patchcore_preprocess).

Mechanics tests use an UNTRAINED WideResNet-50-2 (for real shapes) or tiny synthetic feature tensors (for the
scoring maths), so they never need the pretrained weights. The one real-weight smoke test reads a single
train/good image and skips when the weights are absent.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.ai.models import patchcore as pc
from app.ai.models.patchcore import PatchCoreConfig, PatchCoreDetector, PatchCoreModelError
from app.ai.models.resnet18 import IMAGENET_MEAN, IMAGENET_STD
from app.ai.models.wide_resnet50 import WRN50_2_SHA256, WideResNet50_2, wide_resnet50_weights_path
from app.ai.preprocessing import patchcore_preprocess as pp
from app.inspections.storage import DATASET_ROOT

BACKEND_DIR = Path(__file__).resolve().parent.parent
TILE_TRAIN_GOOD_IMAGE = DATASET_ROOT / "tile" / "train" / "good" / "000.png"


class _NoExtractor(torch.nn.Module):
    """Stand-in for detectors that only ever see precomputed features."""

    def forward(self, *args, **kwargs):
        raise AssertionError("the backbone must not be called in this test")


@pytest.fixture(scope="module")
def untrained_wrn():
    torch.manual_seed(0)
    model = WideResNet50_2().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _detector(**config) -> PatchCoreDetector:
    return PatchCoreDetector(PatchCoreConfig(**config), extractor=_NoExtractor())


def _features(n=6, p=64, d=16, seed=0) -> torch.Tensor:
    return torch.randn(n, p, d, generator=torch.Generator().manual_seed(seed))


def _write_png(path: Path, bgr: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), bgr)
    return path


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_crop224_and_full256_shapes_from_non_square_input(tmp_path):
    rng = np.random.default_rng(0)
    path = _write_png(tmp_path / "wide.png", rng.integers(0, 256, (300, 420, 3), dtype=np.uint8))
    crop = pp.preprocess_for_patchcore(path, pp.MODE_CROP224)
    full = pp.preprocess_for_patchcore(path, pp.MODE_FULL256)
    assert tuple(crop.shape) == (3, 224, 224) and crop.dtype == torch.float32
    assert tuple(full.shape) == (3, 256, 256) and full.dtype == torch.float32


def test_channels_are_rgb_and_imagenet_normalised(tmp_path):
    red_bgr = np.zeros((300, 260, 3), dtype=np.uint8)
    red_bgr[..., 2] = 255  # pure red in OpenCV's BGR order
    path = _write_png(tmp_path / "red.png", red_bgr)
    for mode in pp.MODES:
        out = pp.preprocess_for_patchcore(path, mode)
        expected = [(1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
                    (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]]
        for channel, value in enumerate(expected):
            assert torch.allclose(out[channel], torch.full_like(out[channel], value), atol=1e-5)
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406) and IMAGENET_STD == (0.229, 0.224, 0.225)


def test_crop224_is_the_centre_of_the_256_resize(tmp_path):
    # 300x450 with a white band in columns 150-300: resizing the shorter side to 256 gives 256x384, so the
    # band lands on columns 128-256 and the centre crop (columns 80-304) shows it at 48-176, black either side.
    # A few pixels of margin allow for INTER_AREA blending at the edges.
    image = np.zeros((300, 450, 3), dtype=np.uint8)
    image[:, 150:300] = 255
    rgb = pp.resize_and_crop(cv2.imread(str(_write_png(tmp_path / "bands.png", image))), pp.MODE_CROP224)
    assert rgb.shape == (224, 224, 3)
    assert rgb[:, 52:172].min() == 255
    assert rgb[:, :44].max() == 0 and rgb[:, 180:].max() == 0


def test_preprocessing_is_deterministic_and_bytes_match_path(tmp_path):
    rng = np.random.default_rng(1)
    path = _write_png(tmp_path / "img.png", rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    for mode in pp.MODES:
        first, second = pp.preprocess_for_patchcore(path, mode), pp.preprocess_for_patchcore(path, mode)
        assert torch.equal(first, second)
        assert torch.equal(first, pp.preprocess_for_patchcore(path.read_bytes(), mode))


def test_unknown_mode_and_undecodable_bytes_are_rejected():
    with pytest.raises(ValueError):
        pp.input_size("crop999")
    with pytest.raises(pp.ImageDecodeError):
        pp.preprocess_for_patchcore(b"not an image", pp.MODE_CROP224)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_and_validation():
    config = PatchCoreConfig()
    assert config.to_dict() == {
        "backbone": "wide_resnet50_2_imagenet_frozen", "weights_sha256": WRN50_2_SHA256, "layers": [2, 3],
        "neighbourhood": "avg_pool_3x3", "preprocessing": "crop224", "input_size": 224, "coreset_ratio": 0.10,
        "coreset_seed": 0, "projection_dim": 64, "k": 1, "aggregation": "max",
    }
    assert PatchCoreConfig(preprocessing="full256").input_size == 256
    assert PatchCoreConfig.from_dict(config.to_dict()) == config
    for bad in ({"preprocessing": "crop999"}, {"aggregation": "median"}, {"k": 3}, {"coreset_ratio": 0.0},
                {"preprocessing": "full256", "input_size": 224}):
        with pytest.raises(ValueError):
            PatchCoreConfig(**bad)


# ---------------------------------------------------------------------------
# Feature extraction (untrained WRN-50-2, real shapes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, patches", [("crop224", 784), ("full256", 1024)])
def test_extract_features_shapes(untrained_wrn, mode, patches):
    detector = PatchCoreDetector(PatchCoreConfig(preprocessing=mode), extractor=untrained_wrn)
    side = pp.input_size(mode)
    features = detector.extract_features(torch.rand(2, 3, side, side))
    assert tuple(features.shape) == (2, patches, 1536)
    assert detector.config.patches_per_image == patches


def test_extract_features_into_preallocates_and_matches_single_pass(untrained_wrn, monkeypatch):
    detector = PatchCoreDetector(PatchCoreConfig(), extractor=untrained_wrn)
    images = torch.rand(5, 3, 224, 224, generator=torch.Generator().manual_seed(3))
    reference = detector.extract_features(images)

    def _no_stack(*args, **kwargs):
        raise AssertionError("features must not be stacked from per-image lists")

    monkeypatch.setattr(torch, "stack", _no_stack)
    chunked = detector.extract_features_into((images[i] for i in range(5)), count=5, batch_size=2)
    monkeypatch.undo()
    assert tuple(chunked.shape) == (5, 784, 1536)
    assert torch.allclose(chunked, reference, atol=1e-5)
    with pytest.raises(ValueError):
        detector.extract_features_into((images[i] for i in range(3)), count=5)


# ---------------------------------------------------------------------------
# Memory bank and scoring (synthetic features)
# ---------------------------------------------------------------------------

def test_fit_bank_size_and_determinism():
    features = _features()
    total = features.shape[0] * features.shape[1]
    first = _detector().fit_from_features(features)
    second = _detector().fit_from_features(features)
    assert abs(first.shape[0] - math.ceil(0.10 * total)) <= 1
    assert first.shape[1] == features.shape[2]
    assert torch.equal(first, second)
    other_seed = _detector(coreset_seed=1).fit_from_features(features)
    assert not torch.equal(first, other_seed)


def test_patch_scores_equal_brute_force_and_do_not_depend_on_chunk_size():
    detector = _detector()
    detector.fit_from_features(_features(seed=1))
    query = _features(n=3, seed=2)
    brute = torch.cdist(query.reshape(-1, 16), detector.bank).min(dim=1).values.reshape(3, 64)
    by_256 = detector.patch_scores_from_features(query, chunk=256)
    by_1 = detector.patch_scores_from_features(query, chunk=1)
    assert tuple(by_256.shape) == (3, 64)
    assert torch.allclose(by_256, brute, atol=1e-5)
    assert torch.allclose(by_1, by_256, rtol=0, atol=1e-6)
    for bad in (0, 257):
        with pytest.raises(ValueError):
            detector.patch_scores_from_features(query, chunk=bad)


def test_bank_members_score_zero():
    detector = _detector(coreset_ratio=1.0)
    features = _features(n=2)
    detector.fit_from_features(features)
    # torch.cdist's fast matmul path leaves ~1e-3 rounding noise on self-distances (same tolerance as the
    # existing detector's test); distances to other data are ~3, i.e. >> that.
    assert torch.allclose(detector.patch_scores_from_features(features), torch.zeros(2, 64), atol=1e-2)
    assert float(detector.patch_scores_from_features(_features(n=2, seed=5)).mean()) > 1.0


def test_scoring_without_a_bank_is_refused():
    with pytest.raises(PatchCoreModelError):
        _detector().patch_scores_from_features(_features(n=1))


@pytest.mark.parametrize("patches, k", [(784, 8), (1024, 11)])
def test_aggregations(patches, k):
    scores = torch.zeros(2, patches)
    scores[0, :k] = torch.arange(1, k + 1, dtype=torch.float32)  # top-k are 1..k
    scores[0, k:] = 0.5
    scores[1] = 2.0
    detector = _detector()
    assert detector.image_scores(scores, "max").tolist() == [float(k), 2.0]
    top = detector.image_scores(scores, "top1pct_mean")
    assert math.ceil(0.01 * patches) == k
    assert top.tolist() == pytest.approx([(k + 1) / 2, 2.0])
    with pytest.raises(ValueError):
        detector.image_scores(scores, "median")


# ---------------------------------------------------------------------------
# Anomaly maps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("patches, side", [(784, 224), (1024, 256)])
def test_anomaly_map_shape_and_bounds(patches, side):
    scores = torch.rand(2, patches, generator=torch.Generator().manual_seed(4))
    maps = _detector().anomaly_map(scores, side)
    assert tuple(maps.shape) == (2, side, side) and maps.dtype == torch.float32
    assert torch.isfinite(maps).all()
    assert float(maps.max()) <= float(scores.max()) + 1e-6
    assert float(maps.min()) >= float(scores.min()) - 1e-6


def test_hot_patch_peaks_inside_its_footprint():
    grid = torch.zeros(28, 28)
    row, col = 10, 17
    grid[row, col] = 5.0
    maps = _detector().anomaly_map(grid.reshape(1, -1), 224)
    peak = int(maps[0].argmax())
    y, x = divmod(peak, 224)
    assert row * 8 <= y < (row + 1) * 8 and col * 8 <= x < (col + 1) * 8
    assert float(maps.max()) <= 5.0
    assert pc.gaussian_kernel_1d().numel() == 33 and float(pc.gaussian_kernel_1d().sum()) == pytest.approx(1.0)


def test_non_square_patch_count_is_rejected():
    with pytest.raises(ValueError):
        _detector().anomaly_map(torch.zeros(1, 10), 224)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _fitted(tmp_path) -> tuple[PatchCoreDetector, torch.Tensor]:
    detector = _detector(aggregation="top1pct_mean")
    detector.fit_from_features(_features())
    detector.save(tmp_path)
    return detector, _features(n=2, seed=9)


def test_save_load_roundtrip_gives_identical_scores(tmp_path):
    detector, query = _fitted(tmp_path)
    reloaded = PatchCoreDetector.load(tmp_path, extractor=_NoExtractor())
    assert reloaded.config == detector.config
    assert torch.equal(reloaded.bank, detector.bank)
    before = detector.image_scores(detector.patch_scores_from_features(query))
    after = reloaded.image_scores(reloaded.patch_scores_from_features(query))
    assert torch.equal(before, after)
    record = json.loads((tmp_path / "model_config.json").read_text(encoding="utf-8"))
    assert record["weights_sha256"] == WRN50_2_SHA256
    assert record["bank_shape"] == list(detector.bank.shape)
    state = torch.load(tmp_path / "model_state.pt", map_location="cpu", weights_only=True)
    assert torch.equal(state["bank"], detector.bank)


def test_load_refuses_a_different_backbone_weights_hash(tmp_path):
    _fitted(tmp_path)
    config_path = tmp_path / "model_config.json"
    record = json.loads(config_path.read_text(encoding="utf-8"))
    record["weights_sha256"] = "0" * 64
    config_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PatchCoreModelError, match="backbone weights"):
        PatchCoreDetector.load(tmp_path, extractor=_NoExtractor())


def test_load_refuses_an_altered_state_file(tmp_path):
    _fitted(tmp_path)
    torch.save({"bank": torch.zeros(3, 16)}, tmp_path / "model_state.pt")
    with pytest.raises(PatchCoreModelError, match="SHA-256"):
        PatchCoreDetector.load(tmp_path, extractor=_NoExtractor())
    (tmp_path / "model_state.pt").unlink()
    with pytest.raises(PatchCoreModelError, match="Incomplete"):
        PatchCoreDetector.load(tmp_path, extractor=_NoExtractor())


# ---------------------------------------------------------------------------
# Real weights, one train/good image
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (wide_resnet50_weights_path().is_file() and TILE_TRAIN_GOOD_IMAGE.is_file()),
    reason="WideResNet-50-2 weights or the MVTec tile train/good image not present in this environment",
)
def test_real_weights_score_and_map_one_train_good_image():
    detector = PatchCoreDetector(PatchCoreConfig())
    image = pp.preprocess_for_patchcore(TILE_TRAIN_GOOD_IMAGE, pp.MODE_CROP224).unsqueeze(0)
    features = detector.extract_features(image)
    assert tuple(features.shape) == (1, 784, 1536)
    detector.fit_from_features(features)
    flipped = torch.flip(image, dims=[3])
    scores, maps = detector.score_images(flipped)
    assert tuple(scores.shape) == (1,) and torch.isfinite(scores).all() and float(scores[0]) > 0
    assert tuple(maps.shape) == (1, 224, 224) and torch.isfinite(maps).all()


# ---------------------------------------------------------------------------
# Test-set guard
# ---------------------------------------------------------------------------

NEW_SOURCES = [
    BACKEND_DIR / "app" / "ai" / "models" / "wide_resnet50.py",
    BACKEND_DIR / "app" / "ai" / "models" / "patchcore.py",
    BACKEND_DIR / "app" / "ai" / "preprocessing" / "patchcore_preprocess.py",
    BACKEND_DIR / "scripts" / "patchcore_peak_ram_check.py",
]


@pytest.mark.parametrize("path", NEW_SOURCES, ids=lambda p: p.name)
def test_new_modules_never_reference_test_or_ground_truth_images(path):
    source = path.read_text(encoding="utf-8")
    assert "/test/" not in source and "\\test\\" not in source
    assert "ground_truth" not in source
