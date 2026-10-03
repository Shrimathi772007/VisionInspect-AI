"""WideResNet-50-2 backbone (app.ai.models.wide_resnet50): structure, verified weights, golden fingerprint.

Structure tests use an untrained network. Tests that need the real (Git-ignored) ImageNet weights skip when
the file is absent, like the ResNet-18 tests in test_ai_anomaly_models.py.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from app.ai.models.resnet18 import PretrainedWeightsError, file_sha256
from app.ai.models.wide_resnet50 import (
    WRN50_2_FILENAME,
    WRN50_2_SHA256,
    WRN50_2_SIZE_BYTES,
    WideResNet50_2,
    load_pretrained_wide_resnet50_2,
    wide_resnet50_weights_path,
)

REAL_WEIGHTS = wide_resnet50_weights_path()
requires_weights = pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="pretrained WideResNet-50-2 weights not present")
SCRATCH_IMPL = Path(os.environ.get("TEMP", "")) / "patchcore_preflight2" / "wrn50_2.py"

# Golden fingerprint, recorded once from the VERIFIED scratch implementation (%TEMP%/patchcore_preflight2/wrn50_2.py)
# with the real weights: model built and loaded first, then torch.manual_seed(1234), then x = torch.rand(1, 3, 224, 224).
# That scratch implementation matched torchvision's wide_resnet50_2 (same weights, same input) with max abs
# difference 0.0 on layer2, layer3 and the logits.
GOLDEN = {
    2: {
        "shape": (1, 512, 28, 28),
        "mean": 0.07859237492084503,
        "std": 0.1202363595366478,
        "abs_sum": 31547.609375,
        "first8": [0.0216473788022995, 0.21408087015151978, 0.33878812193870544, 0.2081160694360733,
                   0.34046125411987305, 0.28648924827575684, 0.3179744482040405, 0.10248641669750214],
    },
    3: {
        "shape": (1, 1024, 14, 14),
        "mean": 0.04657962545752525,
        "std": 0.08824794739484787,
        "abs_sum": 9348.716796875,
        "first8": [0.018359169363975525, 0.1283632218837738, 0.07870346307754517, 0.09450925141572952,
                   0.08486273139715195, 0.08187215775251389, 0.08003853261470795, 0.09471806138753891],
    },
}


def _expected_torchvision_keys() -> set[str]:
    """The key set of torchvision's wide_resnet50_2 state dict, generated from its naming scheme."""
    bn = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    keys = {"conv1.weight", "fc.weight", "fc.bias"} | {f"bn1.{p}" for p in bn}
    for layer, blocks in enumerate((3, 4, 6, 3), start=1):
        for block in range(blocks):
            prefix = f"layer{layer}.{block}"
            for i in (1, 2, 3):
                keys.add(f"{prefix}.conv{i}.weight")
                keys |= {f"{prefix}.bn{i}.{p}" for p in bn}
            if block == 0:
                keys.add(f"{prefix}.downsample.0.weight")
                keys |= {f"{prefix}.downsample.1.{p}" for p in bn}
    return keys


@pytest.fixture(scope="module")
def untrained():
    torch.manual_seed(0)
    return WideResNet50_2().eval()


@pytest.fixture(scope="module")
def pretrained():
    if not REAL_WEIGHTS.is_file():
        pytest.skip("pretrained WideResNet-50-2 weights not present")
    return load_pretrained_wide_resnet50_2()


def _fingerprint_input(model) -> torch.Tensor:
    # Same order as when GOLDEN was recorded: the model already exists, then seed, then input.
    torch.manual_seed(1234)
    return torch.rand(1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_stride_is_on_conv2_of_each_downsampling_block(untrained):
    for name in ("layer2", "layer3", "layer4"):
        first = getattr(untrained, name)[0]
        assert first.conv1.stride == (1, 1), name
        assert first.conv2.stride == (2, 2), name
        assert first.downsample[0].stride == (2, 2), name
    first = untrained.layer1[0]
    assert first.conv1.stride == first.conv2.stride == first.downsample[0].stride == (1, 1)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        for block in list(getattr(untrained, name))[1:]:
            assert block.conv1.stride == block.conv2.stride == (1, 1)
            assert block.downsample is None


def test_widths_and_downsample_shapes(untrained):
    block = untrained.layer1[0]
    assert tuple(block.conv1.weight.shape) == (128, 64, 1, 1)
    assert tuple(block.conv2.weight.shape) == (128, 128, 3, 3)
    assert tuple(block.conv3.weight.shape) == (256, 128, 1, 1)
    assert tuple(untrained.layer2[0].conv2.weight.shape) == (256, 256, 3, 3)
    downsample = {name: tuple(getattr(untrained, name)[0].downsample[0].weight.shape)
                  for name in ("layer1", "layer2", "layer3", "layer4")}
    assert downsample == {"layer1": (256, 64, 1, 1), "layer2": (512, 256, 1, 1),
                          "layer3": (1024, 512, 1, 1), "layer4": (2048, 1024, 1, 1)}
    assert tuple(untrained.fc.weight.shape) == (1000, 2048)
    assert [len(getattr(untrained, f"layer{i}")) for i in (1, 2, 3, 4)] == [3, 4, 6, 3]


def test_state_dict_keys_match_torchvision_layout(untrained):
    expected = _expected_torchvision_keys()
    assert len(expected) == 320
    assert set(untrained.state_dict()) == expected


def test_feature_map_shapes_at_224(untrained):
    with torch.no_grad():
        maps = untrained(torch.rand(1, 3, 224, 224), (2, 3))
        layer2, layer3 = untrained.layer2_layer3(torch.rand(1, 3, 224, 224))
    assert tuple(maps[2].shape) == (1, 512, 28, 28)
    assert tuple(maps[3].shape) == (1, 1024, 14, 14)
    assert set(maps) == {2, 3}
    assert tuple(layer2.shape) == (1, 512, 28, 28) and tuple(layer3.shape) == (1, 1024, 14, 14)


# ---------------------------------------------------------------------------
# Real weights
# ---------------------------------------------------------------------------

@requires_weights
def test_real_weights_hash_size_and_keys():
    assert REAL_WEIGHTS.name == WRN50_2_FILENAME
    assert REAL_WEIGHTS.stat().st_size == WRN50_2_SIZE_BYTES
    assert file_sha256(REAL_WEIGHTS) == WRN50_2_SHA256
    assert WRN50_2_SHA256.startswith("95faca4d")
    state = torch.load(REAL_WEIGHTS, map_location="cpu", weights_only=True)
    assert len(state) == 320
    assert set(state) == _expected_torchvision_keys()


def test_real_weights_load_strictly_frozen_float32_eval(pretrained):
    assert not pretrained.training
    assert all(p.dtype == torch.float32 for p in pretrained.parameters())
    assert not any(p.requires_grad for p in pretrained.parameters())


def test_missing_and_tampered_weights_are_refused(tmp_path):
    with pytest.raises(PretrainedWeightsError, match="not found"):
        load_pretrained_wide_resnet50_2(tmp_path / "nope.pth")
    bad = tmp_path / WRN50_2_FILENAME
    bad.write_bytes(b"not the real weights")
    with pytest.raises(PretrainedWeightsError, match="hash mismatch"):
        load_pretrained_wide_resnet50_2(bad)


@requires_weights
def test_one_flipped_byte_is_refused(tmp_path):
    data = bytearray(REAL_WEIGHTS.read_bytes())
    data[len(data) // 2] ^= 0x01
    tampered = tmp_path / WRN50_2_FILENAME
    tampered.write_bytes(bytes(data))
    with pytest.raises(PretrainedWeightsError, match="hash mismatch"):
        load_pretrained_wide_resnet50_2(tampered)


def test_golden_fingerprint_of_layer2_and_layer3(pretrained):
    x = _fingerprint_input(pretrained)
    with torch.no_grad():
        maps = pretrained(x, (2, 3))
    for layer, golden in GOLDEN.items():
        out = maps[layer]
        assert tuple(out.shape) == golden["shape"]
        assert float(out.mean()) == pytest.approx(golden["mean"], rel=1e-3)
        assert float(out.std()) == pytest.approx(golden["std"], rel=1e-3)
        assert float(out.abs().sum()) == pytest.approx(golden["abs_sum"], rel=1e-3)
        assert out.flatten()[:8].tolist() == pytest.approx(golden["first8"], rel=1e-3, abs=1e-6)


def test_logits_are_finite_and_not_degenerate(pretrained):
    with torch.no_grad():
        logits = pretrained.logits(torch.rand(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 1000)
    assert torch.isfinite(logits).all()
    assert float(logits.std()) > 0.1


@pytest.mark.skipif(not SCRATCH_IMPL.is_file(), reason="verified scratch implementation not present on this machine")
def test_matches_verified_scratch_implementation(pretrained):
    spec = importlib.util.spec_from_file_location("wrn50_2_scratch", SCRATCH_IMPL)
    scratch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scratch)
    reference = scratch.load_wrn50_2()
    x = _fingerprint_input(pretrained)
    with torch.no_grad():
        ours, theirs = pretrained(x, (2, 3)), reference(x, (2, 3))
    for layer in (2, 3):
        assert float((ours[layer] - theirs[layer]).abs().max()) < 1e-5
