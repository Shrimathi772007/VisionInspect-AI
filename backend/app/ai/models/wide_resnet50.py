"""Frozen ImageNet WideResNet-50-2 feature extractor in pure PyTorch (no torchvision dependency).

The architecture mirrors torchvision's wide_resnet50_2 exactly (same module names), so the official
ImageNet weights file `wide_resnet50_2-95faca4d.pth` loads with strict key/shape matching. Like
app.ai.models.resnet18, the weights are a controlled artifact, never fetched at runtime: they live under
backend/ai_models/_pretrained/ (Git-ignored) and their SHA-256 is verified before every process's first use.
Nothing here is ever fine-tuned - the extractor is used in eval mode under no_grad only.

Bottleneck blocks: 1x1 reduce -> 3x3 -> 1x1 expand (x4), inner width = planes * 128/64 (the "wide" part).

Stride placement ("ResNet V1.5", as in torchvision): the first block of layer2/3/4 downsamples in its 3x3
conv2 (stride 2) while its 1x1 conv1 keeps stride 1. The original V1 layout puts the stride on conv1. The
weight shapes are identical in both layouts, so a strict state_dict load cannot tell them apart - only the
outputs differ (moving the stride to conv1 changes layer2 by ~0.95 for the same input). This port was
verified against torchvision's own wide_resnet50_2 with these weights: max abs difference 0.0 on layer2,
layer3 and the logits (see tests/test_ai_wrn50.py for the recorded fingerprint).
"""

from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from app.ai.models.resnet18 import PretrainedWeightsError, file_sha256
from app.ai.training import artifacts

WRN50_2_FILENAME = "wide_resnet50_2-95faca4d.pth"
# Official PyTorch checkpoint (IMAGENET1K_V1); the filename's hash prefix (95faca4d) is the start of this digest.
WRN50_2_SHA256 = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
WRN50_2_SOURCE_URL = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"
WRN50_2_SIZE_BYTES = 138_223_492

WIDTH_PER_GROUP = 128
LAYER_BLOCKS = (3, 4, 6, 3)
EXPANSION = 4


class Bottleneck(nn.Module):
    expansion = EXPANSION

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        width = int(planes * (WIDTH_PER_GROUP / 64.0))
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # V1.5: the stride lives on the 3x3 convolution.
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * EXPANSION, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * EXPANSION)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


def _make_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * EXPANSION:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * EXPANSION, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * EXPANSION),
        )
    layers = [Bottleneck(inplanes, planes, stride, downsample)]
    layers += [Bottleneck(planes * EXPANSION, planes) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class WideResNet50_2(nn.Module):
    """WideResNet-50-2 exposing intermediate feature maps: layer1 (256ch, /4), layer2 (512ch, /8),
    layer3 (1024ch, /16), layer4 (2048ch, /32). avgpool/fc exist only so the checkpoint loads strictly."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, LAYER_BLOCKS[0], 1)
        self.layer2 = _make_layer(256, 128, LAYER_BLOCKS[1], 2)
        self.layer3 = _make_layer(512, 256, LAYER_BLOCKS[2], 2)
        self.layer4 = _make_layer(1024, 512, LAYER_BLOCKS[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * EXPANSION, 1000)

    def forward(self, x: torch.Tensor, layers: tuple[int, ...] = (2, 3)) -> dict[int, torch.Tensor]:
        """ImageNet-normalized NCHW -> {layer index: feature map}; stops after the deepest requested layer."""
        out = {}
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for index, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), start=1):
            if index > max(layers):
                break
            x = layer(x)
            if index in layers:
                out[index] = x
        return out

    def layer2_layer3(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(layer2, layer3) feature maps for an ImageNet-normalized batch."""
        maps = self.forward(x, (2, 3))
        return maps[2], maps[3]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Full ImageNet classifier output - only used to sanity-check the network end to end."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def wide_resnet50_weights_path() -> Path:
    return artifacts.ARTIFACTS_ROOT / "_pretrained" / WRN50_2_FILENAME


@lru_cache(maxsize=4)
def _verified_sha256(path_str: str, mtime_ns: int, size: int) -> str:
    return file_sha256(Path(path_str))


def load_pretrained_wide_resnet50_2(path: Path | None = None, expected_sha256: str = WRN50_2_SHA256) -> WideResNet50_2:
    """A frozen (eval, requires_grad=False, float32) WideResNet-50-2 loaded from the verified ImageNet weights file."""
    path = Path(path) if path is not None else wide_resnet50_weights_path()
    if not path.is_file():
        raise PretrainedWeightsError(f"Pretrained weights not found: {path.name} (expected under ai_models/_pretrained/).")
    stat = path.stat()
    actual = _verified_sha256(str(path), stat.st_mtime_ns, stat.st_size)
    if actual != expected_sha256:
        raise PretrainedWeightsError(f"Pretrained weights hash mismatch: expected {expected_sha256}, found {actual}.")

    model = WideResNet50_2()
    # The checkpoint stores some tensors as float16; load_state_dict copies them into the float32 parameters.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.float()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
