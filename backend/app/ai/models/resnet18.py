"""Frozen ImageNet ResNet-18 feature extractor in pure PyTorch (no torchvision dependency).

The architecture mirrors torchvision's resnet18 exactly (same module names), so the official
ImageNet weights file `resnet18-f37072fd.pth` loads with strict key/shape matching. The weights are a
controlled artifact, never fetched at runtime: they live under backend/ai_models/_pretrained/
(Git-ignored, like every model artifact) and their SHA-256 is verified before every process's first
use. Nothing here is ever fine-tuned - the extractor is used in eval mode under no_grad only.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from app.ai.training import artifacts

RESNET18_FILENAME = "resnet18-f37072fd.pth"
# Official PyTorch checkpoint; the filename's hash prefix (f37072fd) is the start of this digest.
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET18_SOURCE_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PretrainedWeightsError(Exception):
    """The pretrained weights file is missing or is not the expected file."""


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _make_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    layers = [BasicBlock(inplanes, planes, stride, downsample)]
    layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    """ResNet-18 exposing intermediate feature maps: layer1 (64ch, /4), layer2 (128ch, /8), layer3 (256ch, /16)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _make_layer(64, 64, 2, 1)
        self.layer2 = _make_layer(64, 128, 2, 2)
        self.layer3 = _make_layer(128, 256, 2, 2)
        self.layer4 = _make_layer(256, 512, 2, 2)
        self.fc = nn.Linear(512, 1000)  # unused for features; kept so the official file loads strictly

    def forward(self, x: torch.Tensor, layers: tuple[int, ...] = (2, 3)) -> dict[int, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out: dict[int, torch.Tensor] = {}
        x = self.layer1(x)
        if 1 in layers:
            out[1] = x
        x = self.layer2(x)
        if 2 in layers:
            out[2] = x
        x = self.layer3(x)
        if 3 in layers:
            out[3] = x
        return out


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """[0,1] RGB NCHW -> ImageNet-normalized. Applied on top of the project's existing preprocessing output;
    it changes no existing preprocessing, only adapts the range to what the frozen backbone was trained on."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def pretrained_weights_path() -> Path:
    return artifacts.ARTIFACTS_ROOT / "_pretrained" / RESNET18_FILENAME


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=4)
def _verified_sha256(path_str: str, mtime_ns: int, size: int) -> str:
    return file_sha256(Path(path_str))


def load_pretrained_resnet18(path: Path | None = None, expected_sha256: str = RESNET18_SHA256) -> ResNet18:
    """A frozen (eval, requires_grad=False) ResNet-18 loaded from the verified ImageNet weights file."""
    path = Path(path) if path is not None else pretrained_weights_path()
    if not path.is_file():
        raise PretrainedWeightsError(f"Pretrained weights not found: {path.name} (expected under ai_models/_pretrained/).")
    stat = path.stat()
    actual = _verified_sha256(str(path), stat.st_mtime_ns, stat.st_size)
    if actual != expected_sha256:
        raise PretrainedWeightsError(f"Pretrained weights hash mismatch: expected {expected_sha256}, found {actual}.")

    model = ResNet18()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
