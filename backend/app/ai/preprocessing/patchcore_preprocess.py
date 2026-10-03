"""Preprocessing for the PatchCore (WideResNet-50-2) detector.

A separate module so the existing pipeline (app.ai.preprocessing.pipeline) and every model that depends on
it stay byte-for-byte unchanged. File decoding reuses that pipeline's validating loader read-only, so a path
is checked (exists, allowed extension, decodable) exactly as for every other model.

Modes (output: float32 CHW tensor, RGB, ImageNet mean/std normalised):
  crop224  resize the SHORTER side to 256 (cv2.INTER_AREA), centre-crop 224x224 (the PatchCore paper setup)
  full256  resize the whole image to 256x256 (cv2.INTER_AREA), no crop - nothing near the border is cut away
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from app.ai.models.resnet18 import IMAGENET_MEAN, IMAGENET_STD
from app.ai.preprocessing.errors import ImageDecodeError
from app.ai.preprocessing.pipeline import _load_image

MODE_CROP224 = "crop224"
MODE_FULL256 = "full256"
MODES = (MODE_CROP224, MODE_FULL256)
INPUT_SIZES = {MODE_CROP224: 224, MODE_FULL256: 256}
RESIZE_SHORTER_SIDE = 256

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def input_size(mode: str) -> int:
    if mode not in INPUT_SIZES:
        raise ValueError(f"Unknown PatchCore preprocessing mode '{mode}'. Expected one of {MODES}.")
    return INPUT_SIZES[mode]


def decode_image(source: Path | str | bytes) -> np.ndarray:
    """BGR uint8 image from a file path (validated by the existing pipeline loader) or encoded bytes."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        image = cv2.imdecode(np.frombuffer(bytes(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError("Bytes could not be decoded as an image.")
        return image
    return _load_image(Path(source))


def resize_and_crop(image_bgr: np.ndarray, mode: str) -> np.ndarray:
    """BGR uint8 -> RGB uint8 at the mode's input size (see module docstring)."""
    size = input_size(mode)
    if mode == MODE_FULL256:
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    else:
        height, width = image_bgr.shape[:2]
        scale = RESIZE_SHORTER_SIDE / min(height, width)
        new_width, new_height = max(size, round(width * scale)), max(size, round(height * scale))
        resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        top, left = (new_height - size) // 2, (new_width - size) // 2
        resized = resized[top : top + size, left : left + size]
    return cv2.cvtColor(np.ascontiguousarray(resized), cv2.COLOR_BGR2RGB)


def normalize(image_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 HWC -> float32 CHW, /255 then ImageNet mean/std."""
    image = (image_rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def preprocess_for_patchcore(source: Path | str | bytes, mode: str) -> torch.Tensor:
    """File path or encoded bytes -> model-ready float32 (3, S, S) tensor for `mode`."""
    return normalize(resize_and_crop(decode_image(source), mode))
