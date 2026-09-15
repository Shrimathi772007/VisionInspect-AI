"""Real, measured identity of a model artifact and the environment that evaluated it.

Every field here is either read from the artifact file itself (hash, size) or
from an actually-imported library's own version string - nothing is
hard-coded, so this stays correct if dependencies are ever upgraded.
"""

import hashlib
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import sklearn
import torch


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _file_hash(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ModelMetadata:
    """Everything needed to identify exactly which model artifact was evaluated."""

    category: str
    model_name: str
    architecture: str
    latent_channels: int
    artifact_path: str
    artifact_size_bytes: int
    sha256: str
    md5: str
    torch_version: str = field(default_factory=lambda: torch.__version__)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    scikit_learn_version: str = field(default_factory=lambda: sklearn.__version__)
    opencv_version: str = field(default_factory=lambda: cv2.__version__)
    opencv_headless_package_version: str = field(default_factory=lambda: _package_version("opencv-python-headless"))
    device: str = "cpu"


def compute_model_metadata(
    model_path: Path,
    category: str,
    model_name: str = "autoencoder",
    architecture: str = "conv_autoencoder",
    latent_channels: int = 128,
    device: str = "cpu",
) -> ModelMetadata:
    """Read the real, on-disk artifact at `model_path` and record its identity.

    Raises FileNotFoundError if the artifact does not exist - callers should
    not fabricate metadata for a model that isn't actually on disk.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    return ModelMetadata(
        category=category,
        model_name=model_name,
        architecture=architecture,
        latent_channels=latent_channels,
        artifact_path=str(model_path),
        artifact_size_bytes=model_path.stat().st_size,
        sha256=_file_hash(model_path, "sha256"),
        md5=_file_hash(model_path, "md5"),
        device=device,
    )
