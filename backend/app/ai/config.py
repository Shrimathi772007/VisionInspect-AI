"""Shared configuration for the Milestone 2 AI package.

Reuses the existing dataset/storage roots from app.inspections.storage rather
than redefining them, so there is a single source of truth for these paths.
"""

import os

import torch

from app.inspections.storage import DATASET_ROOT, STORAGE_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How many per-category models (memory banks) app.ai.inference.serving keeps in memory at once. Optional:
# unset, empty or invalid values mean the default. The frozen backbones are shared and not counted.
MODEL_CACHE_SIZE_ENV = "VISIONINSPECT_AI_MODEL_CACHE_SIZE"
DEFAULT_MODEL_CACHE_SIZE = 3
MAX_MODEL_CACHE_SIZE = 15


def model_cache_size() -> int:
    """VISIONINSPECT_AI_MODEL_CACHE_SIZE as an int in [1, 15]; DEFAULT_MODEL_CACHE_SIZE when unset or invalid."""
    raw = os.environ.get(MODEL_CACHE_SIZE_ENV, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MODEL_CACHE_SIZE
    return value if 1 <= value <= MAX_MODEL_CACHE_SIZE else DEFAULT_MODEL_CACHE_SIZE


__all__ = ["DATASET_ROOT", "STORAGE_ROOT", "DEVICE", "MODEL_CACHE_SIZE_ENV", "DEFAULT_MODEL_CACHE_SIZE", "model_cache_size"]
