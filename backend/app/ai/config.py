"""Shared configuration for the Milestone 2 AI package.

Reuses the existing dataset/storage roots from app.inspections.storage rather
than redefining them, so there is a single source of truth for these paths.
"""

import torch

from app.inspections.storage import DATASET_ROOT, STORAGE_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["DATASET_ROOT", "STORAGE_ROOT", "DEVICE"]
