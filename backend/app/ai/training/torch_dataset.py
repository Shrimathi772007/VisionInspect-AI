"""torch.utils.data.Dataset over ONLY the normal ("good") training images.

Wraps the existing Phase 3 discovery (discover_train_samples) and Phase 2
preprocessing (load_sample) - no duplicated file listing, decoding, resizing
or normalization logic lives here.
"""

import torch
from torch.utils.data import Dataset

from app.ai.training.dataset import discover_train_samples, load_sample
from app.ai.training.schemas import DatasetSample


class GoodImageDataset(Dataset):
    """Yields CHW float32 tensors (values in [0, 1]) for every train/good image in `category`."""

    def __init__(self, category: str, image_size: tuple[int, int]):
        self.category = category
        self.image_size = image_size
        self.samples: list[DatasetSample] = discover_train_samples(category)
        if not self.samples:
            raise ValueError(f"No training ('good') images found for category {category!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples[index]
        result = load_sample(sample, target_size=self.image_size)
        # HWC float32 [0,1] -> CHW tensor, matching the model's expected input layout
        return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()
