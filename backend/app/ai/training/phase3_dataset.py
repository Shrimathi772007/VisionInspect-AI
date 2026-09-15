"""Milestone 4 Phase 3: a training dataset over an EXPLICIT sample list.

app.ai.training.torch_dataset.GoodImageDataset always rediscovers every
train/good image for a category (all 209, for bottle) - that is exactly
right for reproducing the Phase 1 production model, so it is left
unmodified. Phase 3 needs the opposite: a dataset that trains on only the
`training` subset of a TrainValidationSplit
(app.ai.training.validation_split), so that the `validation` subset never
reaches the DataLoader and never contributes a gradient.
"""

import torch
from torch.utils.data import Dataset

from app.ai.training.dataset import load_sample
from app.ai.training.schemas import DatasetSample


class GoodImageSubsetDataset(Dataset):
    """Yields CHW float32 tensors (values in [0, 1]) for exactly the samples given -
    no rediscovery, no filtering, nothing beyond what the caller passed in."""

    def __init__(self, samples: list[DatasetSample], image_size: tuple[int, int]):
        self.samples = list(samples)
        self.image_size = image_size
        if not self.samples:
            raise ValueError("No samples provided for GoodImageSubsetDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples[index]
        result = load_sample(sample, target_size=self.image_size)
        return torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()
