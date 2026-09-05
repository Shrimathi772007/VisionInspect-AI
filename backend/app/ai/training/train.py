"""Training loop for the anomaly-detection autoencoder.

    train/good images -> GoodImageDataset -> DataLoader -> ConvAutoencoder
    -> MSE reconstruction loss -> Adam -> saved model artifact

Trains ONLY on normal images (see torch_dataset.GoodImageDataset, which is
built on discover_train_samples - train/good exclusively). No labels are
used anywhere in this loop.
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.ai.training.artifacts import get_model_path, save_model
from app.ai.training.model import build_model
from app.ai.training.schemas import TrainingConfig, TrainingResult
from app.ai.training.torch_dataset import GoodImageDataset


def set_seed(seed: int) -> None:
    """Seed torch's RNG so weight initialization and DataLoader shuffling are reproducible."""
    torch.manual_seed(seed)


def train_anomaly_model(config: TrainingConfig) -> TrainingResult:
    """Run the full training loop described by `config` and save the resulting model.

    Returns a TrainingResult containing only values actually observed during
    this run (loss, duration, image count) - nothing is invented.
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    dataset = GoodImageDataset(config.category, config.image_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model = build_model(latent_channels=config.model.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    loss_history: list[float] = []
    start = time.perf_counter()

    for _epoch in range(config.epochs):
        epoch_loss_total = 0.0
        batch_count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss_total += loss.item()
            batch_count += 1
        loss_history.append(epoch_loss_total / max(batch_count, 1))

    duration_seconds = time.perf_counter() - start

    model_path = get_model_path(config.category)
    save_model(model, model_path)

    return TrainingResult(
        category=config.category,
        device=str(device),
        epochs=config.epochs,
        final_loss=loss_history[-1] if loss_history else float("nan"),
        duration_seconds=duration_seconds,
        num_training_images=len(dataset),
        model_path=model_path,
        loss_history=loss_history,
    )
