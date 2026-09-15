"""Milestone 4 Phase 3: train an autoencoder on an explicit sample subset,
saving to an explicit path - never the Phase 1 production location.

Mirrors app.ai.training.train.train_anomaly_model exactly: same seeding
(app.ai.training.train.set_seed), same loop structure, same optimizer
(Adam), same loss (MSE), same architecture (ConvAutoencoder via
app.ai.training.model.build_model). The ONLY methodological difference is
what the DataLoader iterates over - here, an explicit, caller-provided
sample list (via GoodImageSubsetDataset) instead of every train/good image
for a category - and where the result is saved. Does not modify, retrain,
or touch the existing Phase 1 production model or its training code.
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.ai.training.artifacts import save_model
from app.ai.training.model import build_model
from app.ai.training.phase3_dataset import GoodImageSubsetDataset
from app.ai.training.schemas import DatasetSample, TrainingConfig, TrainingResult
from app.ai.training.train import set_seed


def train_anomaly_model_on_samples(
    samples: list[DatasetSample], config: TrainingConfig, model_path: Path
) -> tuple[TrainingResult, nn.Module]:
    """Train a fresh ConvAutoencoder on exactly `samples`, save it to `model_path`.

    Returns (TrainingResult, the trained model in eval() mode) so the caller
    can immediately compute validation reconstruction errors without a
    reload-from-disk round trip.
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    dataset = GoodImageSubsetDataset(samples, config.image_size)
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

    saved_path = save_model(model, model_path)
    model.eval()

    result = TrainingResult(
        category=config.category,
        device=str(device),
        epochs=config.epochs,
        final_loss=loss_history[-1] if loss_history else float("nan"),
        duration_seconds=duration_seconds,
        num_training_images=len(dataset),
        model_path=saved_path,
        loss_history=loss_history,
    )
    return result, model
