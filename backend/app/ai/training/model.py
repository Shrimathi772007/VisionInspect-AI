"""Lightweight convolutional autoencoder for CPU-trainable anomaly detection.

Trained ONLY on normal ("good") images to reconstruct them. It has no
classification head and never sees labels during training - it simply
learns what a normal bottle looks like. Anomaly scoring from reconstruction
error is intentionally out of scope for this phase; this module only
defines and exposes the reconstruction network itself.

Fully convolutional (4 stride-2 downsamples, 4 stride-2 upsamples), so it
works with any square input whose side length is a multiple of 16 - no
fixed-size linear layer to keep it lightweight and CPU-friendly.
"""

import torch
from torch import nn

DEFAULT_LATENT_CHANNELS = 128


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = DEFAULT_LATENT_CHANNELS):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output scaled to [0, 1], matching the pipeline's normalized_image range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


def build_model(latent_channels: int = DEFAULT_LATENT_CHANNELS) -> ConvAutoencoder:
    return ConvAutoencoder(latent_channels=latent_channels)
