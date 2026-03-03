"""Convolutional autoencoder for disk image representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Compact convolutional autoencoder with configurable latent size."""

    def __init__(self, latent_dim: int = 64, image_size: int = 512) -> None:
        """Initialize network.

        Args:
            latent_dim: Latent vector dimension.
            image_size: Square image size expected by the network.
        """
        super().__init__()

        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.feature_side = image_size // 8
        flat_dim = 128 * self.feature_side * self.feature_side

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image batch to latent vectors.

        Args:
            x: Tensor of shape `(N, 1, H, W)`.

        Returns:
            Tensor of shape `(N, latent_dim)`.
        """
        x = self.encoder_cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.encoder_fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to reconstructed images.

        Args:
            z: Tensor of shape `(N, latent_dim)`.

        Returns:
            Tensor of shape `(N, 1, H, W)`.
        """
        x = self.decoder_fc(z)
        x = x.view(-1, 128, self.feature_side, self.feature_side)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full encode/decode pass.

        Args:
            x: Tensor of shape `(N, 1, H, W)`.

        Returns:
            Reconstructed image tensor.
        """
        return self.decode(self.encode(x))
