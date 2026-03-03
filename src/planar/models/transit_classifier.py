"""1D CNN classifier for exoplanet transit detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitCNN(nn.Module):
    """1D CNN that outputs binary transit logits."""

    def __init__(self) -> None:
        """Initialize the classifier network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference.

        Args:
            x: Input tensor of shape `(N, 1, T)`.

        Returns:
            Logit tensor of shape `(N, 1)`.
        """
        return self.net(x)

    @staticmethod
    def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities.

        Args:
            logits: Raw model outputs.

        Returns:
            Sigmoid probabilities.
        """
        return torch.sigmoid(logits)
