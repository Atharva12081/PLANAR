"""Tests for neural model forward passes."""

from __future__ import annotations

import torch

from planar.models.autoencoder import ConvAutoencoder
from planar.models.transit_classifier import TransitCNN


def test_autoencoder_forward_shape() -> None:
    """Autoencoder should reconstruct to the same shape."""
    model = ConvAutoencoder(latent_dim=16, image_size=64)
    batch = torch.rand(4, 1, 64, 64)
    output = model(batch)
    assert output.shape == batch.shape


def test_transit_classifier_output_shape() -> None:
    """Transit classifier should output one logit per sample."""
    model = TransitCNN()
    batch = torch.rand(8, 1, 200)
    logits = model(batch)
    probs = torch.sigmoid(logits)
    assert logits.shape == (8, 1)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
