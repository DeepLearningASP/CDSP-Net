"""Checkpoint loading utilities."""

from typing import Any

import torch
import torch.nn as nn


def load_checkpoint(
    model: nn.Module,
    model_path: str,
) -> nn.Module:
    """Load model parameters from a training checkpoint."""
    checkpoint: Any = torch.load(
        model_path,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must be a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model
