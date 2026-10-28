"""Loss functions used by the DOA estimation models."""

import math
from itertools import permutations

import torch
import torch.nn as nn


class RMSPELoss_Vectorized(nn.Module):
    """Permutation-invariant RMS periodic error."""

    def __init__(self, sources_num: int, reduction: str = "mean"):
        super().__init__()

        if sources_num < 1:
            raise ValueError("sources_num must be a positive integer.")
        if reduction not in {"mean", "batch"}:
            raise ValueError("reduction must be either 'mean' or 'batch'.")

        self.sources_num = sources_num
        self.reduction = reduction

        permutation_indices = torch.tensor(
            list(permutations(range(sources_num))),
            dtype=torch.long,
        )
        self.register_buffer(
            "perm_indices",
            permutation_indices,
        )

    def forward(
        self,
        doa_predictions: torch.Tensor,
        doa: torch.Tensor,
    ) -> torch.Tensor:
        if doa_predictions.shape != doa.shape:
            raise ValueError("doa_predictions and doa must have identical shapes.")
        if doa_predictions.ndim != 2:
            raise ValueError("Expected tensors with shape [batch_size, sources_num].")
        if doa_predictions.shape[1] != self.sources_num:
            raise ValueError("The input source dimension does not match sources_num.")

        batch_size = doa_predictions.shape[0]
        num_permutations = self.perm_indices.shape[0]
        permutation_indices = self.perm_indices.to(doa_predictions.device)

        predictions = doa_predictions.unsqueeze(1).expand(
            batch_size,
            num_permutations,
            self.sources_num,
        )
        indices = permutation_indices.unsqueeze(0).expand(
            batch_size,
            num_permutations,
            self.sources_num,
        )
        permuted_predictions = predictions.gather(
            dim=2,
            index=indices,
        )

        error = permuted_predictions - doa.unsqueeze(1)
        error = (
            torch.remainder(
                error + torch.pi / 2,
                torch.pi,
            )
            - torch.pi / 2
        )

        rmspe = torch.linalg.vector_norm(
            error,
            dim=-1,
        ) / math.sqrt(self.sources_num)
        minimum_rmspe = rmspe.min(dim=1).values

        if self.reduction == "mean":
            return minimum_rmspe.mean()
        return minimum_rmspe


class BCE(nn.Module):
    """Binary cross-entropy loss wrapper."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return self.bce(prediction, label)
