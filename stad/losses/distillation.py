"""Distillation losses comparing student and teacher feature maps.

Each loss takes two parallel sequences of feature tensors and the indices of
the layers at which to enforce agreement.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _flatten_tokens(t: torch.Tensor) -> torch.Tensor:
    """`(B, ...)` → `(B, -1)` for cosine similarity over arbitrary feature shapes."""
    return t.reshape(t.shape[0], -1)


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error reduced over all non-batch dims → `(B,)`."""
    diff = (pred - target) ** 2
    return diff.flatten(1).mean(dim=1)


class MseDirectionLoss(nn.Module):
    """Cosine direction loss + λ · MSE magnitude loss, summed across selected layers."""

    def __init__(self, lamda: float, layer_indices: Sequence[int]):
        super().__init__()
        self.lamda = float(lamda)
        self.layer_indices = tuple(layer_indices)

    def forward(
        self,
        pred_features: Sequence[torch.Tensor],
        target_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        total = pred_features[0].new_zeros(())
        for k in self.layer_indices:
            yp, y = pred_features[k], target_features[k]
            cos = F.cosine_similarity(_flatten_tokens(yp), _flatten_tokens(y), dim=1)
            direction = (1.0 - cos).mean()
            magnitude = _per_sample_mse(yp, y).mean()
            total = total + direction + self.lamda * magnitude
        return total


class DirectionOnlyLoss(nn.Module):
    def __init__(self, layer_indices: Sequence[int]):
        super().__init__()
        self.layer_indices = tuple(layer_indices)

    def forward(
        self,
        pred_features: Sequence[torch.Tensor],
        target_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        total = pred_features[0].new_zeros(())
        for k in self.layer_indices:
            yp, y = pred_features[k], target_features[k]
            cos = F.cosine_similarity(_flatten_tokens(yp), _flatten_tokens(y), dim=1)
            total = total + (1.0 - cos).mean()
        return total


class MseLoss(nn.Module):
    def __init__(self, lamda: float, layer_indices: Sequence[int]):
        super().__init__()
        self.lamda = float(lamda)
        self.layer_indices = tuple(layer_indices)

    def forward(
        self,
        pred_features: Sequence[torch.Tensor],
        target_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        total = pred_features[0].new_zeros(())
        for k in self.layer_indices:
            total = total + _per_sample_mse(pred_features[k], target_features[k]).mean()
        return self.lamda * total
