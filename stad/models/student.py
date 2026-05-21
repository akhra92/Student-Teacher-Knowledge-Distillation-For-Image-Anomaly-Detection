"""Student / cloner network.

A small VGG-like CNN that mirrors the teacher's feature-map shapes at the
same layer indices, with deliberately reduced capacity. The capacity
bottleneck is what makes anomalies stand out: the student can mimic the
teacher within the normal distribution but fails on out-of-distribution inputs.

Implementation note: instead of relying on hard-coded module indices (which
shift if batch-norm is added/removed), both the teacher and the student
return one feature tensor *per ReLU activation*. This keeps the two outputs
aligned by position regardless of architectural details.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

# 'A' = full-size mirror of VGG16; 'B' = compressed cloner (paper default).
_CFG: dict[str, list[int | str]] = {
    "A": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "B": [8, 8, "M", 8, 128, "M", 8, 8, 256, "M", 8, 8, 512, "M", 8, 8, 512, "M"],
}


def _make_layers(cfg: Sequence[int | str], use_bias: bool, batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            assert isinstance(v, int)
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=use_bias)
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Student(nn.Module):
    """Student CNN.

    `forward` returns one tensor per ReLU activation (13 total in VGG16).
    Loss/eval code uses positions in this list (e.g. layers 3/6/9/12),
    not raw module indices, so the two networks stay aligned even though
    one has BatchNorm and the other does not.
    """

    def __init__(self, equal_size: bool = False, use_bias: bool = False, batch_norm: bool = True):
        super().__init__()
        cfg_key = "A" if equal_size else "B"
        self.features = _make_layers(_CFG[cfg_key], use_bias=use_bias, batch_norm=batch_norm)
        # Used by gradient-attribution localization (saliency maps).
        self.activation: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._target_layer_relu_idx: int | None = None

    def _activation_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor, target_layer: int | None = None) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        relu_count = 0
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                outputs.append(x)
                if target_layer is not None and target_layer == relu_count:
                    self.activation = x
                    if x.requires_grad:
                        x.register_hook(self._activation_hook)
                relu_count += 1
        return outputs
