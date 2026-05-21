"""Teacher networks. Two flavors:

* `Vgg16Teacher` — modern torchvision API; emits one feature tensor per ReLU
  activation (13 total) so loss code can compare student/teacher outputs
  by *position* without depending on raw module indices.
* `TimmTeacher` — wraps any timm model (ViT/DINOv2/EfficientNet/...). Returns
  per-block token features for ViTs or stage features for CNNs.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class Vgg16Teacher(nn.Module):
    """Frozen VGG16 returning a list of intermediate feature maps (one per ReLU)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16(weights=weights)
        self.features = nn.ModuleList(list(backbone.features))
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                outputs.append(x)
        return outputs


class TimmTeacher(nn.Module):
    """Frozen timm backbone returning per-block features.

    Works for ViT-family (DINOv2, CLIP-ViT, plain ViT) — features come back as
    a list of `(B, N, D)` token tensors. Also handles CNN backbones via timm's
    `features_only=True` mode, returning `(B, C, H, W)` maps.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        feature_layers: Sequence[int] | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("`timm` is required for TimmTeacher. pip install timm.") from e

        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self._mode = "vit"
            n_blocks = len(getattr(self.backbone, "blocks", []))
            if n_blocks == 0:
                raise AttributeError
            self._n_blocks = n_blocks
        except (AttributeError, RuntimeError):
            self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            self._mode = "cnn"
            self._n_blocks = len(self.backbone.feature_info)

        self._feature_layers = (
            tuple(feature_layers) if feature_layers is not None else tuple(range(self._n_blocks))
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.eval()

    @property
    def output_indices(self) -> tuple[int, ...]:
        return self._feature_layers

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._mode == "cnn":
            feats = self.backbone(x)
            return [feats[i] for i in self._feature_layers if i < len(feats)]

        b = self.backbone
        x = b.patch_embed(x)
        if hasattr(b, "_pos_embed"):
            x = b._pos_embed(x)
        else:
            if getattr(b, "cls_token", None) is not None:
                cls = b.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            x = x + b.pos_embed
        x = b.norm_pre(x) if hasattr(b, "norm_pre") else x

        outputs: list[torch.Tensor] = []
        last_needed = max(self._feature_layers)
        for i, blk in enumerate(b.blocks):
            x = blk(x)
            if i in self._feature_layers:
                outputs.append(x)
            if i >= last_needed:
                break
        return outputs
