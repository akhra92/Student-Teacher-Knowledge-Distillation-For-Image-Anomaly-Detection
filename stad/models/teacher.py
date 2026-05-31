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


def _vit_block_features(backbone, x: torch.Tensor) -> list[torch.Tensor]:
    """Walk a timm ViT block-by-block, returning one `(B, N, D)` token tensor
    per transformer block. Shared by `TimmTeacher` and `ViTStudent` so the two
    stay aligned by block position."""
    b = backbone
    x = b.patch_embed(x)
    x = b._pos_embed(x)
    if hasattr(b, "patch_drop"):
        x = b.patch_drop(x)
    x = b.norm_pre(x) if hasattr(b, "norm_pre") else x
    outputs: list[torch.Tensor] = []
    for blk in b.blocks:
        x = blk(x)
        outputs.append(x)
    return outputs


class TimmTeacher(nn.Module):
    """Frozen timm ViT backbone returning per-block token features.

    Works for the ViT family (DINOv2, CLIP-ViT, plain ViT): `forward` returns
    one `(B, N, D)` token tensor *per transformer block*, so loss/eval code can
    index by block position — exactly like `Vgg16Teacher` indexes by ReLU
    position. `dynamic_img_size` + `dynamic_img_pad` let it accept arbitrary
    input resolutions (e.g. 32x32 MNIST, 256x256 MVTec) despite the model's
    native size.

    Structural attributes (`embed_dim`, `patch_size`, `num_prefix_tokens`,
    `num_heads`, `depth`) are exposed so a matching `ViTStudent` can be built.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        feature_layers: Sequence[int] | None = None,  # accepted for API symmetry; unused
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("`timm` is required for TimmTeacher. pip install timm.") from e

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            dynamic_img_size=True, dynamic_img_pad=True,
        )
        if len(getattr(self.backbone, "blocks", [])) == 0:
            raise ValueError(
                f"TimmTeacher only supports ViT-family models with `.blocks`; "
                f"'{model_name}' has none."
            )

        ps = self.backbone.patch_embed.patch_size
        self.depth: int = len(self.backbone.blocks)
        self.embed_dim: int = int(self.backbone.embed_dim)
        self.patch_size: int = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
        self.num_prefix_tokens: int = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.num_heads: int = int(
            getattr(self.backbone.blocks[0].attn, "num_heads", max(1, self.embed_dim // 64))
        )

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return _vit_block_features(self.backbone, x)
