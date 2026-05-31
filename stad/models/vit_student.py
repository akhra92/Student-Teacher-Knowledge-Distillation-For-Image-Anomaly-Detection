"""ViT student / cloner for the timm (DINOv2 / ViT) teacher.

The VGG-style `Student` mirrors a *CNN* teacher's spatial feature maps. A ViT
teacher instead emits `(B, N, D)` token tensors, so the student must speak the
same language. `ViTStudent` is a from-scratch Vision Transformer whose patch
size, embedding dim and prefix-token count are matched to the teacher — so its
per-block outputs have the *identical* shape and can be compared directly by the
distillation loss — while its capacity is deliberately reduced (smaller MLP
ratio) so it mimics the teacher on normal data but diverges on anomalies.
"""

from __future__ import annotations

import torch
from torch import nn

from .teacher import _vit_block_features


class ViTStudent(nn.Module):
    """Trainable ViT returning one `(B, N, D)` token tensor per block.

    `embed_dim`, `patch_size` and `num_prefix_tokens` must match the teacher so
    the token tensors line up; `depth` need only cover the deepest compared
    layer. Reduced `mlp_ratio` is the capacity bottleneck.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_prefix_tokens: int,
        depth: int,
        num_heads: int,
        img_size: int = 518,
        mlp_ratio: float = 2.0,
        in_chans: int = 3,
    ):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_token=num_prefix_tokens >= 1,
            reg_tokens=max(0, num_prefix_tokens - 1),
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.depth = depth

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return _vit_block_features(self.backbone, x)
