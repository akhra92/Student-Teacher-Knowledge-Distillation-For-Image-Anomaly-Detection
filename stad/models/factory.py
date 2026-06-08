from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .teacher import TimmTeacher
from .vit_student import ViTStudent


def build_networks(
    cfg: dict[str, Any],
    device: torch.device,
    load_checkpoint: str | Path | None = None,
) -> tuple[nn.Module, nn.Module, list[int]]:
    """Build (teacher, student, feature_layer_indices).

    `feature_layer_indices` are positions into each network's returned feature
    list — used by losses and detection so they stay in lockstep.

    A timm ViT teacher is paired with a `ViTStudent` whose token shape matches
    the teacher at every block, with reduced capacity for the anomaly bottleneck.
    """
    t_cfg = cfg["teacher"]
    s_cfg = cfg["student"]

    if t_cfg["kind"] != "timm":
        raise ValueError(
            f"Unknown teacher.kind: {t_cfg['kind']!r} (only 'timm' is supported)."
        )

    teacher = TimmTeacher(
        model_name=t_cfg["timm_model"],
        pretrained=bool(t_cfg.get("pretrained", True)),
    )
    # Keep only block indices that exist in this backbone (depth-1 max).
    requested = list(t_cfg.get("feature_layers", [2, 5, 8, 11]))
    feature_layers = [i for i in requested if 0 <= i < teacher.depth]
    if not feature_layers:
        raise ValueError(
            f"teacher.feature_layers {requested} has no valid index for a "
            f"{teacher.depth}-block model (valid range 0..{teacher.depth - 1})."
        )
    if feature_layers != requested:
        import warnings
        warnings.warn(
            f"Dropped out-of-range feature_layers; using {feature_layers} "
            f"for a {teacher.depth}-block teacher.",
            stacklevel=2,
        )
    student: nn.Module = ViTStudent(
        embed_dim=teacher.embed_dim,
        patch_size=teacher.patch_size,
        num_prefix_tokens=teacher.num_prefix_tokens,
        depth=max(feature_layers) + 1,
        num_heads=teacher.num_heads,
        mlp_ratio=float(s_cfg.get("mlp_ratio", 2.0)),
    )

    teacher.to(device).eval()
    student.to(device)

    if load_checkpoint is not None:
        ckpt = torch.load(str(load_checkpoint), map_location=device)
        state = ckpt["student"] if isinstance(ckpt, dict) and "student" in ckpt else ckpt
        student.load_state_dict(state)

    return teacher, student, feature_layers
