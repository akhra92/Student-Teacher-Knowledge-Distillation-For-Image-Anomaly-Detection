from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .student import Student
from .teacher import TimmTeacher, Vgg16Teacher
from .vit_student import ViTStudent


def build_networks(
    cfg: dict[str, Any],
    device: torch.device,
    load_checkpoint: str | Path | None = None,
) -> tuple[nn.Module, nn.Module, list[int]]:
    """Build (teacher, student, feature_layer_indices).

    `feature_layer_indices` are positions into each network's returned feature
    list — used by losses and detection so they stay in lockstep.

    * `vgg16` teacher → VGG-style `Student` (per-ReLU spatial maps).
    * `timm` ViT teacher → `ViTStudent` whose token shape matches the teacher
      at every block, with reduced capacity for the anomaly bottleneck.
    """
    t_cfg = cfg["teacher"]
    s_cfg = cfg["student"]

    if t_cfg["kind"] == "vgg16":
        teacher: nn.Module = Vgg16Teacher(pretrained=bool(t_cfg.get("pretrained", True)))
        student: nn.Module = Student(
            equal_size=bool(s_cfg.get("equal_size", False)),
            use_bias=bool(s_cfg.get("use_bias", False)),
            batch_norm=True,
        )
        feature_layers = list(t_cfg.get("feature_layers", [3, 6, 9, 12]))

    elif t_cfg["kind"] == "timm":
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
        student = ViTStudent(
            embed_dim=teacher.embed_dim,
            patch_size=teacher.patch_size,
            num_prefix_tokens=teacher.num_prefix_tokens,
            depth=max(feature_layers) + 1,
            num_heads=teacher.num_heads,
            mlp_ratio=float(s_cfg.get("mlp_ratio", 2.0)),
        )

    else:
        raise ValueError(f"Unknown teacher.kind: {t_cfg['kind']}")

    teacher.to(device).eval()
    student.to(device)

    if load_checkpoint is not None:
        ckpt = torch.load(str(load_checkpoint), map_location=device)
        state = ckpt["student"] if isinstance(ckpt, dict) and "student" in ckpt else ckpt
        student.load_state_dict(state)

    return teacher, student, feature_layers
