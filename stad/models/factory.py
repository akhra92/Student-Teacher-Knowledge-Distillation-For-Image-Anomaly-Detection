from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .student import Student
from .teacher import TimmTeacher, Vgg16Teacher


def build_networks(
    cfg: dict[str, Any],
    device: torch.device,
    load_checkpoint: str | Path | None = None,
) -> tuple[nn.Module, Student, list[int]]:
    """Build (teacher, student, feature_layer_indices).

    `feature_layer_indices` are positions into each network's returned feature
    list — used by losses and detection so they stay in lockstep.
    """
    t_cfg = cfg["teacher"]
    s_cfg = cfg["student"]

    if t_cfg["kind"] == "vgg16":
        teacher: nn.Module = Vgg16Teacher(pretrained=bool(t_cfg.get("pretrained", True)))
    elif t_cfg["kind"] == "timm":
        teacher = TimmTeacher(
            model_name=t_cfg["timm_model"],
            pretrained=bool(t_cfg.get("pretrained", True)),
            feature_layers=t_cfg.get("feature_layers"),
        )
    else:
        raise ValueError(f"Unknown teacher.kind: {t_cfg['kind']}")

    student = Student(
        equal_size=bool(s_cfg.get("equal_size", False)),
        use_bias=bool(s_cfg.get("use_bias", False)),
        batch_norm=True,
    )

    teacher.to(device).eval()
    student.to(device)

    feature_layers = list(t_cfg.get("feature_layers", [3, 6, 9, 12]))

    if load_checkpoint is not None:
        ckpt = torch.load(str(load_checkpoint), map_location=device)
        state = ckpt["student"] if isinstance(ckpt, dict) and "student" in ckpt else ckpt
        student.load_state_dict(state)

    return teacher, student, feature_layers
