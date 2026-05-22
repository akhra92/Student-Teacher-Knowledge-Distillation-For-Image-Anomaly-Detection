"""Image-level anomaly detection: per-image score = student/teacher mismatch."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .metrics import roc_auc

# MVTec ImageFolder assigns "good" the index of its alphabetical position.
_MVTEC_GOOD_IDX: dict[str, int] = {
    "bottle": 3, "cable": 5, "capsule": 2, "carpet": 2, "grid": 3,
    "hazelnut": 2, "leather": 4, "metal_nut": 3, "pill": 5, "screw": 0,
    "tile": 2, "toothbrush": 1, "transistor": 3, "wood": 2, "zipper": 4,
}


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).flatten(1).mean(dim=1)


@torch.no_grad()
def score_batch(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    layer_indices: Sequence[int],
    lamda: float,
    direction_only: bool = False,
) -> torch.Tensor:
    """Returns `(B,)` anomaly scores. Higher = more anomalous."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    pred = student(x)
    real = teacher(x)
    score = x.new_zeros(x.shape[0])
    for k in layer_indices:
        yp, y = pred[k], real[k]
        cos = F.cosine_similarity(_flat(yp), _flat(y), dim=1)
        direction = 1.0 - cos
        if direction_only:
            score = score + direction
        else:
            score = score + direction + lamda * _per_sample_mse(yp, y)
    return score


def detection_test(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_indices: Sequence[int],
    lamda: float,
    device: torch.device,
    dataset_name: str,
    normal_class: int | str,
    direction_only: bool = False,
) -> dict[str, float]:
    """Run detection over a loader. Returns AUROC and label statistics."""
    student.eval()
    teacher.eval()

    if dataset_name == "mvtec":
        if not isinstance(normal_class, str):
            raise ValueError("MVTec normal_class must be a category string.")
        normal_index = _MVTEC_GOOD_IDX[normal_class]
    else:
        normal_index = int(normal_class)

    all_labels: list[int] = []
    all_scores: list[float] = []
    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        s = score_batch(student, teacher, X, layer_indices, lamda, direction_only)
        all_scores.extend(s.detach().cpu().tolist())
        all_labels.extend(Y.tolist())

    labels = np.asarray(all_labels)
    scores = np.asarray(all_scores)
    binary = (labels != normal_index).astype(np.int32)  # 1 = anomaly
    n_normal = int((binary == 0).sum())
    n_anom = int((binary == 1).sum())

    return {
        "auroc": roc_auc(binary, scores, anomaly_label=1),
        "n_normal": n_normal,
        "n_anomaly": n_anom,
        "score_min": float(scores.min()) if scores.size else float("nan"),
        "score_max": float(scores.max()) if scores.size else float("nan"),
    }
