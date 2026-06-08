"""Pixel-level anomaly localization.

Heatmaps are built with the `feature_distance` method: per-spatial-location MSE
between student and teacher feature maps, upsampled to image size and averaged
over layers. It is fast (no backprop), dense, and works for the ViT teacher's
token features as well as spatial maps.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.utils.data import DataLoader

from .metrics import per_region_overlap, roc_auc


@torch.no_grad()
def feature_distance_localization(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_indices: Sequence[int],
    device: torch.device,
    out_size: int,
) -> np.ndarray:
    """Returns `(N, H, W)` per-image anomaly heatmaps."""
    student.eval()
    teacher.eval()
    maps: list[torch.Tensor] = []
    for X, _ in loader:
        X = X.to(device, non_blocking=True)
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        pred = student(X)
        real = teacher(X)
        batch_map: torch.Tensor | None = None
        for k in layer_indices:
            yp, y = pred[k], real[k]
            if yp.dim() == 4:  # (B, C, H, W)
                d = ((yp - y) ** 2).mean(dim=1, keepdim=True)
            else:  # (B, N, D) ViT tokens — drop CLS, reshape to grid
                d = ((yp - y) ** 2).mean(dim=-1)
                if d.shape[1] > 1:
                    n_tokens = d.shape[1]
                    side = int(round((n_tokens - 1) ** 0.5))
                    if side * side == n_tokens - 1:
                        d = d[:, 1:]  # drop CLS
                        d = d.reshape(d.shape[0], 1, side, side)
                    else:
                        side = int(round(n_tokens ** 0.5))
                        d = d.reshape(d.shape[0], 1, side, side)
            d = F.interpolate(d, size=(out_size, out_size), mode="bilinear", align_corners=False)
            batch_map = d if batch_map is None else batch_map + d
        maps.append(batch_map.squeeze(1).cpu())
    full = torch.cat(maps, dim=0).numpy()
    # Light spatial smoothing.
    for i in range(full.shape[0]):
        full[i] = gaussian_filter(full[i], sigma=4)
    return full


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def localization_test(
    student: nn.Module,
    teacher: nn.Module,
    test_loader: DataLoader,
    gt_loader: DataLoader,
    layer_indices: Sequence[int],
    device: torch.device,
    method: str,
    img_size: int,
) -> dict[str, float]:
    """Run pixel-level localization and report AUROC + PRO."""
    if method != "feature_dist":
        raise ValueError(f"Unknown localization method: {method}")
    score_maps = feature_distance_localization(
        student, teacher, test_loader, layer_indices, device, img_size,
    )

    # Collect ground-truth masks.
    masks: list[np.ndarray] = []
    for gt, _ in gt_loader:
        masks.append(gt.numpy())
    gt_full = np.concatenate(masks, axis=0).squeeze(1)
    gt_bin = (gt_full > 0.5).astype(np.uint8)

    n = min(score_maps.shape[0], gt_bin.shape[0])
    score_maps = score_maps[:n]
    gt_bin = gt_bin[:n]

    pix_auroc = roc_auc(gt_bin.flatten(), score_maps.flatten(), anomaly_label=1)
    pro = per_region_overlap(score_maps, gt_bin)
    return {"pixel_auroc": pix_auroc, "pro": pro, "n_images": int(n)}
