"""Pixel-level anomaly localization.

Three saliency-map methods are kept for compatibility with the original repo
(`gradients`, `smooth_grad`, `gbp`), but the recommended default is the new
`feature_distance` method: per-spatial-location MSE between student and
teacher feature maps, upsampled to image size and averaged over layers.
This is faster (no backprop), denser, and avoids ReLU-hook hacks that break
on modern PyTorch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader

from .metrics import per_region_overlap, roc_auc


# ---------------------------------------------------------------------------
# Recommended default: feature-distance maps (no backprop required)
# ---------------------------------------------------------------------------

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
# Gradient-based saliency (kept for parity with the original repo)
# ---------------------------------------------------------------------------

def _convert_to_grayscale(im: np.ndarray) -> np.ndarray:
    g = np.sum(np.abs(im), axis=0)
    hi = np.percentile(g, 99)
    lo = g.min()
    return np.clip((g - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _saliency_loss(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    layer_indices: Sequence[int],
    lamda: float,
) -> torch.Tensor:
    pred = student(x)
    real = teacher(x)
    total = x.new_zeros(())
    for k in layer_indices:
        yp, y = pred[k], real[k]
        cos = F.cosine_similarity(yp.flatten(1), y.flatten(1), dim=1).mean()
        mse = ((yp - y) ** 2).flatten(1).mean(dim=1).mean()
        total = total + (1.0 - cos) + lamda * mse
    return total


def gradients_localization(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_indices: Sequence[int],
    lamda: float,
    device: torch.device,
) -> np.ndarray:
    student.eval()
    out: list[np.ndarray] = []
    for X, _ in loader:
        X = X.to(device).requires_grad_(True)
        for i in range(X.shape[0]):
            xi = X[i : i + 1].detach().clone().requires_grad_(True)
            student.zero_grad(set_to_none=True)
            loss = _saliency_loss(student, teacher, xi, layer_indices, lamda)
            loss.backward()
            grad = xi.grad.detach().cpu().squeeze(0).numpy()
            heat = _convert_to_grayscale(grad)
            out.append(gaussian_filter(heat, sigma=4))
    return np.stack(out, axis=0)


def smooth_grad_localization(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_indices: Sequence[int],
    lamda: float,
    device: torch.device,
    n_samples: int = 25,
    sigma_multiplier: float = 0.05,
) -> np.ndarray:
    student.eval()
    out: list[np.ndarray] = []
    for X, _ in loader:
        X = X.to(device)
        for i in range(X.shape[0]):
            xi = X[i : i + 1]
            sigma = sigma_multiplier / max(1e-6, (xi.max() - xi.min()).item())
            agg: np.ndarray | None = None
            for _ in range(n_samples):
                noisy = (xi + torch.randn_like(xi) * (sigma ** 2)).detach().clone().requires_grad_(True)
                student.zero_grad(set_to_none=True)
                loss = _saliency_loss(student, teacher, noisy, layer_indices, lamda)
                loss.backward()
                g = noisy.grad.detach().cpu().squeeze(0).numpy()
                agg = g if agg is None else agg + g
            agg = np.abs(agg / n_samples)
            heat = _convert_to_grayscale(agg)
            out.append(gaussian_filter(heat, sigma=4))
    return np.stack(out, axis=0)


class _GuidedBackprop:
    def __init__(self, student: nn.Module):
        self.student = student
        self.forward_outputs: list[torch.Tensor] = []
        self.hooks: list = []
        student.eval()
        self._register()

    def _register(self) -> None:
        def fwd(_module, _inp, out):
            self.forward_outputs.append(out)

        def bwd(_module, grad_in, grad_out):
            fo = self.forward_outputs.pop()
            mask = (fo > 0).to(grad_in[0].dtype)
            return (mask * torch.clamp(grad_in[0], min=0.0),)

        for m in self.student.modules():
            if isinstance(m, ReLU):
                self.hooks.append(m.register_forward_hook(fwd))
                self.hooks.append(m.register_full_backward_hook(bwd))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()


def gbp_localization(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_indices: Sequence[int],
    lamda: float,
    device: torch.device,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for X, _ in loader:
        X = X.to(device)
        for i in range(X.shape[0]):
            gbp = _GuidedBackprop(deepcopy(student))
            xi = X[i : i + 1].detach().clone().requires_grad_(True)
            gbp.student.zero_grad(set_to_none=True)
            loss = _saliency_loss(gbp.student, teacher, xi, layer_indices, lamda)
            loss.backward()
            grad = xi.grad.detach().cpu().squeeze(0).numpy()
            gbp.remove()
            heat = _convert_to_grayscale(np.abs(grad))
            out.append(gaussian_filter(heat, sigma=4))
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def localization_test(
    student: nn.Module,
    teacher: nn.Module,
    test_loader: DataLoader,
    gt_loader: DataLoader,
    layer_indices: Sequence[int],
    lamda: float,
    device: torch.device,
    method: str,
    img_size: int,
    smooth_grad_n: int = 25,
    smooth_grad_sigma: float = 0.05,
) -> dict[str, float]:
    """Run pixel-level localization and report AUROC + PRO."""
    if method == "feature_dist":
        score_maps = feature_distance_localization(
            student, teacher, test_loader, layer_indices, device, img_size,
        )
    elif method == "gradients":
        score_maps = gradients_localization(student, teacher, test_loader, layer_indices, lamda, device)
    elif method == "smooth_grad":
        score_maps = smooth_grad_localization(
            student, teacher, test_loader, layer_indices, lamda, device,
            n_samples=smooth_grad_n, sigma_multiplier=smooth_grad_sigma,
        )
    elif method == "gbp":
        score_maps = gbp_localization(student, teacher, test_loader, layer_indices, lamda, device)
    else:
        raise ValueError(f"Unknown localization method: {method}")

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
