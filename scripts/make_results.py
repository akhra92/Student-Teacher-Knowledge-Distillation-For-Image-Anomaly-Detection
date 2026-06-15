"""Render the README results figures: input · detected anomaly · ground truth.

For each MVTec category below, loads its trained checkpoint, scores a defective
test image, and writes a side-by-side panel (plus one combined grid) to
`assets/results/`. Re-run after retraining to refresh the README visuals.

    python scripts/make_results.py
"""

from __future__ import annotations

import os
from pathlib import Path

# ViT pos-embed resampling lacks some MPS kernels; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from stad.eval import score_batch
from stad.models import build_networks
from stad.utils import resolve_device, set_seed

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "results"
COL_TITLES = ["Input", "Detected anomaly", "Ground truth"]

# model -> (input test image, ground-truth mask, defect name)
JOBS = {
    "mvtec_bottle": (
        "data/mvtec/bottle/test/broken_large/000.png",
        "data/mvtec/bottle/ground_truth/broken_large/000_mask.png",
        "broken_large",
    ),
    "mvtec_cable": (
        "data/mvtec/cable/test/bent_wire/000.png",
        "data/mvtec/cable/ground_truth/bent_wire/000_mask.png",
        "bent_wire",
    ),
    "mvtec_capsule": (
        "data/mvtec/capsule/test/scratch/000.png",
        "data/mvtec/capsule/ground_truth/scratch/000_mask.png",
        "scratch",
    ),
}


def build_transform(cfg: dict) -> transforms.Compose:
    """ImageNet-normalized resize, matching the training/inference pipeline."""
    imagenet = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    size = int(cfg["data"]["mvtec_img_size"])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        imagenet,
    ])


@torch.no_grad()
def compute_heatmap(student, teacher, x, layer_indices, out_size) -> np.ndarray:
    pred = student(x)
    real = teacher(x)
    heat: torch.Tensor | None = None
    for k in layer_indices:
        yp, y = pred[k], real[k]
        if yp.dim() == 4:
            d = ((yp - y) ** 2).mean(dim=1, keepdim=True)
        else:
            d = ((yp - y) ** 2).mean(dim=-1)
            n_tokens = d.shape[1]
            side = int(round((n_tokens - 1) ** 0.5))
            if side * side == n_tokens - 1:
                d = d[:, 1:].reshape(d.shape[0], 1, side, side)
            else:
                side = int(round(n_tokens ** 0.5))
                d = d.reshape(d.shape[0], 1, side, side)
        d = F.interpolate(d, size=(out_size, out_size), mode="bilinear", align_corners=False)
        heat = d if heat is None else heat + d
    heat = heat.squeeze().cpu().numpy()
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)


def overlay_heatmap(base: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    size = heat.shape[0]
    base_rgb = np.asarray(base.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
    colored = matplotlib.colormaps["jet"](heat)[..., :3]
    blended = (1 - alpha) * base_rgb + alpha * colored
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def render(model: str, img_p: str, gt_p: str, defect: str):
    ckpt = torch.load(
        ROOT / "outputs" / model / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=False,
    )
    cfg = ckpt["config"]
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device("cpu")
    teacher, student, layers = build_networks(cfg, device=device)
    student.load_state_dict(ckpt["student"])
    student.eval()

    img = Image.open(ROOT / img_p).convert("RGB")
    x = build_transform(cfg)(img).unsqueeze(0).to(device)
    score = score_batch(
        student, teacher, x, layer_indices=layers, lamda=float(cfg["train"]["lamda"])
    ).item()
    out_size = int(cfg["data"]["mvtec_img_size"])
    overlay = overlay_heatmap(img, compute_heatmap(student, teacher, x, layers, out_size))
    gt = Image.open(ROOT / gt_p).convert("L")
    return img, overlay, gt, score


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    for model, (img_p, gt_p, defect) in JOBS.items():
        img, overlay, gt, score = render(model, img_p, gt_p, defect)
        panels.append((model, defect, img, overlay, gt, score))
        print(f"{model}: score={score:.4f}")

    # Combined grid: one row per category.
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i, (model, defect, img, overlay, gt, score) in enumerate(panels):
        for j, im in enumerate([img, overlay, gt]):
            ax = axes[i][j]
            ax.imshow(im, cmap="gray" if j == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(COL_TITLES[j], fontsize=13)
        axes[i][0].set_ylabel(
            f"{model.replace('mvtec_', '')}\n({defect})\nscore={score:.3f}",
            fontsize=10, rotation=0, ha="right", va="center", labelpad=40,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mvtec_results.png", dpi=110, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'mvtec_results.png'}")

    # Per-category panels.
    for model, defect, img, overlay, gt, score in panels:
        f, a = plt.subplots(1, 3, figsize=(9, 3.2))
        for j, (im, t) in enumerate(zip([img, overlay, gt], COL_TITLES)):
            a[j].imshow(im, cmap="gray" if j == 2 else None)
            a[j].set_title(t, fontsize=12)
            a[j].axis("off")
        f.suptitle(
            f"{model.replace('mvtec_', '')} — {defect} (anomaly score {score:.3f})",
            fontsize=13,
        )
        f.tight_layout()
        f.savefig(OUT_DIR / f"{model}.png", dpi=110, bbox_inches="tight")
        print(f"saved {OUT_DIR / f'{model}.png'}")


if __name__ == "__main__":
    main()
