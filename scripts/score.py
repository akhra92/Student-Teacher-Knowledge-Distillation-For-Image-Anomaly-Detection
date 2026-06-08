"""Single-image inference: print anomaly score and optionally save heatmap."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ViT pos-embed resampling lacks some MPS kernels; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stad.eval import score_batch
from stad.models import build_networks
from stad.utils import load_config, resolve_device, set_seed


def _build_transform(cfg: dict) -> transforms.Compose:
    name = cfg["data"]["dataset_name"]
    # Match the training pipeline: ImageNet normalization for the timm/ViT teacher.
    imagenet = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if name == "cifar10":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet,
        ])
    if name == "mvtec":
        size = int(cfg["data"]["mvtec_img_size"])
    else:  # mnist / fashionmnist: ViT teacher needs a real resolution
        size = int(cfg["data"].get("vit_img_size", 224))
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), imagenet])


def _save_heatmap(student, teacher, x: torch.Tensor, layer_indices, out_size: int, path: Path) -> None:
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
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    Image.fromarray((heat * 255).astype(np.uint8)).save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--heatmap", type=str, default=None, help="Optional path to save heatmap PNG")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))

    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(cfg["output_dir"]) / cfg["experiment_name"] / "checkpoints" / "best.pth"
    )
    teacher, student, layer_indices = build_networks(cfg, device=device, load_checkpoint=ckpt)
    student.eval()

    tfm = _build_transform(cfg)
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = score_batch(
            student, teacher, x,
            layer_indices=layer_indices,
            lamda=float(cfg["train"]["lamda"]),
            direction_only=bool(cfg["train"].get("direction_loss_only", False)),
        ).item()
    print(f"anomaly_score: {score:.6f}")

    if args.heatmap:
        out_size = (
            int(cfg["data"]["mvtec_img_size"])
            if cfg["data"]["dataset_name"] == "mvtec"
            else x.shape[-1]
        )
        _save_heatmap(student, teacher, x, layer_indices, out_size, Path(args.heatmap))
        print(f"heatmap saved to {args.heatmap}")


if __name__ == "__main__":
    main()