"""Evaluate a trained student on the test set."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ViT pos-embed resampling lacks some MPS kernels; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stad.data import build_loaders, build_localization_loader
from stad.eval import detection_test, localization_test
from stad.models import build_networks
from stad.utils import load_config, resolve_device, set_seed, setup_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint .pth. Defaults to <output_dir>/<exp>/checkpoints/best.pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--localization", action="store_true",
                   help="Force localization mode (overrides config).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.localization:
        cfg["eval"]["localization"] = True

    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    log = setup_logger("test", log_dir=out_dir / "logs")
    device = resolve_device(cfg.get("device", "auto"))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "best.pth"
    log.info(f"Loading checkpoint: {ckpt_path}")
    teacher, student, layer_indices = build_networks(cfg, device=device, load_checkpoint=ckpt_path)
    student.eval()

    if cfg["eval"].get("localization", False):
        test_loader, gt_loader = build_localization_loader(cfg)
        results = localization_test(
            student=student, teacher=teacher,
            test_loader=test_loader, gt_loader=gt_loader,
            layer_indices=layer_indices,
            lamda=float(cfg["train"]["lamda"]),
            device=device,
            method=cfg["eval"].get("localization_method", "feature_dist"),
            img_size=int(cfg["data"]["mvtec_img_size"]),
            smooth_grad_n=int(cfg["eval"].get("smooth_grad_n", 25)),
            smooth_grad_sigma=float(cfg["eval"].get("smooth_grad_sigma", 0.05)),
        )
        log.info(
            f"Localization — pixel AUROC={results['pixel_auroc']:.4f}  "
            f"PRO={results['pro']:.4f}  n={results['n_images']}"
        )
    else:
        _, _, test_loader = build_loaders(cfg)
        results = detection_test(
            student=student, teacher=teacher, loader=test_loader,
            layer_indices=layer_indices,
            lamda=float(cfg["train"]["lamda"]),
            device=device,
            dataset_name=cfg["data"]["dataset_name"],
            normal_class=cfg["data"]["normal_class"],
            direction_only=bool(cfg["train"].get("direction_loss_only", False)),
        )
        log.info(
            f"Detection — AUROC={results['auroc']:.4f}  "
            f"(n_normal={results['n_normal']}, n_anom={results['n_anomaly']})"
        )


if __name__ == "__main__":
    main()
