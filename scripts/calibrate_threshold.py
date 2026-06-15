"""Embed a val-calibrated anomaly threshold into existing checkpoint(s).

Older checkpoints were trained before the training loop learned to save an
operating threshold, so `stad-score` / the Streamlit app can't turn a raw score
into a normal/anomalous verdict for them. This recomputes that threshold from a
single validation pass — the exact value retraining would embed — and writes it
back into the checkpoint, leaving the trained weights untouched.

    # one checkpoint
    python scripts/calibrate_threshold.py --checkpoint outputs/mvtec_bottle/checkpoints/best.pth

    # every best.pth under outputs/
    python scripts/calibrate_threshold.py --all
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# A few ViT pos-embed ops lack MPS kernels; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from stad.data import build_loaders
from stad.eval import detection_test
from stad.models import build_networks
from stad.utils import resolve_device, set_seed

ROOT = Path(__file__).resolve().parent.parent


def calibrate(ckpt_path: Path, device_pref: str) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "student" not in ckpt:
        print(f"  ✗ {ckpt_path}: not a trainable checkpoint (missing config/student); skipped")
        return
    cfg = ckpt["config"]
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(device_pref)

    teacher, student, layer_indices = build_networks(cfg, device=device)
    student.load_state_dict(ckpt["student"])
    student.eval()

    # Same val split + metric the training loop calibrates on (no test leakage).
    _, val_loader, _ = build_loaders(cfg)
    metrics = detection_test(
        student=student, teacher=teacher, loader=val_loader,
        layer_indices=layer_indices, lamda=float(cfg["train"]["lamda"]),
        device=device, dataset_name=cfg["data"]["dataset_name"],
        normal_class=cfg["data"]["normal_class"],
        direction_only=bool(cfg["train"].get("direction_loss_only", False)),
    )

    ckpt["threshold"] = metrics["threshold"]
    ckpt["score_min"] = metrics["score_min"]
    ckpt["score_max"] = metrics["score_max"]
    ckpt["val_auroc"] = metrics["auroc"]
    torch.save(ckpt, str(ckpt_path))

    auroc = metrics["auroc"]
    auroc_s = f"{auroc:.4f}" if auroc == auroc else "n/a (normal-only val)"
    print(
        f"  ✓ {ckpt_path}: threshold={metrics['threshold']:.4f} "
        f"(val_auroc={auroc_s}, n_normal={metrics['n_normal']}, n_anom={metrics['n_anomaly']})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, help="Path to a single best.pth")
    p.add_argument("--all", action="store_true", help="Calibrate every outputs/*/checkpoints/best.pth")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.all:
        targets = sorted((ROOT / "outputs").glob("*/checkpoints/best.pth"))
        if not targets:
            raise SystemExit("No checkpoints found under outputs/*/checkpoints/best.pth")
    elif args.checkpoint:
        targets = [Path(args.checkpoint)]
    else:
        raise SystemExit("Pass --checkpoint <path> or --all")

    print(f"Calibrating {len(targets)} checkpoint(s) on device '{args.device}':")
    for t in targets:
        calibrate(t, args.device)


if __name__ == "__main__":
    main()
