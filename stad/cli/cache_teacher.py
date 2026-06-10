"""Pre-compute teacher features for the train set and dump them to disk.

Useful when training many epochs on a small dataset (e.g. MVTec): the teacher
is frozen, so its forward pass is wasted work every epoch. Run this once,
then set `train.cache_teacher: true` (in-memory) — or load these tensors
directly in a custom pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stad.data import build_loaders
from stad.models import build_networks
from stad.utils import load_config, resolve_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out = Path(args.out) if args.out else (
        Path(cfg["output_dir"]) / cfg["experiment_name"] / "teacher_cache.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    train_loader, _, _ = build_loaders(cfg)
    teacher, _, layer_indices = build_networks(cfg, device=device)
    teacher.eval()

    cache: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for X, _ in tqdm(train_loader, desc="caching"):
            X = X.to(device, non_blocking=True)
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            feats = teacher(X)
            cache.append([f.cpu() for f in feats])

    torch.save({"cache": cache, "layer_indices": layer_indices}, out)
    print(f"saved {len(cache)} batches → {out}")


if __name__ == "__main__":
    main()
