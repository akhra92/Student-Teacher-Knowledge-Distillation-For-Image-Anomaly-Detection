"""Train the student to mimic the teacher on normal-class images.

Improvements vs. the original:
* Validation split (no test-set leakage) drives best-checkpoint selection.
* AMP (mixed precision) for ~2x speedup on modern GPUs.
* TensorBoard logging.
* Optional teacher-feature caching: precompute teacher outputs once.
* CLI overrides for common knobs (--epochs, --lr, --batch_size, --device).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stad.data import build_loaders
from stad.eval import detection_test
from stad.losses import DirectionOnlyLoss, MseDirectionLoss
from stad.models import build_networks
from stad.utils import load_config, resolve_device, set_seed, setup_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.resume is not None:
        cfg["train"]["resume"] = args.resume
    if args.no_amp:
        cfg["train"]["amp"] = False
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _apply_overrides(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    tb_dir = out_dir / "tb"
    for d in (ckpt_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)
    log = setup_logger("train", log_dir=log_dir)
    writer = SummaryWriter(str(tb_dir))

    device = resolve_device(cfg.get("device", "auto"))
    log.info(f"Device: {device}")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    log.info(
        f"Loaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    teacher, student, layer_indices = build_networks(
        cfg, device=device, load_checkpoint=cfg["train"].get("resume"),
    )

    train_cfg = cfg["train"]
    if train_cfg.get("direction_loss_only", False):
        criterion = DirectionOnlyLoss(layer_indices=layer_indices)
    else:
        criterion = MseDirectionLoss(lamda=float(train_cfg["lamda"]), layer_indices=layer_indices)

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- Optional teacher-feature cache --------------------------------
    cached_train: list[list[torch.Tensor]] | None = None
    if bool(train_cfg.get("cache_teacher", False)):
        log.info("Caching teacher features for the train set...")
        cached_train = []
        teacher.eval()
        with torch.no_grad():
            for X, _ in train_loader:
                X = X.to(device, non_blocking=True)
                if X.shape[1] == 1:
                    X = X.repeat(1, 3, 1, 1)
                cached_train.append([f.detach().cpu() for f in teacher(X)])

    # ---- Training loop -------------------------------------------------
    num_epochs = int(train_cfg["num_epochs"])
    log_every = int(train_cfg.get("log_every", 10))
    val_every = int(train_cfg.get("val_every", 1))
    ckpt_every = int(train_cfg.get("ckpt_every", 50))
    best_auroc = -1.0
    global_step = 0

    for epoch in range(num_epochs + 1):
        student.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        iterator = enumerate(train_loader)
        if cached_train is not None:
            iterator = enumerate(zip(train_loader, cached_train))

        for batch_idx, payload in iterator:
            if cached_train is None:
                X, _ = payload
            else:
                (X, _), cached = payload

            X = X.to(device, non_blocking=True)
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)

            with torch.amp.autocast(enabled=use_amp):
                pred = student(X)
                if cached_train is None:
                    with torch.no_grad():
                        target = teacher(X)
                else:
                    target = [t.to(device, non_blocking=True) for t in cached]
                loss = criterion(pred, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if batch_idx % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

        avg_loss = epoch_loss / max(1, n_batches)
        dt = time.time() - t0
        log.info(f"epoch [{epoch}/{num_epochs}]  loss={avg_loss:.4f}  time={dt:.1f}s")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # ---- Validation -----------------------------------------------
        if epoch % val_every == 0:
            metrics = detection_test(
                student=student, teacher=teacher, loader=val_loader,
                layer_indices=layer_indices, lamda=float(train_cfg["lamda"]),
                device=device, dataset_name=cfg["data"]["dataset_name"],
                normal_class=cfg["data"]["normal_class"],
                direction_only=bool(train_cfg.get("direction_loss_only", False)),
            )
            auroc = metrics["auroc"]
            log.info(f"  [val] auroc={auroc:.4f}  (n_normal={metrics['n_normal']}, n_anom={metrics['n_anomaly']})")
            writer.add_scalar("val/auroc", auroc if auroc == auroc else 0.0, epoch)

            if auroc == auroc and auroc > best_auroc:
                best_auroc = auroc
                torch.save({
                    "student": student.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_auroc": auroc,
                }, ckpt_dir / "best.pth")
                log.info(f"  → new best (auroc={auroc:.4f}) saved to {ckpt_dir/'best.pth'}")

        if epoch % ckpt_every == 0 and epoch > 0:
            torch.save({
                "student": student.state_dict(),
                "config": cfg,
                "epoch": epoch,
            }, ckpt_dir / f"epoch_{epoch}.pth")

    log.info(f"Training done. Best val AUROC: {best_auroc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
