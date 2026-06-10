"""Train the student to mimic the teacher on normal-class images.

Improvements vs. the original:
* Validation split (no test-set leakage) drives best-checkpoint selection.
* TensorBoard logging.
* Optional teacher-feature caching: precompute teacher outputs once.
* CLI overrides for common knobs (--epochs, --lr, --batch_size, --device).

Runs on CPU or Apple-Silicon MPS; no GPU-specific paths.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import time
from pathlib import Path

# A few ViT ops (e.g. antialiased bicubic pos-embed resampling used by the
# DINOv2/ViT student) lack an MPS backward kernel; fall back to CPU for those.
# Tiny tensors, negligible cost. Set before importing torch. Override-able.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.tensorboard import SummaryWriter

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
    return cfg


@torch.no_grad()
def _compute_val_loss(student, teacher, loader, criterion, device) -> float:
    """Average distillation loss over the val set (lower = better mimic)."""
    student.eval()
    total, n = 0.0, 0
    for X, _ in loader:
        X = X.to(device, non_blocking=True)
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        loss = criterion(student(X), teacher(X))
        total += loss.item()
        n += 1
    return total / max(1, n)


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

    teacher, student, layer_indices = build_networks(cfg, device=device)

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

    # ---- Resume ---------------------------------------------------------
    # Restores the full training state (student weights, optimizer, epoch,
    # best metric, global step). Old checkpoints without optimizer state
    # still load — only the weights are restored then.
    start_epoch = 0
    best_metric = float("-inf")
    global_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=True)
        if not (isinstance(ckpt, dict) and "student" in ckpt):
            raise ValueError(f"Cannot resume from {resume_path}: no 'student' state found.")
        student.load_state_dict(ckpt["student"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            log.warning("Checkpoint has no optimizer state; optimizer starts fresh.")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        global_step = int(ckpt.get("global_step", 0))
        log.info(
            f"Resumed from {resume_path}: continuing at epoch {start_epoch} "
            f"(best_metric={best_metric:.4f}, global_step={global_step})"
        )
    # ---- Optional teacher-feature cache --------------------------------
    cached_train: list[list[torch.Tensor]] | None = None
    if bool(train_cfg.get("cache_teacher", False)):
        # build_loaders disables train shuffling in this mode so every epoch
        # replays the exact batch order the cache is built from below.
        log.info("Caching teacher features for the train set (shuffling disabled)...")
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
    # Selection metric: higher is better. When val AUROC is defined (val set
    # contains anomalies) we select on it; otherwise (e.g. MVTec, whose val is
    # normal-only) we fall back to negative val loss so a best checkpoint is
    # still saved.

    for epoch in range(start_epoch, num_epochs + 1):
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

            pred = student(X)
            if cached_train is None:
                with torch.no_grad():
                    target = teacher(X)
            else:
                target = [t.to(device, non_blocking=True) for t in cached]
            loss = criterion(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

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
            val_loss = _compute_val_loss(student, teacher, val_loader, criterion, device)
            log.info(
                f"  [val] auroc={auroc:.4f}  loss={val_loss:.4f}  "
                f"(n_normal={metrics['n_normal']}, n_anom={metrics['n_anomaly']})"
            )
            writer.add_scalar("val/auroc", auroc if auroc == auroc else 0.0, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)

            # AUROC when defined, else -loss. NaN comparisons are False, so a
            # NaN auroc cleanly falls through to the loss-based metric.
            metric = auroc if auroc == auroc else -val_loss
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "global_step": global_step,
                    "val_auroc": auroc,
                    "val_loss": val_loss,
                    # Val-calibrated operating point, used by stad-score to
                    # turn a raw anomaly score into a normal/anomalous verdict.
                    "threshold": metrics["threshold"],
                    "score_min": metrics["score_min"],
                    "score_max": metrics["score_max"],
                }, ckpt_dir / "best.pth")
                tag = f"auroc={auroc:.4f}" if auroc == auroc else f"loss={val_loss:.4f}"
                log.info(
                    f"  → new best ({tag}, threshold={metrics['threshold']:.4f}) "
                    f"saved to {ckpt_dir/'best.pth'}"
                )

        if epoch % ckpt_every == 0 and epoch > 0:
            torch.save({
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_metric": best_metric,
                "global_step": global_step,
            }, ckpt_dir / f"epoch_{epoch}.pth")

    log.info(f"Training done. Best val metric (auroc or -loss): {best_metric:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
