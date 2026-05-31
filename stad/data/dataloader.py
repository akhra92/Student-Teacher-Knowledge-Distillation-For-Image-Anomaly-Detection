"""Data loading with proper train / val / test splits.

The original repo evaluated on the test set during training, leaking signal
into model selection. Here we carve a validation split out of the (filtered)
normal-class training set and keep the test set untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageFolder


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _classification_transform(dataset: str, normalize_imagenet: bool = False) -> transforms.Compose:
    if dataset == "cifar10":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    # mnist / fashionmnist (grayscale). ImageNet-pretrained backbones expect a
    # 3-channel, ImageNet-normalized input — expand to 3 channels (Grayscale is
    # picklable, unlike a Lambda, so it survives num_workers spawn) then normalize.
    steps: list = [transforms.Resize((32, 32))]
    if normalize_imagenet:
        steps += [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    else:
        steps += [transforms.ToTensor()]
    return transforms.Compose(steps)


def _mvtec_transform(img_size: int, normalize_imagenet: bool = False) -> transforms.Compose:
    steps: list = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if normalize_imagenet:
        steps += [transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_val_split(
    dataset: Dataset, val_fraction: float, seed: int
) -> tuple[Dataset, Dataset]:
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_loaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader).

    Validation is a held-out slice of the normal-class training set; it is used
    only to pick the best checkpoint. Test loader spans all classes.
    """
    data_cfg = cfg["data"]
    name = data_cfg["dataset_name"]
    root = Path(data_cfg["data_root"])
    normal_class = data_cfg["normal_class"]
    val_frac = float(data_cfg.get("val_fraction", 0.1))
    num_workers = int(data_cfg.get("num_workers", 4))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg["train"]["batch_size"])
    # ImageNet-pretrained ViT teachers (timm/DINOv2) expect ImageNet-normalized,
    # 3-channel inputs. The VGG16 path keeps its original (unnormalized) pipeline.
    normalize = cfg.get("teacher", {}).get("kind") == "timm"

    if name in {"mnist", "fashionmnist", "cifar10"}:
        tfm = _classification_transform(name, normalize_imagenet=normalize)
        cls = {"mnist": MNIST, "fashionmnist": FashionMNIST, "cifar10": CIFAR10}[name]
        train_root = root / name / "train"
        test_root = root / name / "test"
        train_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)

        full_train = cls(str(train_root), train=True, download=True, transform=tfm)
        test_set = cls(str(test_root), train=False, download=True, transform=tfm)

        # Train only on normal-class images. The validation split mixes held-out
        # normals with a balanced sample of *non-normal* train images (which are
        # never used for training) so val AUROC is well-defined — and the test
        # set stays completely untouched (no leakage).
        targets = np.asarray(full_train.targets)
        normal_idx = np.flatnonzero(targets == int(normal_class))
        anom_pool = np.flatnonzero(targets != int(normal_class))
        rng = np.random.default_rng(seed)

        perm = rng.permutation(len(normal_idx))
        n_val = max(1, int(len(normal_idx) * val_frac))
        val_normal_idx = normal_idx[perm[:n_val]]
        train_idx = normal_idx[perm[n_val:]]

        n_anom_val = min(n_val, len(anom_pool))
        val_anom_idx = anom_pool[rng.permutation(len(anom_pool))[:n_anom_val]]

        train_set = Subset(full_train, train_idx.tolist())
        val_set = ConcatDataset([
            Subset(full_train, val_normal_idx.tolist()),
            Subset(full_train, val_anom_idx.tolist()),
        ])

    elif name == "mvtec":
        if not isinstance(normal_class, str):
            raise ValueError("MVTec normal_class must be a category string (e.g. 'capsule').")
        tfm = _mvtec_transform(int(data_cfg["mvtec_img_size"]), normalize_imagenet=normalize)
        train_dir = root / "mvtec" / normal_class / "train"
        test_dir = root / "mvtec" / normal_class / "test"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(
                f"MVTec dirs missing: expected {train_dir} and {test_dir}. "
                "Download MVTec-AD and place categories under data/mvtec/<category>/."
            )
        full_train = ImageFolder(root=str(train_dir), transform=tfm)
        test_set = ImageFolder(root=str(test_dir), transform=tfm)
        train_set, val_set = _train_val_split(full_train, val_frac, seed)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_localization_loader(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Returns (test_loader, ground_truth_loader) for MVTec localization."""
    data_cfg = cfg["data"]
    if data_cfg["dataset_name"] != "mvtec":
        raise ValueError("Localization is only defined for MVTec.")
    img_size = int(data_cfg["mvtec_img_size"])
    normal_class = data_cfg["normal_class"]
    root = Path(data_cfg["data_root"]) / "mvtec" / normal_class
    test_dir = root / "test"
    gt_dir = root / "ground_truth"

    tfm = _mvtec_transform(img_size)
    test_set = ImageFolder(root=str(test_dir), transform=tfm)
    gt_set = ImageFolder(
        root=str(gt_dir),
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(data_cfg.get("num_workers", 4))
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw)
    gt_loader = DataLoader(gt_set, batch_size=bs, shuffle=False, num_workers=nw)
    return test_loader, gt_loader
