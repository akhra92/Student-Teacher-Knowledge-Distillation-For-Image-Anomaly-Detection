"""Evaluation metrics: image-level AUROC, pixel-level AUROC, MVTec PRO."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve


def roc_auc(labels: np.ndarray, scores: np.ndarray, anomaly_label: int = 1) -> float:
    """ROC-AUC where higher score = more anomalous.

    `anomaly_label` is whichever integer in `labels` represents anomalies. We
    convert to binary {0=normal, 1=anomaly} then compute AUROC.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    binary = (labels == anomaly_label).astype(np.int32)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return float("nan")
    fpr, tpr, _ = roc_curve(binary, scores)
    return float(auc(fpr, tpr))


def pick_threshold_youden(
    labels: np.ndarray, scores: np.ndarray, anomaly_label: int = 1
) -> tuple[float, float]:
    """Pick threshold that maximizes Youden's J = TPR − FPR. Returns (threshold, J)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    binary = (labels == anomaly_label).astype(np.int32)
    fpr, tpr, thr = roc_curve(binary, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(j[idx])


def pick_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    anomaly_label: int = 1,
    normal_quantile: float = 0.99,
) -> float:
    """Operating threshold for "score >= threshold → anomaly".

    With both classes present, maximizes Youden's J. With only normal samples
    (e.g. an MVTec val split carved from normal-only train data), falls back
    to the `normal_quantile` of the normal scores — anything scoring above
    nearly all known-normal samples is flagged. NaN when there are no normal
    samples to calibrate on.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    binary = (labels == anomaly_label).astype(np.int32)
    n_anom = int(binary.sum())
    n_normal = int(len(binary) - n_anom)
    if n_normal == 0:
        return float("nan")
    if n_anom == 0:
        return float(np.quantile(scores[binary == 0], normal_quantile))
    thr, _ = pick_threshold_youden(binary, scores, anomaly_label=1)
    return float(thr)


def per_region_overlap(
    score_maps: np.ndarray,
    masks: np.ndarray,
    num_thresholds: int = 200,
    fpr_limit: float = 0.3,
) -> float:
    """MVTec PRO: integrate per-region recall against false-positive rate.

    `score_maps` and `masks` are `(N, H, W)`. PRO is the area under the
    (FPR vs. mean per-region-overlap) curve up to `fpr_limit`, normalized.
    """
    try:
        from scipy.ndimage import label as cc_label
    except ImportError as e:
        raise ImportError("PRO requires scipy.") from e

    score_maps = np.asarray(score_maps, dtype=np.float64)
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    if score_maps.shape != masks.shape:
        raise ValueError("score_maps and masks must share shape.")

    smin, smax = score_maps.min(), score_maps.max()
    thresholds = np.linspace(smin, smax, num_thresholds)

    inv_masks = 1 - masks
    inv_pixels = inv_masks.sum()
    if inv_pixels == 0:
        return float("nan")

    # Pre-compute connected components per image.
    components: list[tuple[np.ndarray, list[int]]] = []
    for m in masks:
        comp, n = cc_label(m)
        regions = list(range(1, n + 1))
        components.append((comp, regions))

    pro_curve: list[tuple[float, float]] = []
    for t in thresholds:
        pred = (score_maps >= t).astype(np.uint8)
        fpr = float((pred * inv_masks).sum()) / float(inv_pixels)

        recalls: list[float] = []
        for (comp, regions), p in zip(components, pred):
            for r in regions:
                region = (comp == r)
                area = region.sum()
                if area == 0:
                    continue
                recalls.append(float((p[region]).sum()) / float(area))
        if not recalls:
            continue
        pro_curve.append((fpr, float(np.mean(recalls))))

    pro_curve.sort()
    fpr_arr = np.array([x[0] for x in pro_curve])
    pro_arr = np.array([x[1] for x in pro_curve])
    mask = fpr_arr <= fpr_limit
    if mask.sum() < 2:
        return float("nan")
    area = auc(fpr_arr[mask], pro_arr[mask])
    return float(area / fpr_limit)
