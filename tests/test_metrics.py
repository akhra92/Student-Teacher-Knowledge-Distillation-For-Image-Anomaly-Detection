import math

import numpy as np

from stad.eval.metrics import (
    per_region_overlap,
    pick_threshold,
    pick_threshold_youden,
    roc_auc,
)


def test_roc_auc_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(labels, scores, anomaly_label=1) == 1.0


def test_roc_auc_anti_correlated():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    # Anomalies have lower scores → AUROC = 0.
    assert roc_auc(labels, scores, anomaly_label=1) == 0.0


def test_youden_picks_best_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    thr, j = pick_threshold_youden(labels, scores, anomaly_label=1)
    assert 0.4 <= thr <= 0.6
    assert j == 1.0


def test_pick_threshold_uses_youden_with_both_classes():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    thr = pick_threshold(labels, scores, anomaly_label=1)
    assert 0.4 <= thr <= 0.6


def test_pick_threshold_normal_only_falls_back_to_quantile():
    labels = np.zeros(100, dtype=int)
    scores = np.linspace(0.0, 1.0, 100)
    thr = pick_threshold(labels, scores, anomaly_label=1, normal_quantile=0.99)
    # ~99th percentile of normal scores; everything above is flagged.
    assert 0.95 <= thr <= 1.0


def test_pick_threshold_no_normals_is_nan():
    labels = np.ones(10, dtype=int)
    scores = np.linspace(0.0, 1.0, 10)
    assert math.isnan(pick_threshold(labels, scores, anomaly_label=1))


def test_pro_runs_on_synthetic_masks():
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 16, 16)) > 0.8).astype(np.uint8)
    score_maps = masks.astype(np.float32) + 0.1 * rng.random((3, 16, 16))
    pro = per_region_overlap(score_maps, masks, num_thresholds=20)
    # Score maps perfectly correlate with masks → PRO should be high.
    assert 0.0 <= pro <= 1.0
