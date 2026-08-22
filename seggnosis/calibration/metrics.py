from __future__ import annotations

from typing import Tuple

import numpy as np


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE) for per-pixel segmentation predictions.

    Parameters
    ----------
    probs : np.ndarray, shape (N, C) or (C, H, W) or (N, C, H, W)
        Predicted class probabilities. Will be flattened to (num_pixels, C).
    labels : np.ndarray
        Ground-truth class indices, matching shape minus the channel dim.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better; 0 = perfectly calibrated.
    """
    probs = probs.reshape(-1, probs.shape[-3]) if probs.ndim == 4 else probs
    if probs.ndim == 3:  # (C, H, W) -> (H*W, C)
        c = probs.shape[0]
        probs = probs.reshape(c, -1).T
    labels = labels.reshape(-1)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Data for plotting a reliability diagram.

    Returns
    -------
    bin_centers, bin_accuracies, bin_confidences : np.ndarray
        Arrays of length <= n_bins (empty bins are dropped).
    """
    probs = probs.reshape(-1, probs.shape[-3]) if probs.ndim == 4 else probs
    if probs.ndim == 3:
        c = probs.shape[0]
        probs = probs.reshape(c, -1).T
    labels = labels.reshape(-1)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(accuracies[mask].mean())
        confs.append(confidences[mask].mean())
    return np.array(centers), np.array(accs), np.array(confs)
