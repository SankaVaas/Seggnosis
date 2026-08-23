from __future__ import annotations

from typing import Tuple

import numpy as np


def _flatten_probs_and_labels(
    probs: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize any of the accepted probability/label shape pairs down to a
    flat (num_pixels, C) / (num_pixels,) pair. Accepts, for both 2D images
    and 3D volumes:
        probs (N, C)              labels (N,)               -- pre-flattened
        probs (C, *spatial)       labels (*spatial)          -- single sample
        probs (B, C, *spatial)    labels (B, *spatial)       -- batch
    where *spatial is (H, W) for images or (D, H, W) for volumes.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim == 2 and labels.ndim == 1:
        return probs, labels

    if probs.ndim != labels.ndim + 1:
        raise ValueError(
            f"Can't infer the channel axis from probs.shape={probs.shape} and "
            f"labels.shape={labels.shape}. Expected probs to have exactly one "
            "more dimension than labels (the class channel)."
        )

    # Batched: (B, C, *spatial) vs labels (B, *spatial) -- same leading dim.
    if probs.ndim > 2 and probs.shape[0] == labels.shape[0]:
        channel_axis = 1
    else:
        # Unbatched: (C, *spatial) vs labels (*spatial).
        channel_axis = 0

    moved = np.moveaxis(probs, channel_axis, -1)  # channel last
    flat_probs = moved.reshape(-1, moved.shape[-1])
    flat_labels = labels.reshape(-1)
    return flat_probs, flat_labels


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE) for per-pixel/per-voxel segmentation
    predictions.

    Parameters
    ----------
    probs : np.ndarray
        Predicted class probabilities. Accepts (N, C), (C, H, W),
        (B, C, H, W) for images, or (C, D, H, W), (B, C, D, H, W) for
        volumes -- the channel axis is inferred from how many more
        dimensions `probs` has than `labels`.
    labels : np.ndarray
        Ground-truth class indices, matching `probs`'s shape minus the
        channel dim.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better; 0 = perfectly calibrated.
    """
    probs, labels = _flatten_probs_and_labels(probs, labels)

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
    probs, labels = _flatten_probs_and_labels(probs, labels)

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
