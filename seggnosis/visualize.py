"""Plotting helpers. Import matplotlib lazily so seggnosis's core has no hard
dependency on it (useful for headless training/inference servers)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def overlay_uncertainty(
    image: np.ndarray,
    uncertainty_map: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "inferno",
    ax=None,
):
    """
    Overlay a pixel-wise uncertainty heatmap on top of the input image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, 3)
    uncertainty_map : np.ndarray, shape (H, W)
    alpha : float
        Heatmap opacity.
    ax : matplotlib Axes, optional
        Plot into an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    im = ax.imshow(uncertainty_map, cmap=cmap, alpha=alpha)
    ax.set_axis_off()
    ax.set_title("Uncertainty overlay")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_reliability_diagram(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, ax=None
):
    """Plot a reliability diagram (accuracy vs. confidence per bin)."""
    import matplotlib.pyplot as plt

    from .calibration.metrics import reliability_curve, expected_calibration_error

    centers, accs, confs = reliability_curve(probs, labels, n_bins=n_bins)
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.bar(centers, accs, width=1.0 / n_bins, edgecolor="black", alpha=0.7, label="accuracy")
    ax.plot(centers, confs, "ro-", label="avg. confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE={ece:.3f})")
    ax.legend()
    return ax


def plot_mask_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    uncertainty_map: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
):
    """Side-by-side panel: input image, predicted mask, uncertainty map
    (and ground truth if provided)."""
    import matplotlib.pyplot as plt

    n_panels = 3 + (ground_truth is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[0].set_title("Input")
    axes[0].set_axis_off()

    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title("Predicted mask")
    axes[1].set_axis_off()

    im = axes[2].imshow(uncertainty_map, cmap="inferno")
    axes[2].set_title("Uncertainty")
    axes[2].set_axis_off()
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if ground_truth is not None:
        axes[3].imshow(ground_truth, cmap="tab20")
        axes[3].set_title("Ground truth")
        axes[3].set_axis_off()

    fig.tight_layout()
    return fig
