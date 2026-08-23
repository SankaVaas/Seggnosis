"""Low-level numerical helpers shared across seggnosis methods."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_dropout(model: nn.Module) -> None:
    """Set every Dropout / Dropout2d / Dropout3d layer to train mode, while
    leaving BatchNorm and everything else in eval mode. This is what makes
    MC Dropout work on an already-trained model without retraining."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def as_batch(x: torch.Tensor, spatial_dims: int = 2) -> torch.Tensor:
    """
    Ensure input has an explicit batch dimension.

    2D: (C, H, W) -> (1, C, H, W); (B, C, H, W) passed through.
    3D: (C, D, H, W) -> (1, C, D, H, W); (B, C, D, H, W) passed through.

    Batched and unbatched volumes both have 4 total dims as tensors, which is
    exactly the same rank as an unbatched *2D* image's batched form -- so the
    two cases can't be told apart from shape alone. `spatial_dims` (set once,
    when you call `seggnosis.wrap(..., spatial_dims=3)`) resolves the
    ambiguity explicitly instead of guessing.
    """
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3")
    expected_unbatched = spatial_dims + 1   # (C, *spatial)
    expected_batched = spatial_dims + 2     # (B, C, *spatial)
    if x.dim() == expected_unbatched:
        return x.unsqueeze(0)
    if x.dim() == expected_batched:
        return x
    raise ValueError(
        f"With spatial_dims={spatial_dims}, expected a tensor of rank "
        f"{expected_unbatched} (unbatched) or {expected_batched} (batched), "
        f"got a tensor of rank {x.dim()} with shape {tuple(x.shape)}."
    )


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the channel dimension for (B, C, H, W) logits."""
    return F.softmax(logits, dim=1)


def pixelwise_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """
    Predictive entropy per pixel, from mean class probabilities.

    Parameters
    ----------
    mean_probs : np.ndarray, shape (C, H, W)

    Returns
    -------
    np.ndarray, shape (H, W)
        Entropy in nats. Higher = more uncertain.
    """
    eps = 1e-8
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=0)


def pixelwise_variance(prob_samples: np.ndarray) -> np.ndarray:
    """
    Pixel-wise predictive variance across stochastic forward passes.

    Parameters
    ----------
    prob_samples : np.ndarray, shape (N, C, H, W)
        N stochastic softmax outputs (MC dropout samples, ensemble members,
        or TTA variants).

    Returns
    -------
    np.ndarray, shape (H, W)
        Variance summed across classes, then averaged. Higher = more uncertain.
    """
    return prob_samples.var(axis=0).mean(axis=0)


def mutual_information(prob_samples: np.ndarray) -> np.ndarray:
    """
    Epistemic uncertainty via mutual information (BALD score): the gap
    between total predictive entropy and the average entropy of individual
    samples. High MI = the model's disagreement with itself is what's
    driving the uncertainty (as opposed to inherent ambiguity in the image).

    Parameters
    ----------
    prob_samples : np.ndarray, shape (N, C, H, W)

    Returns
    -------
    np.ndarray, shape (H, W)
    """
    mean_probs = prob_samples.mean(axis=0)
    predictive_entropy = pixelwise_entropy(mean_probs)
    eps = 1e-8
    sample_entropies = -np.sum(prob_samples * np.log(prob_samples + eps), axis=1)
    expected_entropy = sample_entropies.mean(axis=0)
    return predictive_entropy - expected_entropy
