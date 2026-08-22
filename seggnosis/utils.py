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


def as_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure input is (B, C, H, W); add a batch dim if it's (C, H, W)."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


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
