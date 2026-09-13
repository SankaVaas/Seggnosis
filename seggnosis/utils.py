"""Low-level numerical helpers shared across seggnosis methods."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_dropout(model: nn.Module) -> None:
    """Set every Dropout / Dropout2d / Dropout3d layer to train mode, while
    leaving BatchNorm and everything else in eval mode. This is what makes
    MC Dropout work on an already-trained model without retraining.

    Callers are responsible for restoring the model's original mode
    afterwards (e.g. via `model.train(was_training)` in a try/finally) --
    this function only flips dropout on, it never flips anything back off.
    """
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


def pixelwise_entropy(mean_probs: np.ndarray, channel_axis: int = 0) -> np.ndarray:
    """
    Predictive entropy per pixel, from mean class probabilities.

    Parameters
    ----------
    mean_probs : np.ndarray, shape (C, *spatial) or (B, C, *spatial)
    channel_axis : int
        Which axis of `mean_probs` is the class-channel axis (0 for
        unbatched (C, *spatial), 1 for batched (B, C, *spatial)).

    Returns
    -------
    np.ndarray, shape (*spatial) or (B, *spatial)
        Entropy in nats. Higher = more uncertain.
    """
    eps = 1e-8
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=channel_axis)


def pixelwise_variance(prob_samples: np.ndarray, channel_axis: int = 1) -> np.ndarray:
    """
    Pixel-wise predictive variance across stochastic forward passes.

    Parameters
    ----------
    prob_samples : np.ndarray, shape (N, C, *spatial) or (N, B, C, *spatial)
        N stochastic softmax outputs (MC dropout samples, ensemble members,
        or TTA variants), optionally with a batch axis after the sample axis.
    channel_axis : int
        Position of the class-channel axis *within `prob_samples`* (1 for
        the unbatched (N, C, *spatial) layout, 2 for the batched
        (N, B, C, *spatial) layout).

    Returns
    -------
    np.ndarray, shape (*spatial) or (B, *spatial)
        Mean variance across classes. Higher = more uncertain. Bounded in
        [0, 0.25], since each class probability is confined to [0, 1] and
        the variance of a [0, 1]-bounded quantity is maximized (at 0.25)
        when it swings between the two extremes.
    """
    variance = prob_samples.var(axis=0)  # removes the sample axis (0)
    return variance.mean(axis=channel_axis - 1)


def mutual_information(prob_samples: np.ndarray, channel_axis: int = 1) -> np.ndarray:
    """
    Epistemic uncertainty via mutual information (BALD score): the gap
    between total predictive entropy and the average entropy of individual
    samples. High MI = the model's disagreement with itself is what's
    driving the uncertainty (as opposed to inherent ambiguity in the image).

    Parameters
    ----------
    prob_samples : np.ndarray, shape (N, C, *spatial) or (N, B, C, *spatial)
    channel_axis : int
        Position of the class-channel axis within `prob_samples` (1 for
        unbatched, 2 for batched -- see `pixelwise_variance`).

    Returns
    -------
    np.ndarray, shape (*spatial) or (B, *spatial)
        Bounded in [0, log(C)], same as predictive entropy.
    """
    mean_probs = prob_samples.mean(axis=0)  # removes the sample axis (0)
    predictive_entropy = pixelwise_entropy(mean_probs, channel_axis=channel_axis - 1)
    eps = 1e-8
    sample_entropies = -np.sum(prob_samples * np.log(prob_samples + eps), axis=channel_axis)
    expected_entropy = sample_entropies.mean(axis=0)
    return predictive_entropy - expected_entropy


def normalization_constant(uncertainty_type: str, n_classes: int) -> float:
    """
    The theoretical maximum value of each uncertainty measure, used to turn
    a raw uncertainty map into a confidence score in [0, 1].

    - "entropy" and "mutual_information" are both bounded above by
      log(n_classes) nats (maximum entropy of a uniform distribution over
      n_classes).
    - "variance" (see `pixelwise_variance`) is bounded above by 0.25,
      independent of n_classes -- normalizing it by log(n_classes) instead
      (as an earlier version of this function did) silently compresses it
      towards 0 and makes `Result.confidence` overstate confidence for
      variance-based uncertainty.
    """
    if uncertainty_type in ("entropy", "mutual_information"):
        return float(np.log(n_classes) + 1e-8)
    if uncertainty_type == "variance":
        return 0.25
    raise ValueError(
        "uncertainty_type must be 'entropy', 'variance', or 'mutual_information'"
    )


def chunked_replicated_batches(
    x: torch.Tensor, n_samples: int, chunk_size: Optional[int] = None
) -> Iterator[Tuple[int, torch.Tensor]]:
    """
    Yield (n, replicated_batch) pairs for running `n_samples` stochastic
    forward passes (MC Dropout, or any other N-samples-of-the-same-input
    method) as a handful of large batched model calls instead of
    `n_samples` sequential single-call passes.

    `x` is a single already-batched input, shape (B, C, *spatial). Each
    yielded `replicated_batch` has shape (n * B, C, *spatial): `n` tiled
    copies of `x`, stacked along the batch axis so one
    `model(replicated_batch)` call produces `n` independent stochastic
    samples at once (independent because dropout masks differ per
    forward-pass "slot" even within the same batched call).

    Parameters
    ----------
    x : torch.Tensor, shape (B, C, *spatial)
    n_samples : int
        Total number of stochastic samples wanted.
    chunk_size : int, optional
        Max number of samples to replicate into a single forward call. If
        None, all `n_samples` are replicated into one call -- fastest, but
        uses `n_samples` times the memory of a single forward pass. Set
        this to trade speed for memory on large volumes or big `n_samples`.

    Yields
    ------
    (n, replicated_batch) : tuple[int, torch.Tensor]
        `n` is how many samples this chunk contributes (sums to
        `n_samples` across all yielded chunks); `replicated_batch` has
        shape (n * B, C, *spatial).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    size = chunk_size if chunk_size is not None else n_samples
    if size < 1:
        raise ValueError("chunk_size must be >= 1")
    remaining = n_samples
    while remaining > 0:
        n = min(size, remaining)
        rep_shape = (n,) + (-1,) * x.dim()
        replicated = x.unsqueeze(0).expand(*rep_shape).reshape(n * x.shape[0], *x.shape[1:])
        yield n, replicated
        remaining -= n
