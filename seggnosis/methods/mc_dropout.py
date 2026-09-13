from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, chunked_replicated_batches, enable_dropout, softmax_probs


class MCDropoutWrapper(BaseWrapper):
    """
    Monte Carlo Dropout uncertainty estimation.

    Works on any model that already has Dropout / Dropout2d layers in it
    (very common in segmentation backbones). No retraining needed: at
    inference time dropout is switched back on, the same image is pushed
    through N times, and the spread across those N stochastic predictions
    becomes the uncertainty signal. The model is returned to whatever mode
    (train/eval) it was in before `.predict()` was called -- including
    restoring dropout to eval mode afterwards -- so wrapping a model and
    calling `.predict()` never leaves it mutated for other code that reuses
    the same model object.

    If the model has no dropout layers, this degrades to N identical
    passes (uncertainty will be ~0 everywhere) -- use `method="tta"` or
    `method="ensemble"` instead in that case.

    Parameters
    ----------
    n_samples : int
        Number of stochastic forward passes. 20-30 is a common default.
    uncertainty : str
        "entropy" (predictive entropy of the mean prediction),
        "variance" (mean pixel-wise variance across samples), or
        "mutual_information" (BALD score; isolates epistemic uncertainty
        specifically caused by model disagreement, not image ambiguity).
    spatial_dims : int
        2 for images (C, H, W), 3 for volumes (C, D, H, W), e.g. CT/MRI.
    chunk_size : int, optional
        By default, all `n_samples` stochastic passes are run as ONE
        batched forward call (the input replicated `n_samples` times along
        the batch axis) instead of `n_samples` sequential single-pass
        calls -- generally much faster on a GPU. This uses roughly
        `n_samples` times the memory of a single forward pass, so for
        large volumes or large `n_samples` set `chunk_size` to cap how many
        samples are replicated into any one call (e.g. `chunk_size=5` runs
        `ceil(n_samples / 5)` batched calls of 5 samples each). Set to `1`
        to fully recover the original one-sample-per-call behavior.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 20,
        uncertainty: str = "entropy",
        device: Optional[str] = None,
        spatial_dims: int = 2,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(model, device=device, spatial_dims=spatial_dims)
        self.n_samples = n_samples
        if uncertainty not in ("entropy", "variance", "mutual_information"):
            raise ValueError(
                "uncertainty must be 'entropy', 'variance', or 'mutual_information'"
            )
        self.uncertainty_type = uncertainty
        self.chunk_size = chunk_size

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Result:
        x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)
        batch_size = x.shape[0]

        was_training = self.model.training
        self.model.eval()           # BatchNorm etc. stay in eval mode...
        enable_dropout(self.model)  # ...only dropout layers go back to train mode
        try:
            sample_chunks = []
            for n, replicated in chunked_replicated_batches(x, self.n_samples, self.chunk_size):
                logits = self.model(replicated)  # (n * B, C, *spatial)
                probs = softmax_probs(logits).cpu().numpy()
                sample_chunks.append(probs.reshape(n, batch_size, *probs.shape[1:]))
        finally:
            # Always restore the model's original mode, even if the forward
            # pass raised -- callers should never see their model left with
            # dropout stuck on.
            self.model.train(was_training)

        samples = np.concatenate(sample_chunks, axis=0)  # (N, B, C, *spatial)
        mean_probs = samples.mean(axis=0)  # (B, C, *spatial)

        if self.uncertainty_type == "entropy":
            from ..utils import pixelwise_entropy
            umap = pixelwise_entropy(mean_probs, channel_axis=1)
        elif self.uncertainty_type == "variance":
            from ..utils import pixelwise_variance
            umap = pixelwise_variance(samples, channel_axis=2)
        else:
            from ..utils import mutual_information
            umap = mutual_information(samples, channel_axis=2)

        return self._finalize(
            x, mean_probs, umap, self.uncertainty_type, raw={"samples": samples}
        )
