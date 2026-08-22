from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, enable_dropout, mutual_information, softmax_probs


class MCDropoutWrapper(BaseWrapper):
    """
    Monte Carlo Dropout uncertainty estimation.

    Works on any model that already has Dropout / Dropout2d layers in it
    (very common in segmentation backbones). No retraining needed: at
    inference time dropout is switched back on, the same image is pushed
    through N times, and the spread across those N stochastic predictions
    becomes the uncertainty signal.

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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 20,
        uncertainty: str = "entropy",
        device: Optional[str] = None,
    ):
        super().__init__(model, device=device)
        self.n_samples = n_samples
        if uncertainty not in ("entropy", "variance", "mutual_information"):
            raise ValueError(
                "uncertainty must be 'entropy', 'variance', or 'mutual_information'"
            )
        self.uncertainty_type = uncertainty

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Result:
        self.model.eval()          # BatchNorm etc. stay in eval mode...
        enable_dropout(self.model)  # ...only dropout layers go back to train mode

        x = as_batch(x).to(self.device)
        samples = []
        for _ in range(self.n_samples):
            logits = self.model(x)
            probs = softmax_probs(logits)[0].cpu().numpy()  # (C, H, W)
            samples.append(probs)
        samples = np.stack(samples, axis=0)  # (N, C, H, W)

        mean_probs = samples.mean(axis=0)  # (C, H, W)

        if self.uncertainty_type == "entropy":
            from ..utils import pixelwise_entropy
            umap = pixelwise_entropy(mean_probs)
        elif self.uncertainty_type == "variance":
            from ..utils import pixelwise_variance
            umap = pixelwise_variance(samples)
        else:
            umap = mutual_information(samples)

        return self._finalize(x, mean_probs, umap, raw={"samples": samples})
