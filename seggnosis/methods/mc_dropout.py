from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, enable_dropout, mutual_information, softmax_probs


class MCDropoutWrapper(BaseWrapper):
    """
    Monte Carlo Dropout uncertainty estimation.
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
