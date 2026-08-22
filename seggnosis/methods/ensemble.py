from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, mutual_information, softmax_probs


class EnsembleWrapper(BaseWrapper):
    """
    Deep ensemble uncertainty estimation.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        models: Optional[List[torch.nn.Module]] = None,
        uncertainty: str = "mutual_information",
        device: Optional[str] = None,
    ):
        # `model` kept as the first positional arg for API symmetry with
        # wrap(model, method="ensemble", models=[m2, m3, ...])
        all_models = [model] + list(models or [])
        if len(all_models) < 2:
            raise ValueError(
                "EnsembleWrapper needs at least 2 models. Pass the rest via "
                "models=[m2, m3, ...] in seggnosis.wrap(model, method='ensemble', models=[...])"
            )
        super().__init__(all_models[0], device=device)
        self.models = [m.to(self.device).eval() for m in all_models]
        if uncertainty not in ("entropy", "variance", "mutual_information"):
            raise ValueError(
                "uncertainty must be 'entropy', 'variance', or 'mutual_information'"
            )
        self.uncertainty_type = uncertainty

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Result:
        x = as_batch(x).to(self.device)
        samples = []
        for m in self.models:
            logits = m(x)
            probs = softmax_probs(logits)[0].cpu().numpy()
            samples.append(probs)
        samples = np.stack(samples, axis=0)  # (N, C, H, W)

        mean_probs = samples.mean(axis=0)

        if self.uncertainty_type == "entropy":
            from ..utils import pixelwise_entropy
            umap = pixelwise_entropy(mean_probs)
        elif self.uncertainty_type == "variance":
            from ..utils import pixelwise_variance
            umap = pixelwise_variance(samples)
        else:
            umap = mutual_information(samples)

        return self._finalize(x, mean_probs, umap, raw={"samples": samples})
