from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, mutual_information, softmax_probs


# Each transform is a (forward, inverse) pair of functions operating on a
# (C, H, W) tensor, so predictions can be mapped back to the original frame.
def _flip_h() -> Tuple[Callable, Callable]:
    f = lambda t: torch.flip(t, dims=[-1])
    return f, f  # flipping twice is the identity


def _flip_v() -> Tuple[Callable, Callable]:
    f = lambda t: torch.flip(t, dims=[-2])
    return f, f


def _rot90(k: int) -> Tuple[Callable, Callable]:
    fwd = lambda t: torch.rot90(t, k=k, dims=[-2, -1])
    inv = lambda t: torch.rot90(t, k=-k, dims=[-2, -1])
    return fwd, inv


DEFAULT_TRANSFORMS = [
    ("identity", (lambda t: t, lambda t: t)),
    ("flip_h", _flip_h()),
    ("flip_v", _flip_v()),
    ("rot90", _rot90(1)),
]


class TTAWrapper(BaseWrapper):
    """
    Test-Time Augmentation uncertainty estimation.

    Applies a small set of label-preserving geometric transforms (flips,
    rotations) to the input, runs the same deterministic model on each
    variant, maps predictions back to the original orientation, and treats
    disagreement across variants as the uncertainty signal.

    Best fit when: you have exactly one trained model and don't want to (or
    can't) touch its architecture -- no dropout required, unlike MC Dropout.

    Parameters
    ----------
    transforms : list of (name, (forward_fn, inverse_fn)), optional
        Defaults to identity + horizontal flip + vertical flip + 90-degree
        rotation. Custom transforms must be spatially invertible.
    uncertainty : str
        "entropy", "variance", or "mutual_information".
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Optional[List] = None,
        uncertainty: str = "variance",
        device: Optional[str] = None,
    ):
        super().__init__(model, device=device)
        self.transforms = transforms or DEFAULT_TRANSFORMS
        if uncertainty not in ("entropy", "variance", "mutual_information"):
            raise ValueError(
                "uncertainty must be 'entropy', 'variance', or 'mutual_information'"
            )
        self.uncertainty_type = uncertainty

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Result:
        self.model.eval()
        x = as_batch(x).to(self.device)
        img = x[0]  # (C, H, W)

        samples = []
        for _name, (fwd, inv) in self.transforms:
            aug = fwd(img).unsqueeze(0)
            logits = self.model(aug)
            probs = softmax_probs(logits)[0]  # (C, H, W), still on device
            probs = inv(probs).cpu().numpy()
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
