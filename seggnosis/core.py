"""
Core abstractions for seggnosis: a model-agnostic uncertainty & OOD wrapper
for image segmentation models.

The central idea: you already have a trained segmentation model (any
architecture, any framework wrapper as long as it's a torch.nn.Module that
returns logits of shape (B, C, H, W)). You wrap it once, and every prediction
comes back with a mask AND a trust signal: pixel-wise uncertainty, an overall
confidence score, and (optionally) a flag for out-of-distribution inputs.

    import seggnosis
    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20)
    result = trusted.predict(image_tensor)

    result.mask              # (H, W) predicted class per pixel
    result.probs             # (C, H, W) mean class probabilities
    result.uncertainty_map   # (H, W) pixel-wise uncertainty, higher = less trust
    result.confidence        # scalar overall confidence in [0, 1]
    result.is_ood            # bool, only set if an OOD detector is attached
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class Result:
    """Unified output of any seggnosis wrapper."""

    mask: np.ndarray                 # (H, W) int array of predicted classes
    probs: np.ndarray                # (C, H, W) mean predicted probabilities
    uncertainty_map: np.ndarray      # (H, W) float array, higher = less trustworthy
    confidence: float                # scalar summary in [0, 1], 1 = fully confident
    is_ood: Optional[bool] = None    # set only if an OOD detector was attached
    ood_score: Optional[float] = None
    raw: dict = field(default_factory=dict)  # method-specific extras (e.g. all samples)

    def summary(self) -> str:
        flag = ""
        if self.is_ood is not None:
            flag = "  [OOD]" if self.is_ood else ""
        return (
            f"confidence={self.confidence:.3f} "
            f"mean_uncertainty={self.uncertainty_map.mean():.3f}{flag}"
        )


class BaseWrapper:
    """Common interface every uncertainty-estimation method implements."""

    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self._ood_detector = None

    def attach_ood_detector(self, detector) -> "BaseWrapper":
        """Attach a fitted OOD detector (e.g. seggnosis.ood.MahalanobisOOD)."""
        self._ood_detector = detector
        return self

    def predict(self, x: torch.Tensor, **kwargs) -> Result:
        raise NotImplementedError

    def _finalize(
        self,
        x: torch.Tensor,
        probs: np.ndarray,
        uncertainty_map: np.ndarray,
        raw: Optional[dict] = None,
    ) -> Result:
        """Shared post-processing: mask, confidence, OOD scoring."""
        mask = probs.argmax(axis=0)
        # confidence: 1 - normalized mean uncertainty, clipped to [0, 1]
        u = uncertainty_map
        u_norm = u.mean() / (np.log(probs.shape[0]) + 1e-8)  # normalize by max entropy
        confidence = float(np.clip(1.0 - u_norm, 0.0, 1.0))

        is_ood, ood_score = None, None
        if self._ood_detector is not None:
            ood_score = float(self._ood_detector.score(self.model, x))
            is_ood = bool(ood_score > self._ood_detector.threshold)

        return Result(
            mask=mask,
            probs=probs,
            uncertainty_map=uncertainty_map,
            confidence=confidence,
            is_ood=is_ood,
            ood_score=ood_score,
            raw=raw or {},
        )


def wrap(
    model: torch.nn.Module,
    method: str = "mc_dropout",
    device: Optional[str] = None,
    **method_kwargs: Any,
) -> BaseWrapper:
    """
    Wrap any segmentation model with an uncertainty-estimation method.

    Parameters
    ----------
    model : torch.nn.Module
        A trained segmentation model. forward(x) must return logits of
        shape (B, C, H, W) (or (C, H, W) for a single image).
    method : str
        One of "mc_dropout", "tta", "ensemble".
    device : str, optional
        Torch device to run on. Defaults to the model's current device.
    **method_kwargs
        Passed through to the chosen method's constructor, e.g.
        n_samples=20 for mc_dropout, or models=[m1, m2, m3] for ensemble.

    Returns
    -------
    BaseWrapper
        An object with a `.predict(image) -> Result` method.
    """
    method = method.lower()
    if method == "mc_dropout":
        from .methods.mc_dropout import MCDropoutWrapper
        return MCDropoutWrapper(model, device=device, **method_kwargs)
    elif method == "tta":
        from .methods.tta import TTAWrapper
        return TTAWrapper(model, device=device, **method_kwargs)
    elif method == "ensemble":
        from .methods.ensemble import EnsembleWrapper
        return EnsembleWrapper(model, device=device, **method_kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: mc_dropout, tta, ensemble."
        )
