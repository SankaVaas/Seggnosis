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

Batches: pass a (B, C, H, W) tensor (B > 1) and every field above gains a
leading batch axis (mask -> (B, H, W), confidence -> a length-B np.ndarray,
etc.) instead of raising or silently dropping all but the first image. A
single unbatched image, or a batch of exactly 1, still returns the plain
per-image shapes/scalars shown above -- existing single-image code is
unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch

from .utils import normalization_constant


@dataclass
class Result:
    """Unified output of any seggnosis wrapper.

    All fields below are shown in their single-image shape. If the input to
    `.predict()` was a batch of more than one image, every field gains a
    leading batch axis instead (e.g. `mask` becomes (B, H, W), `confidence`
    becomes a length-B `np.ndarray` rather than a `float`, etc.).
    """

    mask: np.ndarray                 # (H, W) int array of predicted classes
    probs: np.ndarray                # (C, H, W) mean predicted probabilities
    uncertainty_map: np.ndarray      # (H, W) float array, higher = less trustworthy
    confidence: Union[float, np.ndarray]                # in [0, 1], 1 = fully confident
    is_ood: Optional[Union[bool, np.ndarray]] = None    # set only if an OOD detector was attached
    ood_score: Optional[Union[float, np.ndarray]] = None
    raw: dict = field(default_factory=dict)

    def summary(self) -> str:
        if isinstance(self.confidence, np.ndarray):
            conf = float(self.confidence.mean())
            unc = float(self.uncertainty_map.mean())
            prefix = f"batch of {self.confidence.shape[0]}: "
            flag = ""
            if self.is_ood is not None:
                n_ood = int(np.asarray(self.is_ood).sum())
                flag = f"  [{n_ood} OOD]" if n_ood else ""
            return f"{prefix}mean_confidence={conf:.3f} mean_uncertainty={unc:.3f}{flag}"

        flag = ""
        if self.is_ood is not None:
            flag = "  [OOD]" if self.is_ood else ""
        return (
            f"confidence={self.confidence:.3f} "
            f"mean_uncertainty={self.uncertainty_map.mean():.3f}{flag}"
        )


class BaseWrapper:
    """Common interface every uncertainty-estimation method implements."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[str] = None,
        spatial_dims: int = 2,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self._ood_detector = None
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 (images) or 3 (volumes)")
        self.spatial_dims = spatial_dims

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
        uncertainty_type: str,
        raw: Optional[dict] = None,
    ) -> Result:
        """Shared post-processing: mask, confidence, OOD scoring.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, *spatial)
            The (already batched, on-device) model input, needed only for
            OOD scoring.
        probs : np.ndarray, shape (B, C, *spatial)
            Mean predicted class probabilities, always with a batch axis.
        uncertainty_map : np.ndarray, shape (B, *spatial)
            Pixel-wise uncertainty, always with a batch axis.
        uncertainty_type : str
            "entropy", "variance", or "mutual_information" -- used to pick
            the right normalization constant for `confidence` (see
            `seggnosis.utils.normalization_constant`).

        When `probs.shape[0] == 1`, the returned `Result` has its batch
        axis squeezed away so single-image callers see the same shapes and
        scalar types as before batch support was added.
        """
        n_classes = probs.shape[1]
        mask = probs.argmax(axis=1)  # (B, *spatial)

        max_uncertainty = normalization_constant(uncertainty_type, n_classes)
        u_norm = uncertainty_map.reshape(uncertainty_map.shape[0], -1).mean(axis=1) / max_uncertainty
        confidence = np.clip(1.0 - u_norm, 0.0, 1.0)  # (B,)

        is_ood_arr, ood_score_arr = None, None
        if self._ood_detector is not None:
            ood_score_arr = np.atleast_1d(
                np.asarray(self._ood_detector.score(self.model, x), dtype=float)
            )
            is_ood_arr = ood_score_arr > self._ood_detector.threshold

        batched = probs.shape[0] > 1
        if batched:
            return Result(
                mask=mask,
                probs=probs,
                uncertainty_map=uncertainty_map,
                confidence=confidence,
                is_ood=is_ood_arr,
                ood_score=ood_score_arr,
                raw=raw or {},
            )

        # Single image: squeeze the batch axis for backward-compatible shapes/types.
        return Result(
            mask=mask[0],
            probs=probs[0],
            uncertainty_map=uncertainty_map[0],
            confidence=float(confidence[0]),
            is_ood=bool(is_ood_arr[0]) if is_ood_arr is not None else None,
            ood_score=float(ood_score_arr[0]) if ood_score_arr is not None else None,
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
        Also accepts spatial_dims=2 (default, images) or spatial_dims=3
        (volumetric data, e.g. CT/MRI: (C, D, H, W) or (B, C, D, H, W)).

    Returns
    -------
    BaseWrapper
        An object with a `.predict(image) -> Result` method. `.predict()`
        accepts a batch (B, C, *spatial) with B > 1; see `Result` for how
        batched outputs are shaped.
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
