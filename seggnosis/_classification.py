"""
Classification variant of seggnosis: the same uncertainty-quantification and
OOD-detection machinery as the segmentation API (`seggnosis.wrap`), applied
to whole-image classifiers -- models whose forward pass returns logits of
shape (B, C) rather than per-pixel logits of shape (B, C, H, W).

    import seggnosis
    trusted = seggnosis.classification(model, method="mc_dropout", n_samples=20)
    result = trusted.predict(image)
    result.predicted_class, result.confidence, result.uncertainty, result.is_ood

`seggnosis.segmentation` is a plain alias for `seggnosis.wrap` (the original,
per-pixel API) -- both now live side by side under names that say which
output shape they're for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import torch

from .methods.tta import DEFAULT_TRANSFORMS
from .utils import (
    as_batch,
    chunked_replicated_batches,
    enable_dropout,
    mutual_information,
    normalization_constant,
    pixelwise_entropy,
    pixelwise_variance,
    softmax_probs,
)


@dataclass
class ClassificationResult:
    """Unified output of any `seggnosis.classification()` wrapper.

    All fields below are shown in their single-image shape/type. If the
    input to `.predict()` was a batch of more than one image, every field
    gains a leading batch axis instead (mirroring `seggnosis.core.Result`'s
    batching behavior): `predicted_class` and `probs` become `np.ndarray`,
    `confidence`/`uncertainty`/`is_ood`/`ood_score` become length-B arrays.
    """

    predicted_class: Union[int, np.ndarray]
    probs: np.ndarray                                    # (C,) mean predicted probabilities
    confidence: Union[float, np.ndarray]                 # in [0, 1], 1 = fully confident
    uncertainty: Union[float, np.ndarray]                # >= 0, higher = less trustworthy
    is_ood: Optional[Union[bool, np.ndarray]] = None      # set only if an OOD detector was attached
    ood_score: Optional[Union[float, np.ndarray]] = None
    raw: dict = field(default_factory=dict)

    def summary(self) -> str:
        if isinstance(self.confidence, np.ndarray):
            flag = ""
            if self.is_ood is not None:
                n_ood = int(np.asarray(self.is_ood).sum())
                flag = f"  [{n_ood} OOD]" if n_ood else ""
            return (
                f"batch of {self.confidence.shape[0]}: "
                f"mean_confidence={float(self.confidence.mean()):.3f} "
                f"mean_uncertainty={float(self.uncertainty.mean()):.3f}{flag}"
            )
        flag = ""
        if self.is_ood is not None:
            flag = "  [OOD]" if self.is_ood else ""
        return (
            f"class={self.predicted_class} confidence={self.confidence:.3f} "
            f"uncertainty={self.uncertainty:.3f}{flag}"
        )


class BaseClassificationWrapper:
    """Common interface every classification uncertainty method implements.

    Mirrors `seggnosis.core.BaseWrapper`, but for models whose forward pass
    returns (B, C) class logits instead of (B, C, H, W) per-pixel logits.
    """

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

    def attach_ood_detector(self, detector) -> "BaseClassificationWrapper":
        """Attach a fitted OOD detector (e.g. seggnosis.ood.MahalanobisOOD)."""
        self._ood_detector = detector
        return self

    def predict(self, x: torch.Tensor, **kwargs) -> ClassificationResult:
        raise NotImplementedError

    def _finalize(
        self,
        x: torch.Tensor,
        probs: np.ndarray,
        uncertainty: np.ndarray,
        uncertainty_type: str,
        raw: Optional[dict] = None,
    ) -> ClassificationResult:
        """Shared post-processing: predicted class, confidence, OOD scoring.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, *spatial)
            The (already batched, on-device) model input, needed only for
            OOD scoring.
        probs : np.ndarray, shape (B, C)
            Mean predicted class probabilities, always with a batch axis.
        uncertainty : np.ndarray, shape (B,)
            Scalar-per-image uncertainty (there's no per-pixel map here).
        uncertainty_type : str
            "entropy", "variance", or "mutual_information" -- picks the
            right normalization constant for `confidence` (see
            `seggnosis.utils.normalization_constant`).

        When `probs.shape[0] == 1`, the returned `ClassificationResult` has
        its batch axis squeezed away, same convention as `seggnosis.wrap`.
        """
        n_classes = probs.shape[1]
        predicted_class = probs.argmax(axis=1)  # (B,)

        max_uncertainty = normalization_constant(uncertainty_type, n_classes)
        confidence = np.clip(1.0 - uncertainty / max_uncertainty, 0.0, 1.0)  # (B,)

        is_ood_arr, ood_score_arr = None, None
        if self._ood_detector is not None:
            ood_score_arr = np.atleast_1d(
                np.asarray(self._ood_detector.score(self.model, x), dtype=float)
            )
            is_ood_arr = ood_score_arr > self._ood_detector.threshold

        batched = probs.shape[0] > 1
        if batched:
            return ClassificationResult(
                predicted_class=predicted_class,
                probs=probs,
                confidence=confidence,
                uncertainty=uncertainty,
                is_ood=is_ood_arr,
                ood_score=ood_score_arr,
                raw=raw or {},
            )

        return ClassificationResult(
            predicted_class=int(predicted_class[0]),
            probs=probs[0],
            confidence=float(confidence[0]),
            uncertainty=float(uncertainty[0]),
            is_ood=bool(is_ood_arr[0]) if is_ood_arr is not None else None,
            ood_score=float(ood_score_arr[0]) if ood_score_arr is not None else None,
            raw=raw or {},
        )


def _uncertainty_from_samples(samples: np.ndarray, uncertainty_type: str) -> np.ndarray:
    """samples: (N, B, C) stochastic/ensemble/TTA softmax outputs -> (B,)."""
    mean_probs = samples.mean(axis=0)  # (B, C)
    if uncertainty_type == "entropy":
        return pixelwise_entropy(mean_probs, channel_axis=1)
    if uncertainty_type == "variance":
        return pixelwise_variance(samples, channel_axis=2)
    return mutual_information(samples, channel_axis=2)


def _check_uncertainty_type(uncertainty: str) -> None:
    if uncertainty not in ("entropy", "variance", "mutual_information"):
        raise ValueError(
            "uncertainty must be 'entropy', 'variance', or 'mutual_information'"
        )


class MCDropoutClassifier(BaseClassificationWrapper):
    """Monte Carlo Dropout uncertainty estimation for classifiers.

    See `seggnosis.methods.mc_dropout.MCDropoutWrapper` for the full
    explanation -- this is the same method, just reading (B, C) logits
    instead of (B, C, H, W) ones. `.predict()` restores the model's
    original mode (including dropout) afterwards, even on error.
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
        _check_uncertainty_type(uncertainty)
        self.uncertainty_type = uncertainty
        self.chunk_size = chunk_size

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)
        batch_size = x.shape[0]

        was_training = self.model.training
        self.model.eval()
        enable_dropout(self.model)
        try:
            sample_chunks = []
            for n, replicated in chunked_replicated_batches(x, self.n_samples, self.chunk_size):
                logits = self.model(replicated)  # (n * B, C)
                probs = softmax_probs(logits).cpu().numpy()
                sample_chunks.append(probs.reshape(n, batch_size, probs.shape[-1]))
        finally:
            self.model.train(was_training)

        samples = np.concatenate(sample_chunks, axis=0)  # (N, B, C)
        mean_probs = samples.mean(axis=0)  # (B, C)
        u = _uncertainty_from_samples(samples, self.uncertainty_type)

        return self._finalize(
            x, mean_probs, u, self.uncertainty_type, raw={"samples": samples}
        )


class TTAClassifier(BaseClassificationWrapper):
    """Test-Time Augmentation uncertainty estimation for classifiers.

    Applies the same flips/rotations as `seggnosis.methods.tta.TTAWrapper`
    to the *input* image; since a classifier's output has no spatial
    structure to invert, each transform's predicted class distribution is
    pooled directly (no inverse-transform step is needed, unlike
    segmentation TTA).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Optional[List] = None,
        uncertainty: str = "variance",
        device: Optional[str] = None,
        spatial_dims: int = 2,
    ):
        super().__init__(model, device=device, spatial_dims=spatial_dims)
        self.transforms = transforms or DEFAULT_TRANSFORMS
        _check_uncertainty_type(uncertainty)
        self.uncertainty_type = uncertainty

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        self.model.eval()
        x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)

        samples = []
        for _name, (fwd, _inv) in self.transforms:
            aug = fwd(x)
            logits = self.model(aug)
            probs = softmax_probs(logits).cpu().numpy()  # (B, C)
            samples.append(probs)
        samples = np.stack(samples, axis=0)  # (N, B, C)

        mean_probs = samples.mean(axis=0)
        u = _uncertainty_from_samples(samples, self.uncertainty_type)

        return self._finalize(
            x, mean_probs, u, self.uncertainty_type, raw={"samples": samples}
        )


class EnsembleClassifier(BaseClassificationWrapper):
    """Deep ensemble uncertainty estimation for classifiers.

    See `seggnosis.methods.ensemble.EnsembleWrapper` -- same method, (B, C)
    logits instead of (B, C, H, W).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        models: Optional[List[torch.nn.Module]] = None,
        uncertainty: str = "mutual_information",
        device: Optional[str] = None,
        spatial_dims: int = 2,
    ):
        all_models = [model] + list(models or [])
        if len(all_models) < 2:
            raise ValueError(
                "EnsembleClassifier needs at least 2 models. Pass the rest via "
                "models=[m2, m3, ...] in seggnosis.classification(model, "
                "method='ensemble', models=[...])"
            )
        super().__init__(all_models[0], device=device, spatial_dims=spatial_dims)
        self.models = [m.to(self.device).eval() for m in all_models]
        _check_uncertainty_type(uncertainty)
        self.uncertainty_type = uncertainty

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)
        samples = []
        for m in self.models:
            logits = m(x)
            probs = softmax_probs(logits).cpu().numpy()  # (B, C)
            samples.append(probs)
        samples = np.stack(samples, axis=0)  # (N, B, C)

        mean_probs = samples.mean(axis=0)
        u = _uncertainty_from_samples(samples, self.uncertainty_type)

        return self._finalize(
            x, mean_probs, u, self.uncertainty_type, raw={"samples": samples}
        )


def classification(
    model: torch.nn.Module,
    method: str = "mc_dropout",
    device: Optional[str] = None,
    **method_kwargs: Any,
) -> BaseClassificationWrapper:
    """
    Wrap a whole-image classifier (forward(x) -> (B, C) logits) with an
    uncertainty-estimation method.

    This is the classification counterpart of `seggnosis.wrap` /
    `seggnosis.segmentation` (same methods, same OOD-detector attachment,
    different output shape). See `seggnosis.wrap` for the full parameter
    docs -- they apply here unchanged, except `Result.mask`/`uncertainty_map`
    (per-pixel) become `ClassificationResult.predicted_class`/`uncertainty`
    (per-image).

    Parameters
    ----------
    model : torch.nn.Module
        A trained classifier. forward(x) must return logits of shape
        (B, C) (or (C,) for a single image).
    method : str
        One of "mc_dropout", "tta", "ensemble".
    device : str, optional
        Torch device to run on. Defaults to the model's current device.
    **method_kwargs
        Passed through to the chosen method's constructor, e.g.
        n_samples=20 for mc_dropout, or models=[m2, m3] for ensemble.

    Returns
    -------
    BaseClassificationWrapper
        An object with a `.predict(image) -> ClassificationResult` method
        and an `.attach_ood_detector(detector)` method.
    """
    method = method.lower()
    if method == "mc_dropout":
        return MCDropoutClassifier(model, device=device, **method_kwargs)
    elif method == "tta":
        return TTAClassifier(model, device=device, **method_kwargs)
    elif method == "ensemble":
        return EnsembleClassifier(model, device=device, **method_kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: mc_dropout, tta, ensemble."
        )
