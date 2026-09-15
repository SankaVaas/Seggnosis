from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch

from ..core import BaseWrapper, Result
from ..utils import as_batch, softmax_probs


class ConformalWrapper(BaseWrapper):
    """
    Split conformal prediction for segmentation.

    Unlike the other three methods (mc_dropout / tta / ensemble), which
    produce a *heuristic* uncertainty score from prediction disagreement,
    conformal prediction gives each pixel a *prediction set* of classes --
    e.g. {liver, tumor} instead of just "tumor" -- calibrated on held-out
    labeled data so that, on average across calibration + test pixels drawn
    from the same distribution, the true class is inside the predicted set
    at least `1 - alpha` of the time. That's a distribution-free, finite-
    sample coverage guarantee, not just a plausible-looking score: it holds
    regardless of whether the model is well-calibrated or not.

    Wraps a single deterministic model -- no dropout, no ensemble needed.

    Usage
    -----
    >>> trusted = seggnosis.wrap(model, method="conformal", alpha=0.1)
    >>> trusted.calibrate(held_out_labeled_loader)   # required before predict()
    >>> result = trusted.predict(image)
    >>> result.raw["prediction_set"]   # (C, H, W) bool -- per-pixel class membership
    >>> result.uncertainty_map         # per-pixel (set size - 1); 0 = a confident singleton set

    Caveat on the coverage guarantee
    ---------------------------------
    Standard split conformal prediction assumes the calibration and test
    *points* are exchangeable. Here, each pixel is treated as one
    calibration point, pooled across every image and every pixel position.
    Pixels within the same image are spatially correlated (not truly
    exchangeable with each other), so the `1 - alpha` guarantee should be
    read as a marginal, pooled-over-all-pixels coverage rate rather than a
    per-image or worst-image guarantee. This pixel-pooling approach is
    standard practice in conformal-for-segmentation work and is a
    reasonable, useful approximation in practice, but it is an
    approximation -- a single held-out image's actual pixel coverage will
    vary around `1 - alpha`, sometimes by a fair amount for small images.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate, in (0, 1). alpha=0.1 targets ~90% of
        pixels having their true class inside the prediction set.
    spatial_dims : int
        2 for images (C, H, W), 3 for volumes (C, D, H, W), e.g. CT/MRI.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float = 0.1,
        device: Optional[str] = None,
        spatial_dims: int = 2,
    ):
        super().__init__(model, device=device, spatial_dims=spatial_dims)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.uncertainty_type = "set_size"
        self.qhat: Optional[float] = None  # set by calibrate()

    def _batch_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Labels have no channel dim, unlike `as_batch`'s inputs -- add a
        batch axis if missing."""
        expected_unbatched = self.spatial_dims       # (*spatial)
        expected_batched = self.spatial_dims + 1      # (B, *spatial)
        if y.dim() == expected_unbatched:
            return y.unsqueeze(0)
        if y.dim() == expected_batched:
            return y
        raise ValueError(
            f"With spatial_dims={self.spatial_dims}, expected a label tensor of "
            f"rank {expected_unbatched} (unbatched) or {expected_batched} "
            f"(batched), got rank {y.dim()} with shape {tuple(y.shape)}."
        )

    @torch.no_grad()
    def calibrate(
        self, calibration_data: Iterable, max_batches: Optional[int] = None
    ) -> "ConformalWrapper":
        """
        Fit the conformal threshold on held-out LABELED data (distinct from
        whatever data the model itself was trained on -- reusing training
        data here would break the coverage guarantee).

        Parameters
        ----------
        calibration_data : iterable of (x, y) pairs
            `x`: image/volume tensor, `y`: integer class-label mask of the
            same spatial shape (no channel dimension).
        max_batches : int, optional
            Cap how many batches to use, for speed on large datasets.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            nonconformity_scores = []
            for i, (x, y) in enumerate(calibration_data):
                if max_batches is not None and i >= max_batches:
                    break
                x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)
                y = self._batch_labels(y).to(self.device)

                probs = softmax_probs(self.model(x)).cpu().numpy()  # (B, C, *spatial)
                y_np = y.cpu().numpy()  # (B, *spatial)
                true_class_prob = np.take_along_axis(
                    probs, y_np[:, None, ...], axis=1
                )[:, 0]  # (B, *spatial)
                # Nonconformity score: how far the model's probability for
                # the TRUE class falls short of 1. Low score = the model
                # was confidently right; high score = confidently wrong.
                nonconformity_scores.append((1.0 - true_class_prob).ravel())
        finally:
            self.model.train(was_training)

        scores = np.concatenate(nonconformity_scores)
        n = scores.shape[0]
        if n < 2:
            raise ValueError(
                f"Need at least 2 calibration pixels, got {n}. Pass more "
                "calibration data or raise max_batches."
            )
        # Standard split-conformal quantile (Vovk et al.): the smallest
        # threshold such that at least ceil((n+1)(1-alpha)) of the n
        # calibration scores fall at or below it.
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.qhat = float(np.quantile(scores, q_level, method="higher"))
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Result:
        if self.qhat is None:
            raise RuntimeError(
                "Call .calibrate(calibration_data) before .predict() -- "
                "the conformal threshold hasn't been fit yet."
            )
        self.model.eval()
        x = as_batch(x, spatial_dims=self.spatial_dims).to(self.device)
        probs = softmax_probs(self.model(x)).cpu().numpy()  # (B, C, *spatial)

        # Include every class whose probability is high enough that, had it
        # been the true class, its nonconformity score (1 - prob) would not
        # have exceeded the calibrated threshold.
        prediction_set = probs >= (1.0 - self.qhat)  # (B, C, *spatial) bool

        # Guard against a pathologically small qhat (e.g. alpha very close
        # to 0 with few calibration samples) producing an empty set: always
        # keep at least the top-1 predicted class.
        top1 = probs.argmax(axis=1)  # (B, *spatial)
        np.put_along_axis(prediction_set, top1[:, None, ...], True, axis=1)

        set_size = prediction_set.sum(axis=1).astype(np.float64)  # (B, *spatial)
        umap = set_size - 1.0  # 0 = confident singleton set

        return self._finalize(
            x,
            probs,
            umap,
            self.uncertainty_type,
            raw={
                "prediction_set": prediction_set,
                "qhat": self.qhat,
                "alpha": self.alpha,
                "target_coverage": 1.0 - self.alpha,
            },
        )
