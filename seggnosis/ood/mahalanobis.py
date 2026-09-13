from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class MahalanobisOOD:
    """
    Feature-space Mahalanobis distance for out-of-distribution detection.

    Fits a single Gaussian to the pooled intermediate-layer activations of
    your model over in-distribution (training/validation) data. At
    inference time, a new input's activation vector is scored by its
    Mahalanobis distance to that Gaussian -- inputs that look nothing like
    training data (different scanner, different modality, corrupted image,
    a natural-image photo passed to a medical model by mistake, ...) score
    high and get flagged.

    This does NOT require labels, retraining, or access to the training
    loop -- only a forward hook on one layer of the already-trained model
    and a batch of representative in-distribution images.

    Example
    -------
    >>> detector = MahalanobisOOD(layer_name="encoder.layer4")
    >>> detector.fit(model, in_distribution_loader)
    >>> trusted = seggnosis.wrap(model, method="mc_dropout").attach_ood_detector(detector)
    >>> result = trusted.predict(new_image)
    >>> result.is_ood, result.ood_score
    """

    def __init__(self, layer_name: str, threshold: Optional[float] = None, reg: float = 1e-6):
        self.layer_name = layer_name
        self.threshold = threshold  # set by fit() if not given explicitly
        self.reg = reg
        self._mean: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._feature: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _get_layer(self, model: nn.Module) -> nn.Module:
        module = dict(model.named_modules()).get(self.layer_name)
        if module is None:
            available = ", ".join(list(dict(model.named_modules()).keys())[:15])
            raise ValueError(
                f"Layer '{self.layer_name}' not found in model. "
                f"First few available layers: {available} ..."
            )
        return module

    def _register_hook(self, model: nn.Module) -> None:
        layer = self._get_layer(model)

        def _hook(_module, _input, output):
            # global-average-pool feature map -> a single vector per image
            self._feature = output.mean(dim=list(range(2, output.dim())))

        self._hook_handle = layer.register_forward_hook(_hook)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def fit(
        self,
        model: nn.Module,
        in_distribution_data: Iterable,
        max_batches: Optional[int] = None,
        percentile_for_threshold: float = 95.0,
    ) -> "MahalanobisOOD":
        """
        Parameters
        ----------
        model : nn.Module
        in_distribution_data : iterable of tensors or (tensor, ...) tuples
            E.g. a DataLoader over your training or validation set.
        max_batches : int, optional
            Cap how many batches to use, for speed on large datasets.
        percentile_for_threshold : float
            The OOD threshold is set to this percentile of in-distribution
            scores, so ~ (100 - percentile)% of clean validation data would
            itself be flagged. 95 is a reasonable default.
        """
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        self._register_hook(model)

        try:
            features = []
            for i, batch in enumerate(in_distribution_data):
                if max_batches is not None and i >= max_batches:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x.to(device))
                features.append(self._feature.cpu().numpy())
        finally:
            self._remove_hook()
            model.train(was_training)

        features = np.concatenate(features, axis=0)  # (N, D)
        if features.shape[0] < 2:
            raise ValueError(
                "MahalanobisOOD.fit needs at least 2 in-distribution samples "
                f"to estimate a covariance matrix, got {features.shape[0]}."
            )

        self._mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(cov.shape[0]) * self.reg
        # pinv rather than a plain inverse: robust even when the feature
        # dimension approaches or exceeds the number of fitted samples,
        # where `cov` can be near-singular despite the ridge regularizer.
        self._inv_cov = np.linalg.pinv(cov)

        # Set the threshold from the in-distribution scores themselves
        train_scores = self._mahalanobis_batch(features)
        self.threshold = float(np.percentile(train_scores, percentile_for_threshold))
        return self

    def _mahalanobis_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Vectorized squared Mahalanobis distance for a batch of feature
        vectors, shape (N, D) -> (N,)."""
        diff = vecs - self._mean
        return np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff)

    @torch.no_grad()
    def score(self, model: nn.Module, x: torch.Tensor) -> Union[float, np.ndarray]:
        """
        Mahalanobis distance of `x`'s features to the in-distribution
        Gaussian.

        Returns a plain `float` when `x` is a single image (batch size 1,
        including an unbatched input), or a length-B `np.ndarray` when `x`
        is a batch of B > 1 images -- one score per image.
        """
        if self._mean is None:
            raise RuntimeError("Call .fit(model, in_distribution_data) before scoring.")
        was_training = model.training
        model.eval()
        self._register_hook(model)
        try:
            model(x)
        finally:
            self._remove_hook()
            model.train(was_training)
        scores = self._mahalanobis_batch(self._feature.cpu().numpy())  # (B,)
        if scores.shape[0] == 1:
            return float(scores[0])
        return scores

    def save(self, path: str) -> None:
        """Save the fitted detector (mean, inverse covariance, threshold,
        layer name) to a single `.npz` file for later reuse without
        re-fitting."""
        if self._mean is None:
            raise RuntimeError("Nothing to save -- call .fit(...) first.")
        np.savez(
            path,
            mean=self._mean,
            inv_cov=self._inv_cov,
            threshold=np.array(self.threshold, dtype=np.float64),
            reg=np.array(self.reg, dtype=np.float64),
            layer_name=np.array(self.layer_name),
        )

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        """Load a detector previously saved with `.save(path)`. Skips
        `.fit()` entirely -- ready to `.score(model, x)` immediately."""
        data = np.load(path, allow_pickle=False)
        detector = cls(
            layer_name=str(data["layer_name"]),
            threshold=float(data["threshold"]),
            reg=float(data["reg"]),
        )
        detector._mean = data["mean"]
        detector._inv_cov = data["inv_cov"]
        return detector
