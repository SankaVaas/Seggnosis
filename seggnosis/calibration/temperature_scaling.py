from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler:
    """
    Post-hoc calibration for segmentation logits via temperature scaling
    (Guo et al., 2017), adapted to per-pixel classification.

    Fits a single scalar T > 0 that divides the logits before softmax:
        calibrated_probs = softmax(logits / T)
    T is chosen to minimize NLL on a held-out calibration set, without
    changing the model's predictions (argmax is unaffected) -- only how
    confident it's allowed to sound. Raw deep networks are almost always
    overconfident; this fixes that cheaply and is a near-universal
    complement to any of the uncertainty methods in seggnosis.methods.

    Example
    -------
    >>> scaler = TemperatureScaler().fit(model, calibration_loader)
    >>> calibrated_probs = scaler.apply(logits)
    >>> scaler.temperature
    1.73
    """

    def __init__(self):
        self.temperature: Optional[float] = None

    def fit(
        self,
        model: nn.Module,
        calibration_data,
        max_batches: Optional[int] = None,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> "TemperatureScaler":
        """
        Parameters
        ----------
        model : nn.Module
        calibration_data : iterable of (image, label) pairs
            label is a (H, W) integer tensor of ground-truth classes, held
            out from training (do NOT reuse the training set here).
        """
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        try:
            logits_list, labels_list = [], []
            with torch.no_grad():
                for i, (x, y) in enumerate(calibration_data):
                    if max_batches is not None and i >= max_batches:
                        break
                    logits = model(x.to(device))
                    logits_list.append(logits.cpu())
                    labels_list.append(y.cpu())
        finally:
            model.train(was_training)

        logits = torch.cat(logits_list, dim=0)  # (N, C, H, W)
        labels = torch.cat(labels_list, dim=0)  # (N, H, W)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            scaled = logits / temperature.clamp(min=1e-3)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.temperature = float(temperature.detach().clamp(min=1e-3).item())
        return self

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated per-pixel probabilities for raw logits."""
        if self.temperature is None:
            raise RuntimeError("Call .fit(model, calibration_data) first.")
        return F.softmax(logits / self.temperature, dim=1)

    def save(self, path: str) -> None:
        """Save the fitted temperature to a small JSON file."""
        if self.temperature is None:
            raise RuntimeError("Nothing to save -- call .fit(...) first.")
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load a scaler previously saved with `.save(path)`. Skips
        `.fit()` entirely -- ready to `.apply(logits)` immediately."""
        with open(path) as f:
            data = json.load(f)
        scaler = cls()
        scaler.temperature = float(data["temperature"])
        return scaler
