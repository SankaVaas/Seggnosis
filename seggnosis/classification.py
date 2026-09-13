from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .utils import enable_dropout


@dataclass
class ClassificationResult:
    predicted_class: int
    probs: np.ndarray
    confidence: float
    uncertainty: float
    is_ood: Optional[bool] = None
    ood_score: Optional[float] = None
    raw: dict = field(default_factory=dict)

    def summary(self) -> str:
        flag = ""
        if self.is_ood is not None:
            flag = "  [OOD]" if self.is_ood else ""
        return (
            f"class={self.predicted_class} confidence={self.confidence:.3f} "
            f"uncertainty={self.uncertainty:.3f}{flag}"
        )


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 3 else x


class BaseClassificationWrapper:
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self._ood_detector = None

    def attach_ood_detector(self, detector) -> "BaseClassificationWrapper":
        self._ood_detector = detector
        return self

    def predict(self, x: torch.Tensor) -> ClassificationResult:
        raise NotImplementedError

    def _finalize(self, x, mean_probs, samples=None) -> ClassificationResult:
        eps = 1e-8
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
        confidence = float(mean_probs.max())
        predicted_class = int(mean_probs.argmax())

        is_ood, ood_score = None, None
        if self._ood_detector is not None:
            ood_score = float(self._ood_detector.score(self.model, x))
            is_ood = bool(ood_score > self._ood_detector.threshold)

        return ClassificationResult(
            predicted_class=predicted_class, probs=mean_probs, confidence=confidence,
            uncertainty=entropy, is_ood=is_ood, ood_score=ood_score,
            raw={"samples": samples} if samples is not None else {},
        )


class MCDropoutClassifier(BaseClassificationWrapper):
    def __init__(self, model, n_samples: int = 20, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.n_samples = n_samples

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        self.model.eval()
        enable_dropout(self.model)
        x = _as_batch(x).to(self.device)

        samples = []
        for _ in range(self.n_samples):
            logits = self.model(x)
            samples.append(F.softmax(logits, dim=1)[0].cpu().numpy())
        samples = np.stack(samples, axis=0)
        return self._finalize(x, samples.mean(axis=0), samples)


class TTAClassifier(BaseClassificationWrapper):
    DEFAULT_TRANSFORMS = [
        lambda t: t,
        lambda t: torch.flip(t, dims=[-1]),
        lambda t: torch.flip(t, dims=[-2]),
        lambda t: torch.rot90(t, k=1, dims=[-2, -1]),
    ]

    def __init__(self, model, transforms: Optional[List] = None, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.transforms = transforms or self.DEFAULT_TRANSFORMS

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        self.model.eval()
        x = _as_batch(x).to(self.device)
        img = x[0]

        samples = []
        for t in self.transforms:
            aug = t(img).unsqueeze(0)
            logits = self.model(aug)
            samples.append(F.softmax(logits, dim=1)[0].cpu().numpy())
        samples = np.stack(samples, axis=0)
        return self._finalize(x, samples.mean(axis=0), samples)


class EnsembleClassifier(BaseClassificationWrapper):
    def __init__(self, model, models: Optional[List] = None, device: Optional[str] = None):
        all_models = [model] + list(models or [])
        if len(all_models) < 2:
            raise ValueError(
                "EnsembleClassifier needs at least 2 models. Pass the rest via "
                "models=[m2, m3, ...] in seggnosis.classification(model, method='ensemble', models=[...])"
            )
        super().__init__(all_models[0], device=device)
        self.models = [m.to(self.device).eval() for m in all_models]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> ClassificationResult:
        x = _as_batch(x).to(self.device)
        samples = []
        for m in self.models:
            logits = m(x)
            samples.append(F.softmax(logits, dim=1)[0].cpu().numpy())
        samples = np.stack(samples, axis=0)
        return self._finalize(x, samples.mean(axis=0), samples)


def wrap(model: torch.nn.Module, method: str = "mc_dropout", device: Optional[str] = None, **method_kwargs) -> BaseClassificationWrapper:
    method = method.lower()
    if method == "mc_dropout":
        return MCDropoutClassifier(model, device=device, **method_kwargs)
    elif method == "tta":
        return TTAClassifier(model, device=device, **method_kwargs)
    elif method == "ensemble":
        return EnsembleClassifier(model, device=device, **method_kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: mc_dropout, tta, ensemble.")