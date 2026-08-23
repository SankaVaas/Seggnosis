"""
seggnosis: model-agnostic uncertainty quantification & OOD detection for
image segmentation models.

    import seggnosis
    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20)
    result = trusted.predict(image)
    result.mask, result.uncertainty_map, result.confidence, result.is_ood
"""

from .core import wrap, Result, BaseWrapper
from .ood.mahalanobis import MahalanobisOOD
from .calibration.temperature_scaling import TemperatureScaler
from .calibration.metrics import expected_calibration_error, reliability_curve

__version__ = "0.2.1"

__all__ = [
    "wrap",
    "Result",
    "BaseWrapper",
    "MahalanobisOOD",
    "TemperatureScaler",
    "expected_calibration_error",
    "reliability_curve",
    "__version__",
]