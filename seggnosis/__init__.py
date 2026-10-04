"""
seggnosis: model-agnostic uncertainty quantification & OOD detection for
image segmentation models.

    import seggnosis
    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20)
    result = trusted.predict(image)
    result.mask, result.uncertainty_map, result.confidence, result.is_ood
"""

from .core import wrap, Result, BaseWrapper
from .methods.conformal import ConformalWrapper
from ._classification import classification, ClassificationResult, BaseClassificationWrapper
from .ood.mahalanobis import MahalanobisOOD
from .calibration.temperature_scaling import TemperatureScaler
from .calibration.metrics import expected_calibration_error, reliability_curve

__version__ = "0.5.0"

# `segmentation` is a plain alias for `wrap` (the original, per-pixel API):
# now that `classification()` exists for whole-image models, both names say
# which output shape they're for. `wrap` is kept as-is for backward
# compatibility -- existing `seggnosis.wrap(...)` calls are unaffected.
segmentation = wrap

__all__ = [
    "wrap",
    "segmentation",
    "classification",
    "Result",
    "ClassificationResult",
    "BaseWrapper",
    "BaseClassificationWrapper",
    "ConformalWrapper",
    "MahalanobisOOD",
    "TemperatureScaler",
    "expected_calibration_error",
    "reliability_curve",
    "__version__",
]