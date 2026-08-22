from .core import wrap, Result, BaseWrapper
from .ood.mahalanobis import MahalanobisOOD
from .calibration.temperature_scaling import TemperatureScaler
from .calibration.metrics import expected_calibration_error, reliability_curve

__version__ = "0.1.0"

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
