"""
Tests for ConformalWrapper (seggnosis.methods.conformal).
"""

import numpy as np
import pytest
import torch

import seggnosis
from tests.conftest import TinySegModel


def _random_labeled_batches(n_batches, batch_size, size, n_classes, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = torch.rand(batch_size, 3, size, size, generator=g)
        # Labels are independent of x's content -- the conformal coverage
        # guarantee doesn't require the model to be accurate, only that
        # calibration and test pixels are exchangeable (drawn the same way).
        y = torch.randint(0, n_classes, (batch_size, size, size), generator=g)
        yield x, y


def test_conformal_requires_calibration_before_predict(tiny_model, sample_image):
    trusted = seggnosis.wrap(tiny_model, method="conformal")
    with pytest.raises(RuntimeError):
        trusted.predict(sample_image)


def test_conformal_alpha_validation(tiny_model):
    with pytest.raises(ValueError):
        seggnosis.wrap(tiny_model, method="conformal", alpha=0.0)
    with pytest.raises(ValueError):
        seggnosis.wrap(tiny_model, method="conformal", alpha=1.0)
    with pytest.raises(ValueError):
        seggnosis.wrap(tiny_model, method="conformal", alpha=-0.1)


def test_conformal_basic_predict_shapes(tiny_model, sample_image):
    trusted = seggnosis.wrap(tiny_model, method="conformal", alpha=0.1)
    trusted.calibrate(_random_labeled_batches(20, 4, 16, 4, seed=0))
    result = trusted.predict(sample_image)

    assert result.mask.shape == (32, 32)
    assert result.probs.shape == (4, 32, 32)
    assert result.uncertainty_map.shape == (32, 32)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    prediction_set = result.raw["prediction_set"]
    # raw arrays always keep their batch axis, same convention as the
    # other methods' raw["samples"], even when the top-level Result fields
    # are squeezed for a single image.
    assert prediction_set.shape == (1, 4, 32, 32)
    assert prediction_set.dtype == bool
    # every pixel's set must contain at least one class (the top-1 guard)
    assert np.all(prediction_set.sum(axis=1) >= 1)
    # and the top-1 predicted class specifically
    top1 = result.probs.argmax(axis=0)
    assert np.all(np.take_along_axis(prediction_set[0], top1[None, ...], axis=0))


def test_conformal_batch_support(tiny_model):
    trusted = seggnosis.wrap(tiny_model, method="conformal", alpha=0.1)
    trusted.calibrate(_random_labeled_batches(20, 4, 16, 4, seed=0))

    torch.manual_seed(0)
    batch = torch.rand(3, 3, 16, 16)
    result = trusted.predict(batch)
    assert result.mask.shape == (3, 16, 16)
    assert result.raw["prediction_set"].shape == (3, 4, 16, 16)
    assert isinstance(result.confidence, np.ndarray)
    assert result.confidence.shape == (3,)


def test_conformal_smaller_alpha_gives_larger_or_equal_sets(tiny_model):
    # Lower alpha = higher target coverage = the calibrated threshold should
    # never produce smaller prediction sets on average than a larger alpha.
    torch.manual_seed(0)
    image = torch.rand(3, 16, 16)

    loose = seggnosis.wrap(tiny_model, method="conformal", alpha=0.3)
    loose.calibrate(_random_labeled_batches(30, 4, 16, 4, seed=1))
    strict = seggnosis.wrap(tiny_model, method="conformal", alpha=0.05)
    strict.calibrate(_random_labeled_batches(30, 4, 16, 4, seed=1))

    loose_set_size = loose.predict(image).raw["prediction_set"].sum(axis=0).mean()
    strict_set_size = strict.predict(image).raw["prediction_set"].sum(axis=0).mean()
    assert strict_set_size >= loose_set_size


def test_conformal_empirical_coverage_near_target():
    """
    The core promise of conformal prediction: calibrate on one batch of
    (image, label) pixels, and on a held-out batch drawn the same way, the
    true class should land inside the predicted set at least `1 - alpha`
    of the time (up to finite-sample noise) -- regardless of whether the
    underlying model is any good. Uses an untrained model on purpose: the
    guarantee doesn't depend on model accuracy, only on calibration/test
    exchangeability.
    """
    torch.manual_seed(0)
    model = TinySegModel(n_classes=4)
    alpha = 0.1

    trusted = seggnosis.wrap(model, method="conformal", alpha=alpha)
    trusted.calibrate(_random_labeled_batches(60, 8, 16, 4, seed=42), max_batches=60)

    covered, total = 0, 0
    for x, y in _random_labeled_batches(20, 8, 16, 4, seed=123):
        result = trusted.predict(x)
        pred_set = result.raw["prediction_set"]  # (B, C, H, W)
        y_np = y.numpy()  # (B, H, W)
        in_set = np.take_along_axis(pred_set, y_np[:, None, ...], axis=1)[:, 0]
        covered += int(in_set.sum())
        total += in_set.size

    empirical_coverage = covered / total
    # Loose tolerance: this is a statistical guarantee, not exact equality.
    assert empirical_coverage >= (1 - alpha) - 0.1


def test_conformal_restores_model_mode():
    model = TinySegModel()
    model.train()
    trusted = seggnosis.wrap(model, method="conformal")
    trusted.calibrate(_random_labeled_batches(5, 4, 16, 4, seed=0))
    assert model.training is True
