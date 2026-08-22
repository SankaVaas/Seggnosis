import numpy as np
import torch

import seggnosis
from seggnosis import MahalanobisOOD, TemperatureScaler, expected_calibration_error
from tests.conftest import TinySegModel


def _loader(n_batches=8, batch_size=4, size=16, shift=0.0):
    torch.manual_seed(0)
    for _ in range(n_batches):
        yield torch.rand(batch_size, 3, size, size) + shift


def test_mahalanobis_fit_and_score(tiny_model):
    detector = MahalanobisOOD(layer_name="conv1")
    detector.fit(tiny_model, _loader(), max_batches=8)
    assert detector.threshold is not None

    in_dist_score = detector.score(tiny_model, torch.rand(1, 3, 16, 16))
    ood_score = detector.score(tiny_model, torch.rand(1, 3, 16, 16) + 5.0)
    assert isinstance(in_dist_score, float)
    assert isinstance(ood_score, float)
    # A strongly shifted input should score higher than in-distribution noise
    assert ood_score > in_dist_score


def test_attach_ood_detector_to_wrapper(tiny_model, sample_image):
    detector = MahalanobisOOD(layer_name="conv1")
    detector.fit(tiny_model, _loader(), max_batches=8)

    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    trusted.attach_ood_detector(detector)
    result = trusted.predict(sample_image)

    assert result.is_ood in (True, False)
    assert result.ood_score is not None


def test_invalid_layer_name_raises(tiny_model):
    detector = MahalanobisOOD(layer_name="does_not_exist")
    try:
        detector.fit(tiny_model, _loader(n_batches=1))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_temperature_scaling_fit_and_apply(tiny_model):
    torch.manual_seed(0)
    calib_data = [
        (torch.rand(2, 3, 16, 16), torch.randint(0, 4, (2, 16, 16)))
        for _ in range(5)
    ]
    scaler = TemperatureScaler().fit(tiny_model, calib_data)
    assert scaler.temperature > 0

    logits = tiny_model(torch.rand(1, 3, 16, 16))
    probs = scaler.apply(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1)), atol=1e-4)


def test_expected_calibration_error_bounds():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=[1, 1, 1], size=1000)  # (N, C)
    labels = rng.integers(0, 3, size=1000)
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0
