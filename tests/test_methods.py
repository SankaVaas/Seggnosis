import numpy as np
import torch

import seggnosis
from tests.conftest import TinySegModel


def test_mc_dropout_predict(tiny_model, sample_image):
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=5)
    result = trusted.predict(sample_image)

    assert result.mask.shape == (32, 32)
    assert result.probs.shape == (4, 32, 32)
    assert result.uncertainty_map.shape == (32, 32)
    assert 0.0 <= result.confidence <= 1.0
    assert np.isfinite(result.uncertainty_map).all()


def test_mc_dropout_uncertainty_variants(tiny_model, sample_image):
    for u in ("entropy", "variance", "mutual_information"):
        trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=4, uncertainty=u)
        result = trusted.predict(sample_image)
        assert result.uncertainty_map.shape == (32, 32)


def test_tta_predict(tiny_model, sample_image):
    trusted = seggnosis.wrap(tiny_model, method="tta")
    result = trusted.predict(sample_image)
    assert result.mask.shape == (32, 32)
    assert result.probs.shape[0] == 4


def test_ensemble_predict(sample_image):
    m1, m2, m3 = TinySegModel(), TinySegModel(), TinySegModel()
    trusted = seggnosis.wrap(m1, method="ensemble", models=[m2, m3])
    result = trusted.predict(sample_image)
    assert result.mask.shape == (32, 32)
    assert result.raw["samples"].shape[0] == 3


def test_batched_input_accepted(tiny_model):
    torch.manual_seed(1)
    batch = torch.rand(1, 3, 16, 16)
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    result = trusted.predict(batch)
    assert result.mask.shape == (16, 16)


def test_invalid_method_raises(tiny_model):
    try:
        seggnosis.wrap(tiny_model, method="not_a_real_method")
        assert False, "expected ValueError"
    except ValueError:
        pass
