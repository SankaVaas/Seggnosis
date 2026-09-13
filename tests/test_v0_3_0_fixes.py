"""
Tests for the 0.3.0 fixes/additions:
- MC Dropout no longer leaves the model mutated after predict()
- Result.confidence is normalized correctly per uncertainty type
- Real multi-image batch support across mc_dropout / tta / ensemble
- MahalanobisOOD batch scoring + persistence (save/load)
- TemperatureScaler persistence (save/load)
- CI workflow files live where GitHub Actions actually looks for them
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import seggnosis
from seggnosis import MahalanobisOOD, TemperatureScaler
from seggnosis.utils import normalization_constant
from tests.conftest import TinySegModel


# ---------------------------------------------------------------------------
# Bug fix: MC Dropout must restore the model's original mode after predict()
# ---------------------------------------------------------------------------

def test_mc_dropout_restores_model_state(tiny_model, sample_image):
    assert tiny_model.training is False  # fixture calls .eval()
    assert tiny_model.dropout.training is False

    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=5)
    trusted.predict(sample_image)

    # The model (and specifically its dropout layer) must be back in eval
    # mode after predict() returns -- previously it was left in train mode.
    assert tiny_model.training is False
    assert tiny_model.dropout.training is False


def test_mc_dropout_restores_train_mode_if_model_was_training(sample_image):
    model = TinySegModel()
    model.train()  # simulate a caller who had the model in train mode
    assert model.training is True

    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=3)
    trusted.predict(sample_image)

    # predict() should restore *whatever* mode the model was in, not force eval.
    assert model.training is True


def test_mc_dropout_restores_state_even_on_error(tiny_model):
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    bad_input = torch.rand(3, 5)  # wrong rank, as_batch will raise
    with pytest.raises(ValueError):
        trusted.predict(bad_input)
    assert tiny_model.training is False
    assert tiny_model.dropout.training is False


# ---------------------------------------------------------------------------
# Bug fix: confidence normalization must match each uncertainty type's range
# ---------------------------------------------------------------------------

def test_normalization_constant_values():
    assert normalization_constant("entropy", 4) == pytest.approx(np.log(4), rel=1e-6)
    assert normalization_constant("mutual_information", 4) == pytest.approx(np.log(4), rel=1e-6)
    assert normalization_constant("variance", 4) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        normalization_constant("bogus", 4)


def test_confidence_bounds_hold_for_variance(tiny_model, sample_image):
    # Before the fix, "variance" uncertainty was normalized by log(C)
    # instead of its true bound (0.25), which compresses u_norm towards 0
    # and makes confidence spuriously close to 1 regardless of C.
    trusted = seggnosis.wrap(
        tiny_model, method="mc_dropout", n_samples=10, uncertainty="variance"
    )
    result = trusted.predict(sample_image)
    assert 0.0 <= result.confidence <= 1.0
    # Sanity: confidence should react to the *actual* variance scale, i.e.
    # 1 - (mean_variance / 0.25), not some scaled-down version of it.
    expected = float(np.clip(1.0 - result.uncertainty_map.mean() / 0.25, 0.0, 1.0))
    assert result.confidence == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# New: real multi-image batch support (previously silently returned only
# the first image in the batch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mc_dropout", "tta"])
def test_batched_predict_all_methods_2d(tiny_model, method):
    torch.manual_seed(0)
    batch = torch.rand(4, 3, 16, 16)
    kwargs = {"n_samples": 3} if method == "mc_dropout" else {}
    trusted = seggnosis.wrap(tiny_model, method=method, **kwargs)
    result = trusted.predict(batch)

    assert result.mask.shape == (4, 16, 16)
    assert result.probs.shape == (4, 4, 16, 16)
    assert result.uncertainty_map.shape == (4, 16, 16)
    assert isinstance(result.confidence, np.ndarray)
    assert result.confidence.shape == (4,)
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))


def test_batched_predict_ensemble():
    m1, m2 = TinySegModel(), TinySegModel()
    torch.manual_seed(0)
    batch = torch.rand(3, 3, 16, 16)
    trusted = seggnosis.wrap(m1, method="ensemble", models=[m2])
    result = trusted.predict(batch)
    assert result.mask.shape == (3, 16, 16)
    assert result.raw["samples"].shape == (2, 3, 4, 16, 16)  # (N_models, B, C, H, W)


def test_batched_mc_dropout_varies_per_image(tiny_model):
    # MC Dropout draws its stochastic masks jointly across a replicated
    # batch, so a batched call isn't expected to be bit-identical to
    # separate single-image calls under the same seed -- what matters is
    # that each image in the batch gets its own, distinct result (the
    # pre-fix bug always returned image 0's result for every slot).
    torch.manual_seed(0)
    img_a = torch.rand(3, 16, 16)
    img_b = torch.rand(3, 16, 16) * 5.0  # clearly different input
    batch = torch.stack([img_a, img_b], dim=0)

    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=5)
    result = trusted.predict(batch)
    assert not np.array_equal(result.mask[0], result.mask[1])
    assert not np.array_equal(result.probs[0], result.probs[1])


def test_batched_tta_matches_single_image_calls(tiny_model):
    # TTA is fully deterministic (no dropout, no RNG), and each transform
    # is applied per-image without mixing batch elements -- so a batched
    # call must reproduce exactly what separate single-image calls give.
    torch.manual_seed(0)
    img_a = torch.rand(3, 16, 16)
    img_b = torch.rand(3, 16, 16) * 5.0
    batch = torch.stack([img_a, img_b], dim=0)

    trusted = seggnosis.wrap(tiny_model, method="tta")
    batched_result = trusted.predict(batch)
    single_a = trusted.predict(img_a)
    single_b = trusted.predict(img_b)

    assert np.allclose(batched_result.probs[0], single_a.probs, atol=1e-5)
    assert np.allclose(batched_result.probs[1], single_b.probs, atol=1e-5)
    assert not np.array_equal(batched_result.mask[0], batched_result.mask[1])


def test_single_image_result_shapes_unaffected_by_batch_support(tiny_model, sample_image):
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    result = trusted.predict(sample_image)
    assert result.mask.shape == (32, 32)
    assert isinstance(result.confidence, float)
    assert result.is_ood is None


def test_batch_of_one_returns_unbatched_shapes(tiny_model):
    torch.manual_seed(0)
    batch_of_one = torch.rand(1, 3, 16, 16)
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    result = trusted.predict(batch_of_one)
    assert result.mask.shape == (16, 16)
    assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# New: MahalanobisOOD batch scoring + persistence
# ---------------------------------------------------------------------------

def _loader(n_batches=8, batch_size=4, size=16, shift=0.0):
    torch.manual_seed(0)
    for _ in range(n_batches):
        yield torch.rand(batch_size, 3, size, size) + shift


def test_mahalanobis_batch_scoring(tiny_model):
    detector = MahalanobisOOD(layer_name="conv1")
    detector.fit(tiny_model, _loader(), max_batches=8)

    torch.manual_seed(0)
    batch = torch.cat([torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16) + 5.0], dim=0)
    scores = detector.score(tiny_model, batch)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (4,)
    # the shifted (OOD-like) images should score higher on average
    assert scores[2:].mean() > scores[:2].mean()


def test_mahalanobis_save_load_roundtrip(tiny_model, tmp_path):
    detector = MahalanobisOOD(layer_name="conv1")
    detector.fit(tiny_model, _loader(), max_batches=8)

    path = str(tmp_path / "detector.npz")
    detector.save(path)
    loaded = MahalanobisOOD.load(path)

    assert loaded.layer_name == detector.layer_name
    assert loaded.threshold == pytest.approx(detector.threshold)

    x = torch.rand(1, 3, 16, 16)
    original_score = detector.score(tiny_model, x)
    loaded_score = loaded.score(tiny_model, x)
    assert loaded_score == pytest.approx(original_score, rel=1e-6)


def test_mahalanobis_fit_restores_model_mode():
    model = TinySegModel()
    model.train()
    detector = MahalanobisOOD(layer_name="conv1")
    detector.fit(model, _loader(n_batches=2))
    assert model.training is True


# ---------------------------------------------------------------------------
# New: TemperatureScaler persistence
# ---------------------------------------------------------------------------

def test_temperature_scaler_save_load_roundtrip(tiny_model, tmp_path):
    torch.manual_seed(0)
    calib_data = [
        (torch.rand(2, 3, 16, 16), torch.randint(0, 4, (2, 16, 16)))
        for _ in range(5)
    ]
    scaler = TemperatureScaler().fit(tiny_model, calib_data)

    path = str(tmp_path / "scaler.json")
    scaler.save(path)
    loaded = TemperatureScaler.load(path)

    assert loaded.temperature == pytest.approx(scaler.temperature)

    logits = tiny_model(torch.rand(1, 3, 16, 16))
    assert torch.allclose(scaler.apply(logits), loaded.apply(logits))


# ---------------------------------------------------------------------------
# CI: workflows must live in .github/workflows (plural) or GitHub Actions
# silently never runs them.
# ---------------------------------------------------------------------------

def test_ci_workflows_in_correct_directory():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflows_dir = os.path.join(repo_root, ".github", "workflows")
    assert os.path.isdir(workflows_dir), (
        "GitHub Actions only discovers workflows under .github/workflows/ "
        "(plural) -- .github/workflow/ is silently ignored."
    )
    assert os.path.isfile(os.path.join(workflows_dir, "tests.yml"))
