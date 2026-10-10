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
    # This checks the repo's layout, not the installed library, so it only
    # makes sense inside an actual git checkout -- e.g. when `tests/` is
    # copied/run standalone against an installed wheel (as in a clean-room
    # install test), there's no .github to find at all. Skip rather than
    # fail in that case instead of asserting something this test was never
    # meant to check outside a repo checkout.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(os.path.join(repo_root, "pyproject.toml")):
        pytest.skip("not running inside a seggnosis repo checkout")

    workflows_dir = os.path.join(repo_root, ".github", "workflows")
    assert os.path.isdir(workflows_dir), (
        "GitHub Actions only discovers workflows under .github/workflows/ "
        "(plural) -- .github/workflow/ is silently ignored."
    )
    assert os.path.isfile(os.path.join(workflows_dir, "tests.yml"))


# ---------------------------------------------------------------------------
# chunk_size: automatic default + explicit override
# ---------------------------------------------------------------------------

def test_auto_chunk_size_is_small_for_big_inputs_on_cpu():
    from seggnosis.utils import auto_chunk_size
    assert auto_chunk_size(torch.rand(1, 3, 32, 32), 20) == 4     # tiny input: a few copies
    assert auto_chunk_size(torch.rand(1, 3, 256, 256), 20) == 1   # big input: one at a time
    assert auto_chunk_size(torch.rand(1, 3, 8, 8), 3) == 3        # never more than n_samples


@pytest.mark.parametrize("chunk", [None, 1, 3, 100])
def test_mc_dropout_chunk_size_variants_all_work(tiny_model, chunk):
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=7, chunk_size=chunk)
    r = trusted.predict(torch.rand(2, 3, 16, 16))
    assert r.raw["samples"].shape[:2] == (7, 2)   # exactly n_samples, regardless of chunking


def test_visualize_submodule_reachable_as_attribute():
    # `import seggnosis; seggnosis.visualize.<fn>` should work without a
    # separate `import seggnosis.visualize`.
    import importlib
    import sys
    sys.modules.pop("seggnosis.visualize", None)
    mod = importlib.reload(seggnosis).visualize
    assert callable(mod.overlay_uncertainty)
    with pytest.raises(AttributeError):
        seggnosis.definitely_not_a_thing


# ---------------------------------------------------------------------------
# 0.6.0: predict() must never change the user's model mode -- for EVERY method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start_training", [False, True])
@pytest.mark.parametrize("method", ["mc_dropout", "tta", "ensemble", "conformal"])
def test_predict_never_changes_model_mode(method, start_training):
    import copy
    model, other = TinySegModel(), TinySegModel()
    model.train(start_training)
    other.train(start_training)
    before = copy.deepcopy(model.state_dict())

    kw = {"mc_dropout": {"n_samples": 3}, "ensemble": {"models": [other]}}.get(method, {})
    trusted = seggnosis.wrap(model, method=method, **kw)
    if method == "conformal":
        g = torch.Generator().manual_seed(0)
        trusted.calibrate([(torch.rand(2, 3, 16, 16, generator=g), torch.randint(0, 4, (2, 16, 16), generator=g))])
    trusted.predict(torch.rand(3, 16, 16))

    assert model.training is start_training
    assert model.dropout.training is start_training
    if method == "ensemble":
        assert other.training is start_training
    for k, v in model.state_dict().items():  # weights AND batchnorm running stats untouched
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("method", ["tta", "ensemble"])
def test_classification_predict_never_changes_model_mode(method):
    from tests.test_classification import TinyClassifier
    m, o = TinyClassifier(), TinyClassifier()
    m.train(); o.train()
    kw = {"models": [o]} if method == "ensemble" else {}
    seggnosis.classification(m, method=method, **kw).predict(torch.rand(3, 32, 32))
    assert m.training is True and o.training is True


# ---------------------------------------------------------------------------
# 0.6.0: OOD threshold honesty (cross-fitted threshold)
# ---------------------------------------------------------------------------

def _ood_loader(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(10, 3, 16, 16, generator=g) for _ in range(n // 10)]


def test_ood_cross_fit_threshold_has_lower_false_alarm_rate_than_in_sample():
    model = TinySegModel().eval()
    fit_data = _ood_loader(60, seed=1)          # deliberately few fit images
    held_out = torch.cat(_ood_loader(1500, seed=2))
    rates = {}
    for method in ("in_sample", "cross_fit"):
        det = MahalanobisOOD(layer_name="conv1").fit(model, fit_data, threshold_method=method)
        rates[method] = float((det.score(model, held_out) > det.threshold).mean())
    # nominal rate is 5%: in-sample is badly optimistic with 60 images, cross-fit is not
    assert rates["in_sample"] > 0.08
    assert rates["cross_fit"] < rates["in_sample"]
    assert rates["cross_fit"] <= 0.07


def test_ood_threshold_method_validation(tiny_model):
    with pytest.raises(ValueError):
        MahalanobisOOD(layer_name="conv1").fit(tiny_model, _ood_loader(40, 1), threshold_method="bogus")
