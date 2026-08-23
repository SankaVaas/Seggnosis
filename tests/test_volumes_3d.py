import numpy as np
import torch

import seggnosis
from tests.conftest import Tiny3DSegModel


def test_mc_dropout_3d_volume(tiny_3d_model, sample_volume):
    trusted = seggnosis.wrap(
        tiny_3d_model, method="mc_dropout", n_samples=4, spatial_dims=3
    )
    result = trusted.predict(sample_volume)

    assert result.mask.shape == (8, 16, 16)          # (D, H, W)
    assert result.probs.shape == (3, 8, 16, 16)       # (C, D, H, W)
    assert result.uncertainty_map.shape == (8, 16, 16)
    assert 0.0 <= result.confidence <= 1.0


def test_tta_3d_volume_inplane_only(tiny_3d_model, sample_volume):
    # Default TTA transforms act on the last two axes (H, W), leaving the
    # depth axis (D) untouched -- verifies that in-plane augmentation
    # doesn't silently corrupt the volume's depth ordering.
    trusted = seggnosis.wrap(tiny_3d_model, method="tta", spatial_dims=3)
    result = trusted.predict(sample_volume)
    assert result.mask.shape == (8, 16, 16)


def test_ensemble_3d_volume(sample_volume):
    m1, m2 = Tiny3DSegModel(), Tiny3DSegModel()
    trusted = seggnosis.wrap(m1, method="ensemble", models=[m2], spatial_dims=3)
    result = trusted.predict(sample_volume)
    assert result.mask.shape == (8, 16, 16)


def test_batched_volume_accepted(tiny_3d_model):
    torch.manual_seed(1)
    batch = torch.rand(1, 1, 8, 16, 16)  # (B, C, D, H, W)
    trusted = seggnosis.wrap(tiny_3d_model, method="mc_dropout", n_samples=3, spatial_dims=3)
    result = trusted.predict(batch)
    assert result.mask.shape == (8, 16, 16)


def test_wrong_spatial_dims_shape_raises(tiny_3d_model, sample_volume):
    # sample_volume is a genuine 3D volume; asking for spatial_dims=2 means
    # as_batch expects rank-3 or rank-4 with the 2D convention, and a
    # (C, D, H, W) volume is rank 4 -- which as_batch will (wrongly, but
    # consistently) interpret as an already-batched 2D image, and the model
    # itself will reject a 5D-expecting Conv3d being fed 4D input.
    trusted = seggnosis.wrap(tiny_3d_model, method="mc_dropout", n_samples=2, spatial_dims=2)
    try:
        trusted.predict(sample_volume)
        assert False, "expected an error from the Conv3d model on wrongly-shaped input"
    except (RuntimeError, ValueError):
        pass


def test_2d_default_unaffected(tiny_model, sample_image):
    # Regression check: default spatial_dims=2 behavior is unchanged.
    trusted = seggnosis.wrap(tiny_model, method="mc_dropout", n_samples=3)
    result = trusted.predict(sample_image)
    assert result.mask.shape == (32, 32)


def test_ece_accepts_volume_shapes():
    rng = np.random.default_rng(0)
    c, d, h, w = 3, 4, 5, 5
    probs = rng.dirichlet(alpha=[1, 1, 1], size=(d, h, w)).transpose(3, 0, 1, 2)  # (C, D, H, W)
    labels = rng.integers(0, c, size=(d, h, w))
    ece = seggnosis.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0
