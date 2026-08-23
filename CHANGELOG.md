# Changelog

## 0.2.1
- Fixed author metadata (was a placeholder in the 0.2.0 PyPI upload).

## 0.2.0
- Added 3D volume support (`spatial_dims=3`) across all three uncertainty
  methods (`mc_dropout`, `tta`, `ensemble`).
- TTA's default transforms now explicitly documented as in-plane-only
  (last two axes), so volumes keep their depth ordering intact.
- Calibration metrics (`expected_calibration_error`, `reliability_curve`)
  now accept both image-shaped and volume-shaped probability/label arrays
  via a shared shape-inference helper.
- Added `tests/test_volumes_3d.py`.
- Added `examples/quickstart_3d.py`.

## 0.1.0
- Initial release: `wrap()` with `mc_dropout`, `tta`, `ensemble` methods.
- `MahalanobisOOD` feature-space out-of-distribution detector.
- `TemperatureScaler` + `expected_calibration_error` for calibration.
- `seggnosis.visualize` plotting helpers.