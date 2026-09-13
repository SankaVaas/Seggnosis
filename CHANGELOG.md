# Changelog

## 0.3.0

### Fixed
- **MC Dropout no longer leaves the model mutated.** `enable_dropout()` was
  never paired with a restore step, so after any `MCDropoutWrapper.predict()`
  call the wrapped model's dropout layers stayed in `train()` mode forever --
  including during the same call's OOD scoring, and for any other code that
  reused the same model object afterwards. `predict()` now restores the
  model's exact prior mode (train/eval) in a `finally` block, even if the
  forward pass raises.
- **`Result.confidence` was wrong for `uncertainty="variance"`.** Confidence
  was always computed as `1 - mean_uncertainty / log(n_classes)`, which is
  only the correct normalizer for `"entropy"`/`"mutual_information"`.
  `"variance"` is bounded by `0.25`, not `log(n_classes)` -- using the wrong
  constant silently compressed it towards 0 and made confidence spuriously
  close to 1 regardless of how uncertain the model actually was. Added
  `seggnosis.utils.normalization_constant()` and wired the correct constant
  through for each uncertainty type.
- **Multi-image batches were silently truncated to their first image.**
  `as_batch()` accepted `(B, C, H, W)` with `B > 1`, but every wrapper's
  `predict()` then hard-indexed `[0]`, so a batch of 4 images quietly
  returned a result for image 0 only, four times over. All three methods
  (`mc_dropout`, `tta`, `ensemble`) now process the whole batch; `Result`
  gains a leading batch axis on every field (`confidence`, `is_ood`, and
  `ood_score` become length-B arrays) whenever `B > 1`. Single-image and
  batch-of-1 inputs are unaffected -- shapes and scalar types are identical
  to 0.2.x.
- **`MahalanobisOOD.score()` had the same batch-truncation bug** and is now
  vectorized: pass a batch and get back a length-B array of scores.
- Fixed `.github/workflow/` -> `.github/workflows/` (GitHub Actions only
  discovers workflows under the plural directory name; CI, the build check,
  and the PyPI publish workflow had never actually run).

### Added
- **Persistence for fitted OOD detectors and calibrators.**
  `MahalanobisOOD.save(path)` / `MahalanobisOOD.load(path)` and
  `TemperatureScaler.save(path)` / `TemperatureScaler.load(path)`, so you can
  fit once and load into a serving process without access to the original
  fitting data.
- **Faster MC Dropout via batched replication.** Instead of `n_samples`
  sequential single-pass forward calls, `MCDropoutWrapper` now replicates
  the input along the batch axis and runs all samples in as few batched
  forward calls as memory allows. New `chunk_size` constructor argument
  caps how many samples are replicated into one call, trading speed for
  memory on large volumes or high `n_samples` (`chunk_size=1` recovers the
  original one-call-per-sample behavior).
- More robust covariance handling in `MahalanobisOOD.fit()`: uses
  `np.linalg.pinv` instead of a plain inverse, and exposes the ridge
  regularization strength as a `reg` constructor argument.

### Changed
- `pixelwise_entropy`, `pixelwise_variance`, and `mutual_information` in
  `seggnosis.utils` gained a `channel_axis` argument (default preserves
  0.2.x behavior) so they can operate on batched `(N, B, C, *spatial)`
  sample arrays as well as the original unbatched `(N, C, *spatial)` layout.
- Fixed a docstring/code mismatch in `pixelwise_variance`: the doc said
  "summed across classes", the code always meant (averaged) across classes;
  the docstring now matches the implementation.

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