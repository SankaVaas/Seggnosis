![seggnosis logo](<docs\images\Seggnosis logo.png>)


# seggnosis

**Model-agnostic uncertainty quantification & out-of-distribution detection for image segmentation models.**

You already have a trained segmentation model — a U-Net, nnU-Net, SAM fine-tune,
custom architecture, whatever. `seggnosis` wraps it, with **no retraining**, and
gives every prediction a trust signal alongside the mask:

```python
import seggnosis

trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20)
result = trusted.predict(image)

result.mask              # (H, W) predicted classes
result.uncertainty_map   # (H, W) pixel-wise uncertainty — where NOT to trust the mask
result.confidence        # scalar summary in [0, 1]
result.is_ood            # True/False, if you've attached an OOD detector
```

## Why

Clinicians and researchers routinely say the same thing about deep segmentation
models: *the mask looks fine, but I don't know where it's guessing.* There is a
large and active research literature on uncertainty quantification and OOD
detection for segmentation (evidential deep learning, conformal prediction,
Mahalanobis-distance detectors, deep ensembles...), but almost every
implementation is a one-off repo tied to one paper's specific architecture.

`seggnosis` is a small, dependency-light layer that works with **any** PyTorch
segmentation model and gives you the three standard uncertainty-estimation
families, an OOD detector, and calibration tools, behind one consistent API.

## Install

```bash
pip install -e .          # from a clone, for now
pip install -e ".[viz]"   # + matplotlib for visualization helpers
```

(Not yet on PyPI — see Roadmap.)

## The three uncertainty methods

| Method | Needs | Idea |
|---|---|---|
| `mc_dropout` | Model already has Dropout layers | Re-enable dropout at inference, run N stochastic passes, measure disagreement |
| `tta` | Nothing extra | Run flips/rotations of the input through the same deterministic model, measure disagreement across views |
| `ensemble` | 2+ independently trained models | Run all of them, measure disagreement across models (usually the most reliable, priciest to obtain) |

Each supports three ways of turning a spread of predictions into a single
uncertainty map: `"entropy"`, `"variance"`, or `"mutual_information"` (BALD —
isolates *model* disagreement from inherent image ambiguity).

```python
seggnosis.wrap(model, method="tta", uncertainty="mutual_information")
seggnosis.wrap(model, method="ensemble", models=[model2, model3], uncertainty="variance")
```

## 2D images and 3D volumes

All three methods work on both 2D images `(C, H, W)` and 3D volumes
`(C, D, H, W)` — CT, MRI, or any other volumetric modality — via
`spatial_dims`:

```python
trusted = seggnosis.wrap(model, method="mc_dropout", spatial_dims=3)
result = trusted.predict(ct_volume)          # (C, D, H, W)
result.mask.shape                            # (D, H, W)
result.uncertainty_map.mean(axis=(1, 2))     # per-slice uncertainty, e.g. to
                                              # flag which slices need review
```

TTA's default flip/rotation transforms act in-plane only (the last two axes),
leaving the depth axis untouched, since most segmentation models are far
more sensitive to out-of-plane distortion than in-plane. See
[`examples/quickstart_3d.py`](examples/quickstart_3d.py).

## Out-of-distribution detection

```python
detector = seggnosis.MahalanobisOOD(layer_name="encoder.layer4")
detector.fit(model, in_distribution_loader)   # unlabeled, just representative images

trusted = seggnosis.wrap(model, method="mc_dropout").attach_ood_detector(detector)
result = trusted.predict(new_scan)
result.is_ood, result.ood_score
```

Fits a Gaussian to pooled activations of one layer over in-distribution data
and flags inputs whose activations are statistically far from it — a
different scanner, modality, corruption, or simply the wrong kind of image
being fed to the model.

## Calibration

Raw softmax confidence from deep networks is usually overconfident.
Temperature scaling fixes this cheaply, post-hoc, on a small held-out labeled
set, without changing the predicted mask:

```python
scaler = seggnosis.TemperatureScaler().fit(model, calibration_loader)
calibrated_probs = scaler.apply(logits)

ece = seggnosis.expected_calibration_error(calibrated_probs, labels)
```

## Visualization

```python
from seggnosis.visualize import overlay_uncertainty, plot_reliability_diagram

overlay_uncertainty(image_np, result.uncertainty_map)
plot_reliability_diagram(calibrated_probs, labels)
```//(requires `pip install -e ".[viz]"`)

## Example

See [`examples/quickstart.py`](examples/quickstart.py) for a full runnable
walkthrough (wrap → predict → attach OOD → calibrate).

## Design principles

- **No retraining.** Everything wraps an already-trained model.
- **Architecture-agnostic.** Works on any `nn.Module` returning `(B, C, H, W)` logits.
- **One consistent output.** Every method returns the same `Result` dataclass.
- **Small and readable.** No hidden magic — read `core.py` in five minutes.

## Roadmap / where contributions are wanted

- [ ] Conformal prediction wrapper (pixel-wise coverage guarantees)
- [x] Support for 3D volumes (`spatial_dims=3`)
- [ ] Deep ensemble training helper (currently BYO trained models)
- [ ] `nnU-Net` / `MONAI` integration examples
- [ ] Batch-level (not just single-image) prediction API
- [ ] PyPI release
- [ ] Benchmark notebook against public medical segmentation OOD datasets (e.g. OpenMIBOOD)

Issues and PRs welcome — this is meant to be a genuinely small, focused
utility, not a framework. If a contribution makes the core harder to read in
one sitting, it's probably out of scope for this library (but might be a
great plugin/extension).

## License

MIT — see [LICENSE](LICENSE).
