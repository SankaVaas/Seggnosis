# Contributing to seggnosis

Thanks for considering it — this project is intentionally small, so the bar
for "does this fit" is more about scope than difficulty.

## Setup

```bash
git clone https://github.com/SankaVaas/Seggnosis.git
cd seggnosis
pip install -e ".[dev]"
pytest
```

## Guidelines

- **Keep the core small.** `core.py` should stay readable top-to-bottom in a
  few minutes. New uncertainty methods go in `seggnosis/methods/`, new OOD
  detectors in `seggnosis/ood/`, new calibration tools in
  `seggnosis/calibration/` — all implementing the same shape of interface as
  their siblings.
- **No hard dependency creep.** `torch` and `numpy` are the only required
  dependencies. Anything else (matplotlib, scikit-learn, etc.) must be an
  optional extra and imported lazily inside the function that needs it.
- **Every new method needs a test** using the tiny synthetic model in
  `tests/conftest.py` — no real datasets required for unit tests.
- **Docstrings explain the "why," not just the "what."** Assume the reader
  knows deep learning but not necessarily this specific method's tradeoffs.

## Good first contributions

- A conformal-prediction wrapper (see Roadmap in README)
- 3D volume support in `utils.py` (entropy/variance already generalize; the
  wrappers currently assume `(C, H, W)`)
- A `sklearn`-style `Pipeline`-esque helper to chain calibration + uncertainty + OOD in one call
- More TTA transform presets (elastic, brightness) with correct inverses
- A benchmark script against a public dataset (e.g. OpenMIBOOD) showing ECE / OOD AUROC before and after seggnosis

## Reporting issues

Please include: model architecture (roughly), input shape, which method you
used, and the full traceback if there is one. A minimal reproducible example
using `tests/conftest.py`'s `TinySegModel` is ideal.
