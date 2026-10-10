# Analysis: what seggnosis 0.6.0 actually does

Companion to `REPORT.md` (which is generated and contains every number and figure).
This file is the interpretation, written by hand from one run of
`benchmarks/run_all.py`. Numbers quoted here are from that run.

## Bottom line

On a controlled synthetic task, the library does what it claims:

| Claim | Evidence | Verdict |
|---|---|---|
| Uncertainty marks the pixels the model gets wrong | Error-detection AUROC 0.81-0.93 for all 9 method x uncertainty combinations; accuracy falls from 100% (most-certain quartile) to ~86-90% (least-certain) | **Works** |
| OOD detection flags shifted inputs | AUROC ~1.0 for noise >= 2x, color shift, blur >= 9, novel shapes; control (no shift) 0.45; beats the model's own confidence (MSP/entropy) on blur and mild noise | **Works, with a layer-choice caveat** |
| Temperature scaling fixes overconfidence | ECE 0.133 -> 0.071 and NLL 1.39 -> 0.74 on an overconfident model; leaves a calibrated model alone (T = 0.97) | **Works; only partly fixes covariate shift** |
| Conformal sets have valid coverage | Coverage at or above target (allowing 1 point of finite-sample slack) in all 48 case/score/alpha combinations | **Works** |
| Conformal sets are *useful* | Only with RAPS on a well-trained model (overlap AUROC 0.89). LAC gives singletons, APS pads, nothing helps an overconfident model | **Conditional** |
| A user can `pip install` and use it | 15/15 scenarios pass from the built wheel in a clean venv, including 13 deliberate misuses that all raise clear errors | **Works** |

## Problems this testing found (and which are now fixed)

These are real bugs/design flaws, found by *running* the library as a user would,
not by the unit tests, which all passed while these were present:

1. **OOD false alarms were far above nominal.** The threshold was computed on the
   same images used to fit the Gaussian. With 60 fit images, 58% of *normal*
   images were flagged (nominal 5%); 26% at 120; 12% at 300; 7.6% at 600.
   Fixed with a cross-fitted threshold (now 1.4% / 2.6% / 3.0% / 4.4%, slightly
   conservative by design). `fit(..., threshold_method="in_sample")` restores
   the old behaviour.
2. **`predict()` changed the user's model mode** for TTA, ensemble and conformal
   (the MC-dropout version of this bug had been fixed earlier). A model the user
   left in train mode came back in eval mode. All methods now go through one
   `eval_mode` context manager, and a test covers every method in both starting
   states (weights, BatchNorm statistics and gradients included).
3. **My own earlier "faster MC dropout" default was a regression on CPU.**
   Replicating all samples into one batch was 3.2x *slower* at 256x256 (1623 ms
   vs 502 ms). The default chunk size is now chosen from the input size and
   device. Honest result against v0.2.1: **parity on CPU (0.97-1.35x)**, not a
   speedup. Any GPU benefit is untested.
4. **`import seggnosis; seggnosis.visualize...` raised `AttributeError`.**
   The submodule is now exposed lazily (matplotlib still isn't imported until used).
5. Two of my own checks were wrong and were corrected rather than kept: a
   "confidence ranking improved" criterion (a rescale cannot change rankings) and
   an "ambiguity" metric that counted unlocatable label-noise pixels.

## Things you should know as a user (not bugs, but not obvious)

- **Choose a deep layer for the OOD detector.** `conv1` (shallow) reached only
  0.55-0.80 AUROC where `conv2`/`conv3` reached ~1.0. The detector works on pooled
  features of whichever layer you name; a shallow layer mostly sees local texture.
- **`Result.confidence` is saturated near 1.0 for TTA + variance** (0.993-1.0 in
  these runs, even after the normalization fix). Use `uncertainty="entropy"` if
  you want an informative scalar confidence, or rank by `uncertainty_map`.
- **Entropy found errors best; mutual information found them least** (e.g. TTA:
  0.920 entropy vs 0.812 MI). That is expected here, because most errors are
  *aleatoric* (random labels) and MI measures only *epistemic* disagreement. The
  library's defaults (TTA -> variance, ensemble -> MI) are the epistemic-leaning
  choice; if your question is "where will the prediction be wrong?", try entropy.
- **Conformal: prefer `score="raps"` (lam about 0.01) for well-trained models.**
  `lac` (the default, kept for the smallest sets) returned a singleton at 100% of
  pixels on the well-trained model -- valid, but it carries no information.
  `aps` is conservative and padded. For an overconfident model no score is
  informative: conformal cannot add information the probabilities lack, and
  temperature scaling first did **not** fix that (it only made APS less
  degenerate: 3.87 -> 2.58).
- **Temperature scaling is a one-number fix.** It halves ECE for an overconfident
  model but only trims ECE by about a fifth under distribution shift (0.33 -> 0.26).
- **Fit the OOD detector on at least a few hundred normal images.** The cross-fit
  threshold keeps false alarms below nominal at every size tested, but the Gaussian
  itself is better estimated with more data.

## What is NOT proven

- **No real data.** Everything is a synthetic shapes task with a small CNN.
- **No GPU.** All timings are one CPU core; CUDA behaviour (including the
  automatic chunk size, which uses a larger budget on CUDA) is untested.
- **Subtle semantic OOD** (e.g. a similar-looking unseen class on real scans) is
  the hard case and is not covered; only noise/blur/color shifts and one
  novel-shape case were tested.
- **3-D volumes** are smoke-tested for shapes only, not for uncertainty quality.
- **Conformal coverage is pooled over pixels** (neighbours are not exchangeable),
  so a single image's coverage can differ from the target.
- **Thresholds for PASS/FAIL were my choices**, stated in the report.

## Recommended before publishing

1. Run `python benchmarks/run_all.py` on your own Windows machine (everything
   except the optional install simulation is plain Python) and compare.
2. Run the same checks on a small slice of *your own* data and model; the
   benchmark functions are short and easy to point at a real DataLoader.
3. Decide whether the conformal default should stay `lac` or become `raps`
   (the evidence here favours `raps` for informativeness; `lac` for set size).
   Nothing from 0.3-0.6 has been released, so this is free to change now.
