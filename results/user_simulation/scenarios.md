# User-simulation scenarios (seggnosis 0.6.0, installed from the built wheel)

**15/15 passed.**  Python 3.12.3, torch 2.14.1+cu130, linux.

| ID | Scenario | Result | What was observed |
|---|---|---|---|
| S01 | Installation looks like a normal pip install | PASS | v0.6.0 from /home/claude/seggnosis_work/benchmarks/.cache/user_venv/lib/python3.12/site-packages/seggnosis; pip check clean; cold import incl. torch 2.71s |
| S02 | README quickstart: wrap(...).predict(image) | PASS | mask(32, 32) probs(4, 32, 32); confidence=0.923 mean_uncertainty=0.106 |
| S03 | Every method x every uncertainty type runs | PASS | 9/9 combinations OK |
| S04 | Batch of images | PASS | mask(5, 32, 32) confidence(5,); identical to per-image predictions |
| S05 | 3-D volumes (spatial_dims=3) | PASS | mc_dropout, tta, ensemble OK on single and batched volumes |
| S06 | OOD detection through result.is_ood | PASS | false alarms 7%, shifted detected 100%; save/load round-trip identical |
| S07 | Temperature scaling on an overconfident model | PASS | T=2.22; ECE 0.1332 -> 0.0712 (47% lower) |
| S08 | Conformal prediction (LAC / APS / RAPS) | PASS | lac: cov 0.961, size 1.00; aps: cov 0.996, size 1.78; raps: cov 0.988, size 1.26 \| error before calibrate: "Call .calibrate(calibration_data) before .predict() -- the conformal t..." |
| S09 | Whole-image classification API | PASS | mc_dropout->class 3; tta->class 3; ensemble->class 3; OOD attached OK |
| S10 | Mistakes produce clear, specific error messages | PASS | 13/13 misuse cases raise a clear error |
| S11 | The user's model is left untouched by predict() | PASS | mode, weights and grads unchanged for 4 methods x {eval, train} starting states |
| S12 | Determinism | PASS | tta/ensemble identical across calls; mc_dropout differs between calls, identical with torch.manual_seed |
| S13 | Documentation is present | PASS | all 12 public names documented |
| S14 | Visualization helpers produce figures | PASS | user_mask_comparison.png, user_overlay.png written |
| S15 | Typical latency on CPU (informational) | PASS | cuda available: False; 32x32 mc_dropout: 9 ms; 32x32 tta: 3 ms; 32x32 ensemble: 2 ms; 64x64 mc_dropout: 37 ms; 64x64 tta: 8 ms; 64x64 ensemble: 6 ms |
