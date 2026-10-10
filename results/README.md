# results/

Everything here was produced by `python benchmarks/run_all.py` (see `benchmarks/`).

| Start here | |
|---|---|
| `REPORT.md` | The full generated report: scoreboard, every table, every figure, pass/fail criteria |
| `ANALYSIS.md` | Hand-written interpretation: what works, what was found and fixed, what is NOT proven |
| `user_simulation/` | What a new user sees after `pip install` of the built wheel in a clean venv: `scenarios.md` (15 scenarios), `transcript.txt` (everything printed), `images/` (figures made with `seggnosis.visualize`) |
| `images/` | All benchmark figures (PNG) |
| `data/` | Raw numbers as JSON (`uncertainty`, `ood`, `calibration`, `conformal`, `version_comparison`, ...) |

Figures worth opening first: `images/uncertainty_examples.png`, `images/ood_auroc_vs_severity.png`,
`images/calibration_reliability.png`, `images/conformal_adaptivity.png`, `images/improvement_old_vs_new.png`,
`images/ood_false_alarm_vs_fit_size.png`.

All benchmarks use a synthetic task; see "What this does NOT show" in `REPORT.md`.
