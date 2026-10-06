# Benchmarks

Unlike `tests/` (API contract tests -- shapes, types, error handling),
this directory checks whether seggnosis's numbers are actually *meaningful*
against ground truth, using a small real trained model on a synthetic
segmentation task with genuine, irreducible label ambiguity.

Run: `python validate_library.py` (from inside this directory).

See `RESULTS.md` for the last recorded run and an honest discussion of what
each check does and doesn't prove.
