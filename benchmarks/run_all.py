"""
Regenerate the whole results/ folder:

    python benchmarks/run_all.py                     # everything
    python benchmarks/run_all.py --skip-install-sim  # skip the clean-venv wheel-install simulation
    python benchmarks/run_all.py --reuse-site-packages   # clean venv that reuses your installed torch
                                                         # (much faster/smaller; seggnosis itself is still
                                                         #  installed from the built wheel)

Steps (each can fail independently; the report records what happened):
  1. train/cache the small benchmark models            (benchmarks/.cache)
  2. uncertainty quality, OOD, calibration, conformal  (results/images, results/data)
  3. old (v0.2.1) vs new comparison
  4. build the wheel, install it into a FRESH venv, and run the user simulation there
  5. write results/REPORT.md
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def step(name, fn):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
    t0 = time.time()
    try:
        out = fn()
        print(f"  [{name}] done in {time.time() - t0:.0f}s", flush=True)
        return out
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"  [{name}] FAILED", flush=True)
        return None


def install_simulation(reuse_site_packages):
    repo, results = common.REPO, common.RESULTS
    venv = os.path.join(common.CACHE, "user_venv")
    stage = os.path.join(common.CACHE, "user_stage")
    out = os.path.join(results, "user_simulation")
    for d in (venv, stage):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(stage)
    os.makedirs(out, exist_ok=True)

    dist = os.path.join(repo, "dist")
    shutil.rmtree(dist, ignore_errors=True)
    try:
        import build  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "build"], check=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", dist, repo], check=True)
    wheel = glob.glob(os.path.join(dist, "seggnosis-*.whl"))[0]

    cmd = [sys.executable, "-m", "venv", venv] + (["--system-site-packages"] if reuse_site_packages else [])
    subprocess.run(cmd, check=True)
    py = os.path.join(venv, "Scripts" if sys.platform == "win32" else "bin", "python")
    subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)
    if reuse_site_packages:  # make sure a source checkout on the system path cannot shadow the wheel
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "seggnosis"], check=False)
    subprocess.run([py, "-m", "pip", "install", "-q", wheel, "matplotlib"], check=True)

    for f in ["synthetic_shapes.py", "user_simulation.py"]:
        shutil.copy(os.path.join(HERE, f), stage)
    for w in glob.glob(os.path.join(common.CACHE, "*.pt")):
        shutil.copy(w, stage)
    # run from a neutral directory so nothing from the repo can be imported by accident
    r = subprocess.run([py, os.path.join(stage, "user_simulation.py"), stage, out], cwd=stage)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-install-sim", action="store_true")
    ap.add_argument("--reuse-site-packages", action="store_true")
    args = ap.parse_args()

    import bench_calibration, bench_conformal, bench_ood, bench_uncertainty, bench_version_comparison  # noqa: E402, E401
    import make_report  # noqa: E402

    step("1. models", lambda: [common.get_model(n) for n in common.MODEL_SPECS])
    step("2a. uncertainty quality", bench_uncertainty.run)
    step("2b. OOD detection", bench_ood.run)
    step("2c. calibration", bench_calibration.run)
    step("2d. conformal prediction", bench_conformal.run)
    step("3. old vs new", bench_version_comparison.run)
    if not args.skip_install_sim:
        step("4. clean-install user simulation", lambda: install_simulation(args.reuse_site_packages))
    step("5. report", make_report.run)
    print(f"\nAll done. Open {os.path.join(common.RESULTS, 'REPORT.md')}")


if __name__ == "__main__":
    main()
