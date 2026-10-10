"""Benchmark 5: what did the changes since v0.2.1 (the version on PyPI before this work) actually fix?
Runs the identical experiments on identical model weights against both versions."""
import os, subprocess, sys, tempfile
import numpy as np
from common import DATA, HERE, REPO, get_pyplot, load_json, save_json, savefig

plt = get_pyplot()


def run_worker(lib_path, name):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([lib_path, HERE]))
    out = os.path.join(DATA, f"_compare_{name}.json")
    p = subprocess.run([sys.executable, os.path.join(HERE, "compare_worker.py"), out],
                       env=env, capture_output=True, text=True, cwd=tempfile.gettempdir())
    if p.returncode != 0:
        raise RuntimeError(f"worker({name}) failed:\n{p.stderr[-2000:]}")
    import json
    return json.load(open(out))


def old_version_path():
    """Extract the pre-change release (git commit c0aa6ee = v0.2.1) if not already present."""
    for cand in (os.environ.get("SEGGNOSIS_OLD_PATH"), "/tmp/seggnosis_v021"):
        if cand and os.path.isdir(os.path.join(cand, "seggnosis")):
            return cand
    dest = os.path.join(tempfile.gettempdir(), "seggnosis_v021"); os.makedirs(dest, exist_ok=True)
    arch = subprocess.run(["git", "archive", "c0aa6ee", "seggnosis"], cwd=REPO, capture_output=True)
    subprocess.run(["tar", "-x", "-C", dest], input=arch.stdout, check=True)
    return dest


def run():
    old = run_worker(old_version_path(), "old")
    new = run_worker(REPO, "new")
    print(f"  compared v{old['version']} (old) vs v{new['version']} (new)")

    t1o, t1n = old["T1_batch"], new["T1_batch"]
    t2o, t2n = old["T2_confidence_variance"], new["T2_confidence_variance"]
    t3o, t3n = old["T3_model_state"], new["T3_model_state"]
    t4o, t4n = old["T4_ood_batch"], new["T4_ood_batch"]
    t5o, t5n = old["T5_speed_seconds"], new["T5_speed_seconds"]
    summary = {"old_version": old["version"], "new_version": new["version"], "old": old, "new": new,
               "speedups": {k: t5o[k] / t5n[k] for k in t5o}}
    summary["criteria"] = {
        "batch: new version returns one result per input image": t1n["masks_returned"] == 8,
        "batch: new version is accurate on every image, not just the first": min(t1n["per_image_accuracy_vs_own_ground_truth"]) > 0.8,
        "variance confidence: values are less compressed than the old normalization (std >= 3x)": t2n["conf_std"] >= 3 * max(t2o["conf_std"], 1e-9),
        "model state: new version leaves dropout in eval mode": not t3n["dropout_layer_in_train_mode_after_predict"],
        "model state: new version leaves forward passes deterministic": t3n["two_identical_forward_passes_max_abs_diff"] == 0.0,
        "OOD batch: new version scores every image": t4n["scores_returned"] == 8,
        "speed: new MC-dropout is not slower than old at ANY tested input size (>= 0.9x)": all(v >= 0.9 for v in summary["speedups"].values()),
    }
    save_json("version_comparison.json", summary)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.bar(["v%s\n(old)" % old["version"], "v%s\n(new)" % new["version"]], [t1o["masks_returned"], t1n["masks_returned"]], color=["#d62728", "#2ca02c"])
    ax.set_title("Batch of 8 images in:\nresults returned", fontsize=10); ax.set_ylabel("# masks returned")
    ax = axes[0, 1]
    ax.bar(["old", "new"], [t1o["mean_accuracy"], t1n["mean_accuracy"]], color=["#d62728", "#2ca02c"])
    ax.set_ylim(0, 1); ax.set_title("Accuracy of the returned mask(s)\nvs. each image's own ground truth", fontsize=10)
    for k, v in enumerate([t1o["mean_accuracy"], t1n["mean_accuracy"]]): ax.text(k, v + 0.02, f"{v:.0%}", ha="center")
    ax = axes[0, 2]
    for r_, lab, col in [(t2o, "old", "#d62728"), (t2n, "new", "#2ca02c")]:
        ax.scatter(r_["confidence"], r_["accuracy"], s=14, alpha=0.75, color=col, label=f"{lab}  (conf range {r_['conf_min']:.2f}-{r_['conf_max']:.2f})")
    ax.set_xlim(-0.02, 1.02); ax.set_xlabel("Result.confidence"); ax.set_ylabel("Actual image accuracy"); ax.legend(fontsize=8)
    ax.set_title("TTA + variance: confidence is STILL saturated near 1.0\nafter the fix (60 images; see spread, bottom-left)", fontsize=9)
    ax = axes[1, 0]
    ax.bar(["old", "new"], [t2o["conf_std"], t2n["conf_std"]], color=["#d62728", "#2ca02c"])
    ax.set_title("Spread (std) of Result.confidence, TTA + variance\n(old squashes everything to ~1.0; new is "
                 f"{t2n['conf_std'] / max(t2o['conf_std'], 1e-12):.1f}x wider)", fontsize=9)
    ax.text(0.5, 0.92, "image-level ranking is unchanged by design\n(the fix is a monotone rescale; AUROC "
            f"{t2o['auroc_low_confidence_flags_bad_images']:.2f} -> {t2n['auroc_low_confidence_flags_bad_images']:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="gray")
    ax = axes[1, 1]
    labels = ["dropout left\nin train mode", "forward-pass\nnon-determinism"]
    ax.bar(np.arange(2) - 0.18, [float(t3o["dropout_layer_in_train_mode_after_predict"]), min(1.0, t3o["two_identical_forward_passes_max_abs_diff"])], 0.36, color="#d62728", label="old")
    ax.bar(np.arange(2) + 0.18, [float(t3n["dropout_layer_in_train_mode_after_predict"]), min(1.0, t3n["two_identical_forward_passes_max_abs_diff"])], 0.36, color="#2ca02c", label="new")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=9); ax.legend()
    ax.set_title("Is the user's model left untouched after predict()?\n(0 = untouched)", fontsize=10)
    ax = axes[1, 2]
    keys = list(t5o); x = np.arange(len(keys))
    ax.bar(x - 0.18, [t5o[k] * 1000 for k in keys], 0.36, color="#d62728", label="old")
    ax.bar(x + 0.18, [t5n[k] * 1000 for k in keys], 0.36, color="#2ca02c", label="new")
    for i, k in enumerate(keys): ax.text(i + 0.18, t5n[k] * 1000, f"{t5o[k] / t5n[k]:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=8); ax.set_ylabel("ms per predict() (CPU, log)"); ax.set_yscale("log")
    ax.legend(); ax.set_title("MC-dropout speed, 1 CPU core (best of 3 medians; label = old/new)", fontsize=9)
    fig.suptitle(f"What changed since the PyPI release v{old['version']}  ->  v{new['version']}", y=1.0, fontsize=13)
    fig.tight_layout(); savefig(fig, "improvement_old_vs_new.png"); plt.close(fig)
    return summary["criteria"]


if __name__ == "__main__":
    run()
