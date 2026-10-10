"""Benchmark 3: does temperature scaling fix overconfidence (and do no harm otherwise)?"""
import numpy as np
import torch
import common  # noqa: F401  (puts the repo on sys.path if seggnosis is not installed)
import seggnosis
from common import S, get_model, get_pyplot, save_json, savefig

plt = get_pyplot()


def nll(probs, y):
    p = probs.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8)
    return float(-p.log().mean())


def evaluate(name, model, label_noise, shift=None):
    cx, cy = S.make_batch(60, rng=np.random.default_rng(6), label_noise_p=label_noise)
    tx, ty = S.make_batch(100, rng=np.random.default_rng(7), label_noise_p=label_noise, **(shift or {}))
    scaler = seggnosis.TemperatureScaler().fit(model, [(cx, cy)])
    with torch.no_grad():
        logits = model(tx)
        raw, cal = torch.softmax(logits, 1), scaler.apply(logits)
    y = ty
    out = {
        "accuracy": float((raw.argmax(1) == y).float().mean()),
        "accuracy_after_scaling": float((cal.argmax(1) == y).float().mean()),
        "mean_confidence_raw": float(raw.max(1).values.mean()),
        "mean_confidence_scaled": float(cal.max(1).values.mean()),
        "temperature": float(scaler.temperature),
        "ece_raw": seggnosis.expected_calibration_error(raw.numpy(), y.numpy()),
        "ece_scaled": seggnosis.expected_calibration_error(cal.numpy(), y.numpy()),
        "nll_raw": nll(raw, y), "nll_scaled": nll(cal, y),
    }
    out["ece_reduction_pct"] = 100 * (out["ece_raw"] - out["ece_scaled"]) / max(out["ece_raw"], 1e-9)
    curves = {k: seggnosis.reliability_curve(p.numpy(), y.numpy()) for k, p in [("raw", raw), ("scaled", cal)]}
    print(f"  {name}: ECE {out['ece_raw']:.4f} -> {out['ece_scaled']:.4f}  (T={out['temperature']:.2f}, "
          f"conf {out['mean_confidence_raw']:.3f}->{out['mean_confidence_scaled']:.3f}, acc {out['accuracy']:.3f})", flush=True)
    return out, curves


def run():
    A, OV = get_model("seg_A"), get_model("seg_overfit")
    cases = {
        "Well-trained model (seg_A), clean data": (A, 0.0, None),
        "Overconfident model (memorized 32 noisy images), same-distribution data": (OV, 0.10, None),
        "Overconfident model under covariate shift (3x pixel noise)": (OV, 0.10, {"noise_mult": 3.0}),
    }
    results, curves = {}, {}
    for name, (m, ln, shift) in cases.items():
        results[name], curves[name] = evaluate(name, m, ln, shift)

    r_a, r_ov, r_sh = list(results.values())
    crit = {
        "overconfident model: scaling cuts ECE by >= 30%": r_ov["ece_reduction_pct"] >= 30,
        "overconfident model: scaling lowers NLL": r_ov["nll_scaled"] < r_ov["nll_raw"],
        "overconfident model: scaling raises T above 1 (softens confidence)": r_ov["temperature"] > 1.05,
        "already-calibrated model: scaling does no harm (ECE rises < 0.005)": r_a["ece_scaled"] <= r_a["ece_raw"] + 0.005,
        "scaling never changes accuracy (argmax preserved)": all(abs(r["accuracy"] - r["accuracy_after_scaling"]) < 1e-9 for r in results.values()),
        "under shift: scaling still reduces ECE": r_sh["ece_scaled"] < r_sh["ece_raw"],
    }
    save_json("calibration.json", {"results": results, "criteria": crit})

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    for c, (name, cv) in enumerate(curves.items()):
        for r_, (k, lab, col) in enumerate([("raw", "Raw softmax", "#d62728"), ("scaled", "After temperature scaling", "#2ca02c")]):
            ax = axes[r_, c]
            ctr, acc, conf = cv[k]
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
            ax.plot(conf, acc, "o-", color=col, label=lab)
            ax.fill_between(conf, acc, conf, color=col, alpha=0.15)
            e = results[name]["ece_raw" if k == "raw" else "ece_scaled"]
            ax.set_title(f"{lab}\nECE = {e:.4f}", fontsize=9)
            if r_ == 1: ax.set_xlabel("Confidence")
            if c == 0: ax.set_ylabel("Accuracy")
        axes[0, c].text(0.5, 1.28, name.replace(" (", "\n("), transform=axes[0, c].transAxes, ha="center", fontsize=9, fontweight="bold")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.subplots_adjust(top=0.82)
    savefig(fig, "calibration_reliability.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = ["Well-trained\n(clean)", "Overconfident\n(same dist.)", "Overconfident\n(shifted)"]
    x = np.arange(3)
    for ax, (k1, k2, ttl) in zip(axes, [("ece_raw", "ece_scaled", "Expected Calibration Error (lower = better)"),
                                         ("nll_raw", "nll_scaled", "Negative log-likelihood (lower = better)")]):
        ax.bar(x - 0.18, [r[k1] for r in results.values()], 0.36, color="#d62728", label="raw")
        ax.bar(x + 0.18, [r[k2] for r in results.values()], 0.36, color="#2ca02c", label="temperature-scaled")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8)
    savefig(fig, "calibration_before_after.png"); plt.close(fig)
    return crit


if __name__ == "__main__":
    run()
