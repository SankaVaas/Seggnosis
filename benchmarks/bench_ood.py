"""Benchmark 2: does OOD detection work, and how does it degrade with mild shift?"""
import numpy as np
import torch
import common  # noqa: F401  (puts the repo on sys.path if seggnosis is not installed)
import seggnosis
from common import S, auroc, get_model, get_pyplot, save_json, savefig

plt = get_pyplot()
LAYERS = ["conv1", "conv2", "conv3"]
FAMILIES = {  # name -> (kwargs-key, severities, control value)
    "Pixel noise (x std)": ("noise_mult", [1, 1.5, 2, 3, 4, 6, 8], 1),
    "Blur (kernel size)": ("blur_k", [3, 5, 7, 9, 13], 3),
    "Color inversion (alpha)": ("color_alpha", [0.0, 0.25, 0.5, 0.75, 1.0], 0.0),
}


def baseline_scores(model, x):
    with torch.no_grad():
        p = torch.softmax(model(x), 1)
    msp = (1 - p.max(1).values).flatten(1).mean(1).numpy()                 # 1 - mean max-softmax
    ent = (-(p * torch.log(p + 1e-8)).sum(1)).flatten(1).mean(1).numpy()   # mean entropy
    return msp, ent


def run():
    model = get_model("seg_A")
    fit_imgs, _ = S.make_batch(300, rng=np.random.default_rng(42))
    loader = [fit_imgs[i:i + 20] for i in range(0, 300, 20)]
    dets = {l: seggnosis.MahalanobisOOD(layer_name=l).fit(model, loader) for l in LAYERS}

    id_test, _ = S.make_batch(150, rng=np.random.default_rng(99))
    id_scores = {l: dets[l].score(model, id_test) for l in LAYERS}
    id_msp, id_ent = baseline_scores(model, id_test)
    fpr = {l: float((id_scores[l] > dets[l].threshold).mean()) for l in LAYERS}

    def evaluate(x):
        r = {}
        for l in LAYERS:
            s = dets[l].score(model, x)
            r[l] = {"auroc": auroc(s, id_scores[l]), "flag_rate": float((s > dets[l].threshold).mean()), "scores": s}
        msp, ent = baseline_scores(model, x)
        r["MSP baseline"] = {"auroc": auroc(msp, id_msp)}
        r["entropy baseline"] = {"auroc": auroc(ent, id_ent)}
        return r

    sweeps, all_scores = {}, {}
    for fam, (key, sevs, ctrl) in FAMILIES.items():
        sweeps[fam] = {"severities": sevs, "control": ctrl, "results": []}
        for j, sv in enumerate(sevs):
            x, _ = S.make_batch(100, rng=np.random.default_rng(1000 + j), **{key: sv})
            r = evaluate(x); sweeps[fam]["results"].append({k: {a: b for a, b in v.items() if a != "scores"} for k, v in r.items()})
            all_scores[(fam, sv)] = r["conv2"]["scores"]
        print(f"  {fam}: conv2 AUROC by severity = " + ", ".join(f"{r['conv2']['auroc']:.2f}" for r in sweeps[fam]["results"]), flush=True)

    special = {}
    for name, corrupt in [("Novel shapes (ring/cross, new color)", "novel_shape"), ("Pure noise images", "pure_noise")]:
        x, _ = S.make_batch(100, rng=np.random.default_rng(7), corrupt=corrupt)
        r = evaluate(x); special[name] = {k: {a: b for a, b in v.items() if a != "scores"} for k, v in r.items()}
        all_scores[(name, 1)] = r["conv2"]["scores"]
        print(f"  {name}: " + ", ".join(f"{k}={special[name][k]['auroc']:.2f}" for k in special[name]), flush=True)

    # ---- end-to-end: attach to a wrapper and use result.is_ood -----------------
    trusted = seggnosis.wrap(model, method="tta").attach_ood_detector(dets["conv2"])
    xs_id, _ = S.make_batch(60, rng=np.random.default_rng(555))
    xs_ood, _ = S.make_batch(60, rng=np.random.default_rng(556), noise_mult=4)
    r_id, r_ood = trusted.predict(xs_id), trusted.predict(xs_ood)
    e2e = {"id_flag_rate": float(np.mean(r_id.is_ood)), "ood_flag_rate": float(np.mean(r_ood.is_ood)),
           "batch_is_ood_shape": list(np.shape(r_ood.is_ood)),
           "matches_direct_scoring": bool(np.allclose(r_ood.ood_score, dets["conv2"].score(model, xs_ood)))}
    print("  end-to-end:", e2e, flush=True)

    # ---- false-alarm rate vs. size of the fit set: in-sample vs cross-fit threshold -----
    held = S.make_batch(500, rng=np.random.default_rng(31337))[0]
    fa_sizes = [60, 120, 300, 600]
    fa = {"fit_sizes": fa_sizes, "in_sample": [], "cross_fit": []}
    for n in fa_sizes:
        xs = S.make_batch(n, rng=np.random.default_rng(42))[0]
        ld = [xs[i:i + 20] for i in range(0, n, 20)]
        for meth in ("in_sample", "cross_fit"):
            d = seggnosis.MahalanobisOOD(layer_name="conv2").fit(model, ld, threshold_method=meth)
            fa[meth].append(float((d.score(model, held) > d.threshold).mean()))
    print("  false alarms vs fit size", fa, flush=True)

    # ---- criteria ---------------------------------------------------------
    crit = {
        "cross-fit threshold: false-alarm rate <= 7% (nominal 5%) at EVERY fit size from 60 to 600": max(fa["cross_fit"]) <= 0.07,
        "cross-fit threshold is never worse than the in-sample one": all(c <= i + 1e-9 for c, i in zip(fa["cross_fit"], fa["in_sample"])),
        "false-positive rate on in-distribution data <= 10% (conv2)": fpr["conv2"] <= 0.10,
        "control (no shift) AUROC within 0.4-0.6 for every family (conv2)": all(
            0.4 <= f["results"][0]["conv2"]["auroc"] <= 0.6 for f in sweeps.values()),
        "AUROC >= 0.9 at strongest shift for every family (conv2)": all(
            f["results"][-1]["conv2"]["auroc"] >= 0.9 for f in sweeps.values()),
        "AUROC >= 0.9 on novel shapes and pure noise (conv2)": all(v["conv2"]["auroc"] >= 0.9 for v in special.values()),
        "result.is_ood end-to-end: flags >= 80% of shifted, <= 15% of normal": e2e["ood_flag_rate"] >= 0.8 and e2e["id_flag_rate"] <= 0.15,
        "is_ood batch output has one flag per image": e2e["batch_is_ood_shape"] == [60],
    }
    save_json("ood.json", {"false_alarm_vs_fit_size": fa, "fpr_in_distribution": fpr, "sweeps": sweeps, "special": special, "end_to_end": e2e, "criteria": crit})

    # ---- figure 1: AUROC vs severity --------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    styles = {"conv1": ("#1f77b4", "-o"), "conv2": ("#2ca02c", "-o"), "conv3": ("#9467bd", "-o"),
              "MSP baseline": ("#d62728", "--s"), "entropy baseline": ("#8c564b", "--^")}
    for ax, (fam, d) in zip(axes, sweeps.items()):
        for name, (c, st) in styles.items():
            lab = f"Mahalanobis @ {name}" if name.startswith("conv") else name
            ax.plot(d["severities"], [r[name]["auroc"] for r in d["results"]], st, color=c, ms=4, label=lab)
        ax.axhline(0.5, color="gray", ls=":"); ax.set_xlabel(fam); ax.set_ylim(0.3, 1.03)
    axes[0].set_ylabel("AUROC (shifted vs normal)"); axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("OOD detection vs. shift severity (leftmost point = no shift = control, expect ~0.5)", y=1.02)
    savefig(fig, "ood_auroc_vs_severity.png"); plt.close(fig)

    # ---- figure 2: score histograms ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(max(1, min(id_scores["conv2"].min(), 5))), 4.5, 40)
    ax.hist(id_scores["conv2"], bins=bins, alpha=0.7, color="#2ca02c", label="Normal (held-out)")
    for (fam, sv), lab, col in [(("Pixel noise (x std)", 3), "Noise x3", "#ff7f0e"),
                                (("Color inversion (alpha)", 0.5), "Half color-inverted", "#9467bd"),
                                (("Novel shapes (ring/cross, new color)", 1), "Novel shapes", "#d62728")]:
        ax.hist(all_scores[(fam, sv)], bins=bins, alpha=0.55, color=col, label=lab)
    ax.axvline(dets["conv2"].threshold, color="k", ls="--", label="detector threshold (95th pct of fit data)")
    ax.set_xscale("log"); ax.set_xlabel("Mahalanobis OOD score (log scale)"); ax.set_ylabel("# images"); ax.legend(fontsize=8)
    ax.set_title("OOD score distributions: normal vs shifted images")
    savefig(fig, "ood_score_histograms.png"); plt.close(fig)

    # ---- figure 3: flag rates at default threshold -----------------------
    items = [("Normal\n(false alarms)", fpr["conv2"])]
    for fam, d in sweeps.items():
        items.append((f"{fam.split(' (')[0]}\nstrongest", d["results"][-1]["conv2"]["flag_rate"]))
    for n, v in special.items():
        items.append((n.split(" (")[0].replace(" images", "") + "", v["conv2"]["flag_rate"]))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = ["#2ca02c"] + ["#d62728"] * (len(items) - 1)
    ax.bar([i[0] for i in items], [i[1] for i in items], color=cols)
    for k, i in enumerate(items): ax.text(k, i[1] + 0.02, f"{i[1]:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Fraction flagged is_ood")
    ax.set_title("Flag rate at the default threshold (green should be low, red high)")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    savefig(fig, "ood_flag_rates.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(fa_sizes, [100 * v for v in fa["in_sample"]], "o-", color="#d62728", label="in-sample threshold (before 0.6.0)")
    ax.plot(fa_sizes, [100 * v for v in fa["cross_fit"]], "o-", color="#2ca02c", label="cross-fit threshold (default now)")
    ax.axhline(5, color="k", ls="--", lw=1, label="nominal 5%")
    ax.set_xscale("log"); ax.set_xticks(fa_sizes); ax.set_xticklabels(fa_sizes)
    ax.set_xlabel("# in-distribution images used to fit the detector"); ax.set_ylabel("False alarms on fresh NORMAL images (%)")
    ax.set_title("OOD detector false-alarm rate vs. fit-set size"); ax.legend(fontsize=8)
    savefig(fig, "ood_false_alarm_vs_fit_size.png"); plt.close(fig)
    return crit


if __name__ == "__main__":
    run()
