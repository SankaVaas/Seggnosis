"""Benchmark 4: conformal coverage, and do prediction sets adapt to difficulty? (LAC vs APS vs RAPS)"""
import numpy as np
import common  # noqa: F401  (puts the repo on sys.path if seggnosis is not installed)
import seggnosis
from common import S, auroc, get_model, get_pyplot, save_json, savefig

plt = get_pyplot()
ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
SCORES = {"LAC": dict(score="lac"), "APS": dict(score="aps"),
          "RAPS (lam=0.01)": dict(score="raps", lam=0.01), "RAPS (lam=0.1)": dict(score="raps", lam=0.1)}
COLORS = {"LAC": "#1f77b4", "APS": "#ff7f0e", "RAPS (lam=0.01)": "#2ca02c", "RAPS (lam=0.1)": "#d62728"}
CASES = {"Well-trained model (clean labels + overlap ambiguity)": ("seg_A", 0.0),
         "Overconfident model (10% label noise + overlap ambiguity)": ("seg_overfit", 0.10)}


class _Scaled(__import__("torch").nn.Module):
    """User-level recipe: divide logits by a fitted temperature before conformalizing."""
    def __init__(self, m, T):
        super().__init__(); self.m, self.T = m, T
    def forward(self, x):
        return self.m(x) / self.T


def calibrate_first_experiment():
    """Does temperature-scaling an overconfident model first make conformal sets more informative?"""
    ov = get_model("seg_overfit")
    cal = [S.make_batch(20, rng=np.random.default_rng(300 + i), label_noise_p=0.10) for i in range(15)]
    T = seggnosis.TemperatureScaler().fit(ov, cal[:3]).temperature
    timgs, tm = S.make_batch(120, rng=np.random.default_rng(777), label_noise_p=0.10)
    ovl = S.make_batch(120, rng=np.random.default_rng(777), label_noise_p=0.10, return_ambiguity=True, ambiguity_kind="overlap")[2].numpy()
    y, out = tm.numpy(), {"temperature": float(T)}
    for label, model in [("raw", ov), ("temperature_scaled", _Scaled(ov, T).eval())]:
        for score, kw in [("lac", {}), ("aps", {}), ("raps", {"lam": 0.01})]:
            t = seggnosis.wrap(model, method="conformal", alpha=0.1, score=score, **kw).calibrate(cal)
            ps = np.concatenate([t.predict(timgs[i:i + 20]).raw["prediction_set"] for i in range(0, 120, 20)])
            size = ps.sum(1)
            out[f"{label}/{score}"] = {"coverage": float(np.take_along_axis(ps, y[:, None], 1)[:, 0].mean()),
                                      "mean_set_size": float(size.mean()),
                                      "overlap_auroc": auroc(size[ovl], size[~ovl]) if size.std() > 0 else 0.5}
    return out


def run():
    results, example = {}, {}
    for case, (mname, ln) in CASES.items():
        model = get_model(mname)
        cal = [S.make_batch(20, rng=np.random.default_rng(300 + i), label_noise_p=ln) for i in range(15)]  # 300 imgs
        timgs, tmasks, tamb = S.make_batch(120, rng=np.random.default_rng(777), label_noise_p=ln, return_ambiguity=True)
        y, amb = tmasks.numpy(), tamb.numpy()
        # Same seed => identical images/labels; only the returned mask differs.
        # "overlap" = the visible, predictable ambiguity (shape overlaps). The
        # scattered label-noise pixels have no visual cue, so no method can locate them.
        ovl = S.make_batch(120, rng=np.random.default_rng(777), label_noise_p=ln, return_ambiguity=True, ambiguity_kind="overlap")[2].numpy()
        results[case] = {}
        for sname, kw in SCORES.items():
            results[case][sname] = []
            for alpha in ALPHAS:
                t = seggnosis.wrap(model, method="conformal", alpha=alpha, **kw).calibrate(cal)
                ps = np.concatenate([t.predict(timgs[i:i + 20]).raw["prediction_set"] for i in range(0, 120, 20)])
                in_set = np.take_along_axis(ps, y[:, None], axis=1)[:, 0]
                size = ps.sum(1)
                results[case][sname].append({
                    "alpha": alpha, "target": 1 - alpha,
                    "coverage": float(in_set.mean()),
                    "coverage_on_ambiguous": float(in_set[amb].mean()), "coverage_on_clear": float(in_set[~amb].mean()),
                    "mean_set_size": float(size.mean()),
                    "size_on_overlap": float(size[amb].mean()), "size_off_overlap": float(size[~amb].mean()),
                    "singleton_fraction": float((size == 1).mean()),
                    "set_size_vs_ambiguity_auroc": auroc(size[amb], size[~amb]) if size.std() > 0 else 0.5,
                    "size_on_overlap": float(size[ovl].mean()), "size_off_overlap": float(size[~ovl].mean()),
                    "set_size_vs_overlap_auroc": auroc(size[ovl], size[~ovl]) if size.std() > 0 else 0.5,
                    "coverage_on_overlap": float(in_set[ovl].mean()),
                })
                if alpha == 0.1 and mname == "seg_overfit":
                    example[sname] = size[:4]
            r = results[case][sname][2]
            print(f"  {mname:11s} {sname:16s} @alpha=0.1: coverage={r['coverage']:.3f} size={r['mean_set_size']:.2f} "
                  f"(on overlaps {r['size_on_overlap']:.2f} vs elsewhere {r['size_off_overlap']:.2f}) overlap-AUROC={r['set_size_vs_overlap_auroc']:.2f}", flush=True)
        if mname == "seg_overfit":
            ex_imgs, ex_y, ex_amb = timgs[:4], y[:4], ovl[:4]

    # ---- criteria: finite-sample slack of 1 point on 120*1024 = 123k pixels ---
    allcov = [(c, s, r) for c, d in results.items() for s, rs in d.items() for r in rs]
    crit = {
        "marginal coverage >= 1 - alpha - 0.01 for every case/score/alpha": all(r["coverage"] >= r["target"] - 0.01 for _, _, r in allcov),
        "smaller alpha never shrinks sets (every case/score)": all(
            all(rs[i]["mean_set_size"] >= rs[i + 1]["mean_set_size"] - 1e-9 for i in range(len(rs) - 1)) for d in results.values() for rs in d.values()),
        "every score returns >= 1 class per pixel (singleton fraction <= 1, size >= 1)": all(r["mean_set_size"] >= 1 for _, _, r in allcov),
    }
    # exploratory (not pass/fail): which score adapts best to ambiguous pixels at alpha=0.1
    adapt = {c: {s: results[c][s][2]["set_size_vs_overlap_auroc"] for s in SCORES} for c in results}
    cf = calibrate_first_experiment()
    save_json("conformal_calibrate_first.json", cf)
    save_json("conformal.json", {"results": results, "criteria": crit, "adaptivity_auroc_alpha_0.1": adapt, "alphas": ALPHAS})

    # ---- fig 1: coverage + set size -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for c, case in enumerate(results):
        ax = axes[0, c]
        ax.plot([0.7, 1], [0.7, 1], "k--", lw=1, label="target")
        for s in SCORES:
            ax.plot([r["target"] for r in results[case][s]], [r["coverage"] for r in results[case][s]], "o-", ms=4, color=COLORS[s], label=s)
        ax.set_title(case, fontsize=9); ax.set_xlabel("Target coverage (1 - alpha)"); ax.set_ylabel("Empirical coverage")
        ax = axes[1, c]
        for s in SCORES:
            ax.plot([r["target"] for r in results[case][s]], [r["mean_set_size"] for r in results[case][s]], "o-", ms=4, color=COLORS[s], label=s)
        ax.set_xlabel("Target coverage (1 - alpha)"); ax.set_ylabel("Mean prediction-set size (of 4 classes)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Conformal prediction: coverage holds for every score; sets grow as the target rises", y=1.0)
    fig.tight_layout(); savefig(fig, "conformal_coverage_and_size.png"); plt.close(fig)

    # ---- fig 2: adaptivity at alpha = 0.1 ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, case in zip(axes, results):
        x = np.arange(len(SCORES)); names = list(SCORES)
        ax.bar(x - 0.18, [results[case][s][2]["size_off_overlap"] for s in names], 0.36, color="#7fbf7b", label="elsewhere")
        ax.bar(x + 0.18, [results[case][s][2]["size_on_overlap"] for s in names], 0.36, color="#d7301f", label="shape-overlap pixels (ambiguous, visible)")
        for k, s in enumerate(names):
            ax.text(k, max(results[case][s][2]["size_on_overlap"], results[case][s][2]["size_off_overlap"]) + 0.05,
                    f"AUROC {results[case][s][2]['set_size_vs_overlap_auroc']:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8); ax.set_title(case, fontsize=9)
    axes[0].set_ylabel("Mean set size at alpha = 0.1"); axes[0].legend(fontsize=8)
    fig.suptitle("Do prediction sets get bigger on the visibly ambiguous (shape-overlap) pixels?", y=1.02)
    savefig(fig, "conformal_adaptivity.png"); plt.close(fig)

    # ---- fig 3: set-size maps (overfit model, alpha = 0.1) ---------------
    names = list(SCORES)
    fig, axes = plt.subplots(3, 3 + len(names), figsize=(2.0 * (3 + len(names)), 6.4))
    for r in range(3):
        panels = [ex_imgs[r].permute(1, 2, 0).numpy(), ex_y[r], ex_amb[r]] + [example[n][r] for n in names]
        titles = ["Input", "Label", "Overlap (ambiguous)"] + names
        for c, (p, t) in enumerate(zip(panels, titles)):
            ax = axes[r, c]; ax.axis("off"); ax.grid(False)
            if c == 0: ax.imshow(p)
            elif c == 1: ax.imshow(p, cmap="tab10", vmin=0, vmax=9)
            elif c == 2: ax.imshow(p, cmap="gray_r")
            else: ax.imshow(p, cmap="viridis", vmin=1, vmax=4)
            if r == 0: ax.set_title(t.replace(" (", "\n("), fontsize=8)
    fig.suptitle("Prediction-set size per pixel (1 = confident, 4 = all classes), overconfident model, alpha = 0.1", y=1.0)
    savefig(fig, "conformal_set_size_maps.png"); plt.close(fig)
    return crit


if __name__ == "__main__":
    run()
