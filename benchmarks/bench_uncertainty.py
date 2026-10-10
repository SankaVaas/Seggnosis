"""Benchmark 1: does uncertainty actually track real errors?"""
import numpy as np
import torch
import common  # noqa: F401  (puts the repo on sys.path if seggnosis is not installed)
import seggnosis
from common import S, auroc, get_model, get_pyplot, rankdata, save_json, savefig

plt = get_pyplot()
DEFAULTS = {"mc_dropout": "entropy", "tta": "variance", "ensemble": "mutual_information"}
UNCERTAINTIES = ["entropy", "variance", "mutual_information"]
COLORS = {"mc_dropout": "#1f77b4", "tta": "#ff7f0e", "ensemble": "#2ca02c"}


def build(method, unc, A, B, C):
    kw = {"n_samples": 20} if method == "mc_dropout" else {}
    if method == "ensemble":
        kw["models"] = [B, C]
    return seggnosis.wrap(A, method=method, uncertainty=unc, **kw)


def predict_all(trusted, imgs, bs=10):
    out = {"mask": [], "umap": [], "conf": [], "probs": []}
    for i in range(0, len(imgs), bs):
        r = trusted.predict(imgs[i:i + bs])
        out["mask"].append(r.mask); out["umap"].append(r.uncertainty_map)
        out["conf"].append(r.confidence); out["probs"].append(r.probs)
    return {k: np.concatenate(v) for k, v in out.items()}


def run():
    A, B, C = get_model("seg_A"), get_model("seg_B"), get_model("seg_C")
    imgs, masks, amb = S.make_batch(80, rng=np.random.default_rng(2024), return_ambiguity=True)
    gt, amb = masks.numpy(), amb.numpy()
    results, store = {}, {}

    for method in DEFAULTS:
        for unc in UNCERTAINTIES:
            out = predict_all(build(method, unc, A, B, C), imgs)
            correct = out["mask"] == gt
            u = out["umap"].ravel()
            order = np.argsort(u, kind="stable")
            q = [float(c.mean()) for c in np.array_split(correct.ravel()[order], 4)]
            img_acc = correct.reshape(len(imgs), -1).mean(1)
            conf = out["conf"]
            rho = float(np.corrcoef(rankdata(conf), rankdata(img_acc))[0, 1])
            half = len(imgs) // 2
            by_conf = np.argsort(-conf)
            results[f"{method}/{unc}"] = {
                "method": method, "uncertainty": unc, "is_default": DEFAULTS[method] == unc,
                "pixel_accuracy": float(correct.mean()),
                "error_detection_auroc": auroc(u[~correct.ravel()], u[correct.ravel()]),
                "ambiguity_detection_auroc": auroc(u[amb.ravel()], u[~amb.ravel()]),
                "quartile_accuracy": q,
                "image_conf_vs_acc_spearman": rho,
                "acc_top_half_confident_images": float(img_acc[by_conf[:half]].mean()),
                "acc_bottom_half_confident_images": float(img_acc[by_conf[half:]].mean()),
                "confidence_range": [float(conf.min()), float(conf.max())],
            }
            store[(method, unc)] = (out, correct)
            print(f"  {method:10s}/{unc:19s} err-AUROC={results[f'{method}/{unc}']['error_detection_auroc']:.3f} "
                  f"Q1->Q4 acc {q[0]:.3f}->{q[3]:.3f}", flush=True)

    # ---- criteria -------------------------------------------------------
    crit = {}
    for key, r in results.items():
        if r["is_default"]:
            crit[key] = {
                "error_detection_auroc >= 0.75": r["error_detection_auroc"] >= 0.75,
                "Q4 accuracy at least 5 points below Q1": r["quartile_accuracy"][0] - r["quartile_accuracy"][3] >= 0.05,
                "confident images more accurate than unconfident": r["acc_top_half_confident_images"] > r["acc_bottom_half_confident_images"],
            }
    save_json("uncertainty.json", {"results": results, "criteria": crit, "n_images": len(imgs),
                                   "model_pixel_accuracy_ref": results["mc_dropout/entropy"]["pixel_accuracy"]})

    # ---- figure 1: qualitative examples ---------------------------------
    out_mc, c_mc = store[("mc_dropout", "entropy")]
    err_per_img = (~c_mc).reshape(len(imgs), -1).mean(1)
    picks = np.argsort(-err_per_img)[:4]
    cols = ["Input", "Ground truth", "Prediction", "Errors", "MC-dropout\n(entropy)", "TTA\n(variance)", "Ensemble\n(mutual info)"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.1 * len(picks)))
    cmap = plt.get_cmap("tab10", 4)
    for r, i in enumerate(picks):
        panels = [imgs[i].permute(1, 2, 0).numpy(), gt[i], out_mc["mask"][i], (out_mc["mask"][i] != gt[i]),
                  store[("mc_dropout", "entropy")][0]["umap"][i], store[("tta", "variance")][0]["umap"][i],
                  store[("ensemble", "mutual_information")][0]["umap"][i]]
        for c, p in enumerate(panels):
            ax = axes[r, c]; ax.grid(False); ax.axis("off")
            if c in (1, 2): ax.imshow(p, cmap=cmap, vmin=0, vmax=3)
            elif c == 3: ax.imshow(p, cmap="gray_r")
            elif c >= 4: ax.imshow(p, cmap="magma")
            else: ax.imshow(p)
            if r == 0: ax.set_title(cols[c], fontsize=9)
    fig.suptitle("Uncertainty concentrates where the model is wrong (most-error test images shown)", y=1.0)
    savefig(fig, "uncertainty_examples.png"); plt.close(fig)

    # ---- figure 2: accuracy-rejection curves ----------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fracs = np.linspace(0.1, 1.0, 19)
    c_all = c_mc.ravel()
    for method, unc in DEFAULTS.items():
        out, correct = store[(method, unc)]
        o = np.argsort(out["umap"].ravel(), kind="stable"); cs = correct.ravel()[o]
        ax.plot(fracs, [cs[: int(f * len(cs))].mean() for f in fracs], marker="o", ms=3,
                color=COLORS[method], label=f"{method} ({unc})")
    oracle = np.sort(c_all)[::-1]
    ax.plot(fracs, [oracle[: int(f * len(oracle))].mean() for f in fracs], "k--", lw=1, label="oracle (knows the errors)")
    ax.axhline(c_all.mean(), color="gray", ls=":", label="no rejection / random")
    ax.set_xlabel("Fraction of pixels kept (most-confident first)"); ax.set_ylabel("Accuracy on kept pixels")
    ax.set_title("Accuracy-rejection: dropping uncertain pixels raises accuracy"); ax.legend(fontsize=8, loc="lower left")
    savefig(fig, "uncertainty_accuracy_rejection.png"); plt.close(fig)

    # ---- figure 3: quartile bars ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); w = 0.26
    for k, (method, unc) in enumerate(DEFAULTS.items()):
        q = results[f"{method}/{unc}"]["quartile_accuracy"]
        ax.bar(np.arange(4) + (k - 1) * w, q, w, color=COLORS[method], label=method)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\n(most certain)", "Q2", "Q3", "Q4\n(least certain)"])
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Pixel accuracy"); ax.legend(loc="lower left")
    ax.set_title("Accuracy by uncertainty quartile")
    savefig(fig, "uncertainty_quartiles.png"); plt.close(fig)

    # ---- figure 4: AUROC heatmaps -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in [(axes[0], "error_detection_auroc", "AUROC: uncertainty -> pixel is WRONG"),
                           (axes[1], "ambiguity_detection_auroc", "AUROC: uncertainty -> pixel is truly AMBIGUOUS")]:
        M = np.array([[results[f"{m}/{u}"][key] for u in UNCERTAINTIES] for m in DEFAULTS])
        im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis"); ax.grid(False)
        ax.set_xticks(range(3)); ax.set_xticklabels(["entropy", "variance", "mutual\ninformation"])
        ax.set_yticks(range(3)); ax.set_yticklabels(list(DEFAULTS)); ax.set_title(title, fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.8 else "k", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, "uncertainty_auroc_heatmap.png"); plt.close(fig)

    # ---- figure 5: image-level confidence vs accuracy ------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (method, unc) in zip(axes, DEFAULTS.items()):
        out, correct = store[(method, unc)]
        acc = correct.reshape(len(imgs), -1).mean(1)
        ax.scatter(out["conf"], acc, s=14, color=COLORS[method], alpha=0.8)
        ax.set_title(f"{method} ({unc})\nSpearman={results[f'{method}/{unc}']['image_conf_vs_acc_spearman']:+.2f}", fontsize=9)
        ax.set_xlabel("Result.confidence")
    axes[0].set_ylabel("Actual image accuracy")
    savefig(fig, "uncertainty_image_confidence.png"); plt.close(fig)
    return results, crit


if __name__ == "__main__":
    run()
