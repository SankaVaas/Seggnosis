"""Assemble results/REPORT.md from results/data/*.json and results/user_simulation/."""
import datetime
import json
import os
import platform
import sys

import common

R = common.RESULTS


def load(name, default=None):
    path = os.path.join(common.DATA, name)
    return json.load(open(path)) if os.path.exists(path) else default


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def checklist(crit):
    return "\n".join(f"- {'PASS' if v else '**FAIL**'} -- {k}" for k, v in crit.items())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def flat(crit):
    """Accepts {name: bool} or {group: {name: bool}}."""
    out = {}
    for k, v in crit.items():
        if isinstance(v, dict):
            out.update({f"{k}: {kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out


def run():
    unc, ood, cal = load("uncertainty.json"), load("ood.json"), load("calibration.json")
    conf, cfirst, ver = load("conformal.json"), load("conformal_calibrate_first.json"), load("version_comparison.json")
    sim_path = os.path.join(R, "user_simulation", "scenarios.json")
    sim = json.load(open(sim_path)) if os.path.exists(sim_path) else None

    import seggnosis
    parts = []
    A = parts.append
    A(f"# seggnosis {seggnosis.__version__} -- validation report\n")
    A(f"Generated {datetime.date.today()} on {platform.system()} / Python {sys.version.split()[0]} "
      f"(benchmarks run against the repository checkout; the user simulation runs against the *installed wheel*).\n")
    A("> **Read this first.** Every benchmark below uses a *synthetic* segmentation task (32x32 images of overlapping "
      "shapes) with a small CNN trained on CPU. That makes ground truth available -- including which pixels are "
      "irreducibly ambiguous -- which is what lets us test whether the library's numbers are *meaningful*. It does "
      "**not** prove performance on real medical or natural images. See *What this does NOT show* at the end.\n")

    # ---------------- scoreboard --------------------------------------------------
    areas = []
    for name, c, fig in [("Uncertainty quality (mc_dropout / tta / ensemble)", unc and unc["criteria"], "uncertainty_examples.png"),
                         ("OOD detection (Mahalanobis)", ood and ood["criteria"], "ood_auroc_vs_severity.png"),
                         ("Calibration (temperature scaling)", cal and cal["criteria"], "calibration_reliability.png"),
                         ("Conformal prediction (LAC / APS / RAPS)", conf and conf["criteria"], "conformal_coverage_and_size.png"),
                         ("Fixes vs. the PyPI release v0.2.1", ver and ver["criteria"], "improvement_old_vs_new.png")]:
        if c is None:
            areas.append((name, "not run", "-"))
            continue
        f = flat(c)
        areas.append((name, f"{sum(f.values())}/{len(f)}", f"[figure](images/{fig})"))
    if sim:
        p = sum(s["status"] == "PASS" for s in sim["scenarios"])
        areas.append(("User simulation (fresh install of the built wheel)", f"{p}/{len(sim['scenarios'])}", "[scenarios](user_simulation/scenarios.md)"))
    A("## Scoreboard\n")
    A(table(["Area", "Criteria passed", "Figure"], areas))
    A("\nThe pass/fail thresholds are listed under each section; they are the author's choices, stated up front, "
      "not tuned to the results. A FAIL is a finding, not an error.\n")

    # ---------------- 1. uncertainty ----------------------------------------------
    if unc:
        r = unc["results"]
        A("## 1. Does uncertainty track real errors?\n")
        A(f"{unc['n_images']} held-out test images, {unc['n_images'] * 1024:,} pixels. "
          "'Error AUROC' = how well a pixel's uncertainty separates wrong pixels from right ones "
          "(0.5 = useless, 1.0 = perfect). 'Ambiguity AUROC' = how well it finds the pixels whose labels are "
          "randomly assigned by construction (shape overlaps + label noise) -- errors no model could avoid.\n")
        rows = []
        for key, v in r.items():
            q = v["quartile_accuracy"]
            rows.append((f"{v['method']}{' (default)' if v['is_default'] else ''}", v["uncertainty"], f"{v['error_detection_auroc']:.3f}",
                         f"{v['ambiguity_detection_auroc']:.3f}", f"{pct(q[0])} -> {pct(q[3])}", f"{v['image_conf_vs_acc_spearman']:+.2f}"))
        A(table(["method", "uncertainty", "Error AUROC", "Ambiguity AUROC", "accuracy: most -> least certain quartile", "image confidence vs accuracy (Spearman)"], rows))
        A("\n![examples](images/uncertainty_examples.png)\n")
        A("![accuracy-rejection](images/uncertainty_accuracy_rejection.png)\n")
        A("![heatmap](images/uncertainty_auroc_heatmap.png)\n")
        A("**Criteria (default uncertainty type of each method):**\n")
        A(checklist(flat(unc["criteria"])) + "\n")

    # ---------------- 2. OOD -------------------------------------------------------
    if ood:
        A("## 2. Does OOD detection separate shifted data from normal data?\n")
        A("AUROC of the detector score for 'shifted vs normal' held-out images (0.5 = chance). The leftmost severity of each "
          "family is *no shift* and is a control: it should be ~0.5. MSP / entropy are standard no-extra-model baselines "
          "(the model's own confidence); a useful OOD detector should beat them.\n")
        for fam, d in ood["sweeps"].items():
            rows = []
            for sv, res in zip(d["severities"], d["results"]):
                rows.append((sv + 0 if sv != d["control"] else f"{sv} (control)", f"{res['conv1']['auroc']:.2f}", f"{res['conv2']['auroc']:.2f}",
                             f"{res['conv3']['auroc']:.2f}", f"{res['MSP baseline']['auroc']:.2f}", f"{res['entropy baseline']['auroc']:.2f}",
                             pct(res["conv2"]["flag_rate"], 0)))
            A(f"**{fam}**\n")
            A(table(["severity", "Mahalanobis @conv1", "@conv2", "@conv3", "MSP baseline", "entropy baseline", "conv2 flagged is_ood"], rows) + "\n")
        rows = [(n, f"{v['conv1']['auroc']:.2f}", f"{v['conv2']['auroc']:.2f}", f"{v['conv3']['auroc']:.2f}",
                 f"{v['MSP baseline']['auroc']:.2f}", f"{v['entropy baseline']['auroc']:.2f}", pct(v["conv2"]["flag_rate"], 0))
                for n, v in ood["special"].items()]
        A("**Qualitatively different inputs**\n")
        A(table(["input", "@conv1", "@conv2", "@conv3", "MSP baseline", "entropy baseline", "conv2 flagged"], rows) + "\n")
        A("![auroc](images/ood_auroc_vs_severity.png)\n")
        fa = ood.get("false_alarm_vs_fit_size")
        if fa:
            A("### False alarms on normal data vs. size of the fit set (the 0.6.0 threshold fix)\n")
            A("The detector's threshold is the 95th percentile of in-distribution scores, so ~5% of normal images should be flagged. "
              "Before 0.6.0 those scores were computed on the same images the Gaussian was fitted on (biased low); 0.6.0 cross-fits them.\n")
            A(table(["# fit images"] + [str(n) for n in fa["fit_sizes"]],
                    [["in-sample (old)"] + [pct(x) for x in fa["in_sample"]], ["cross-fit (new default)"] + [pct(x) for x in fa["cross_fit"]]]) + "\n")
            A("![fa](images/ood_false_alarm_vs_fit_size.png)\n")
        A("![flags](images/ood_flag_rates.png)\n")
        A("**Criteria:**\n")
        A(checklist(ood["criteria"]) + "\n")

    # ---------------- 3. calibration -----------------------------------------------
    if cal:
        A("## 3. Does temperature scaling fix overconfidence (and do no harm otherwise)?\n")
        A("The 'overconfident' model has 100% training accuracy on 32 noisy-label images and much lower accuracy on new data -- the usual "
          "way deep nets end up overconfident. ECE = Expected Calibration Error (lower is better).\n")
        rows = [(n.split(" (")[0] if "(" in n else n, pct(v["accuracy"]), pct(v["mean_confidence_raw"]), f"{v['temperature']:.2f}",
                 f"{v['ece_raw']:.4f}", f"{v['ece_scaled']:.4f}", f"{v['ece_reduction_pct']:+.0f}%", f"{v['nll_raw']:.3f} -> {v['nll_scaled']:.3f}")
                for n, v in cal["results"].items()]
        A(table(["case", "accuracy", "mean confidence (raw)", "learned T", "ECE raw", "ECE scaled", "ECE reduction", "NLL"], rows) + "\n")
        A("![cal](images/calibration_reliability.png)\n")
        A("*(isolated points at the far-left of a curve are bins containing only a handful of pixels.)*\n")
        A("**Criteria:**\n")
        A(checklist(cal["criteria"]) + "\n")

    # ---------------- 4. conformal --------------------------------------------------
    if conf:
        A("## 4. Conformal prediction: coverage, and do prediction sets adapt?\n")
        A("At target coverage 90% (alpha = 0.1). 'Overlap AUROC' = how well set size separates the *visibly ambiguous* "
          "shape-overlap pixels from the rest (0.5 = set size carries no information; a plain singleton prediction scores 0.5 by definition).\n")
        for case, d in conf["results"].items():
            rows = []
            for s, rs in d.items():
                x = rs[conf["alphas"].index(0.1)]
                rows.append((s, pct(x["coverage"]), f"{x['mean_set_size']:.2f}", pct(x["singleton_fraction"], 0),
                             f"{x['size_on_overlap']:.2f} vs {x['size_off_overlap']:.2f}", f"{x['set_size_vs_overlap_auroc']:.2f}"))
            A(f"**{case}**\n")
            A(table(["score", "coverage (target 90%)", "mean set size (of 4)", "singleton pixels", "set size: overlap vs elsewhere", "overlap AUROC"], rows) + "\n")
        A("![cov](images/conformal_coverage_and_size.png)\n")
        A("![adapt](images/conformal_adaptivity.png)\n")
        A("![maps](images/conformal_set_size_maps.png)\n")
        if cfirst:
            A("### Does calibrating the model first make the sets more informative?\n")
            A(f"Overconfident model, temperature T={cfirst['temperature']:.2f}, alpha=0.1:\n")
            rows = [(k.replace("/", " + "), pct(v["coverage"]), f"{v['mean_set_size']:.2f}", f"{v['overlap_auroc']:.2f}")
                    for k, v in cfirst.items() if isinstance(v, dict)]
            A(table(["model + score", "coverage", "mean set size", "overlap AUROC"], rows) + "\n")
        A("**Criteria:**\n")
        A(checklist(conf["criteria"]) + "\n")

    # ---------------- 5. old vs new -------------------------------------------------
    if ver:
        o, n = ver["old"], ver["new"]
        A(f"## 5. What changed since the PyPI release v{ver['old_version']} (-> v{ver['new_version']})\n")
        A("Identical experiments, identical model weights, run against both versions.\n")
        rows = [
            ("Batch of 8 images -> results returned", o["T1_batch"]["masks_returned"], n["T1_batch"]["masks_returned"]),
            ("Mean accuracy of returned mask vs each image's own ground truth", pct(o["T1_batch"]["mean_accuracy"], 0), pct(n["T1_batch"]["mean_accuracy"], 0)),
            ("Dropout layer left in train mode after predict()", o["T3_model_state"]["dropout_layer_in_train_mode_after_predict"], n["T3_model_state"]["dropout_layer_in_train_mode_after_predict"]),
            ("Max |difference| between two identical forward passes afterwards", f"{o['T3_model_state']['two_identical_forward_passes_max_abs_diff']:.3f}", f"{n['T3_model_state']['two_identical_forward_passes_max_abs_diff']:.3f}"),
            ("OOD scores returned for a batch of 8", o["T4_ood_batch"]["scores_returned"], n["T4_ood_batch"]["scores_returned"]),
            ("Spread (std) of confidence for TTA + variance", f"{o['T2_confidence_variance']['conf_std']:.5f}", f"{n['T2_confidence_variance']['conf_std']:.5f}"),
        ]
        A(table(["", f"v{ver['old_version']} (old)", f"v{ver['new_version']} (new)"], rows) + "\n")
        A("**Speed of mc_dropout predict() on one CPU core (ms, best of 3 medians):**\n")
        A(table(["input"] + ["old", "new", "old/new"],
                [[k, f"{1000 * o['T5_speed_seconds'][k]:.0f}", f"{1000 * n['T5_speed_seconds'][k]:.0f}", f"{ver['speedups'][k]:.2f}x"] for k in o["T5_speed_seconds"]]) + "\n")
        A("![cmp](images/improvement_old_vs_new.png)\n")
        A("**Criteria:**\n")
        A(checklist(ver["criteria"]) + "\n")

    # ---------------- 6. user simulation -------------------------------------------------
    if sim:
        A("## 6. What a user sees after `pip install` (clean venv, installed from the built wheel)\n")
        A(f"seggnosis {sim['version']} imported from `{sim['location']}`; Python {sim['python']}, torch {sim['torch']}, {sim['platform']}.\n")
        A(table(["ID", "Scenario", "Result", "Observed"],
                [(s["id"], s["title"], s["status"] if s["status"] == "PASS" else "**FAIL**", s["observed"].replace("|", "/")[:400].replace("\n", " "))
                 for s in sim["scenarios"]]) + "\n")
        for img in ("user_mask_comparison.png", "user_overlay.png", "user_reliability.png"):
            if os.path.exists(os.path.join(R, "user_simulation", "images", img)):
                A(f"![{img}](user_simulation/images/{img})\n")
        A("Full transcript: [user_simulation/transcript.txt](user_simulation/transcript.txt)\n")

    # ---------------- limitations --------------------------------------------------------
    A("## What this does NOT show\n")
    A("- **No real data.** Everything is a synthetic shapes task. Real images have richer structure, real distribution shifts "
      "(scanner, site, protocol) and real annotation noise. Run `benchmarks/run_all.py`-style checks on a held-out slice of your "
      "own data before relying on any number here.\n"
      "- **Tiny model, CPU only.** No GPU timing was measured; the batched-replication speed behaviour on CUDA is untested.\n"
      "- **OOD detection** was tested on covariate shift (noise, blur, color) and one novel-shape case. Subtle semantic shift "
      "on real data is the hard case and is not covered.\n"
      "- **3-D volumes** are only smoke-tested (shapes and API), not validated for uncertainty quality.\n"
      "- **Conformal coverage** is a *pooled-over-pixels* guarantee (neighbouring pixels are not exchangeable); per-image coverage varies.\n"
      "- Thresholds in the criteria are the author's choices; different reasonable choices would change some PASS/FAIL marks.\n")

    path = os.path.join(R, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  wrote {os.path.relpath(path, common.REPO)}")


if __name__ == "__main__":
    run()
