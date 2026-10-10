"""
User simulation: behave like someone who just ran `pip install seggnosis`
and is following the README with their OWN model and data.

Run it with the Python of an environment where seggnosis was installed from
the built wheel (NOT from the repo), from a directory that does not contain
the repo, e.g. via benchmarks/run_all.py. It writes:

    <out>/transcript.txt     everything printed
    <out>/scenarios.json     one record per scenario (status, expected, observed)
    <out>/scenarios.md       the same as a readable table
    <out>/images/*.png       figures produced with seggnosis.visualize

usage: python user_simulation.py <stage_dir_with_weights_and_synthetic_shapes> <out_dir>
"""
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

stage, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, stage)
os.makedirs(os.path.join(out, "images"), exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import seggnosis  # noqa: E402  (must come from site-packages)
import synthetic_shapes as S  # noqa: E402  (the "user's" own data/model code)

LOG = open(os.path.join(out, "transcript.txt"), "w", encoding="utf-8")
RECORDS = []


def say(*a):
    print(*a)
    print(*a, file=LOG)
    LOG.flush()


def scenario(sid, title, expected):
    def deco(fn):
        say(f"\n=== {sid}: {title}")
        say(f"    expected: {expected}")
        t0 = time.perf_counter()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                observed = fn()
            status = "PASS"
        except Exception as e:  # noqa: BLE001
            observed = f"{type(e).__name__}: {e}"
            status = "FAIL"
            say(traceback.format_exc())
        captured = buf.getvalue().strip()
        if captured:
            say("    " + captured.replace("\n", "\n    "))
        say(f"    observed: {observed}")
        say(f"    -> {status}  ({time.perf_counter() - t0:.1f}s)")
        RECORDS.append({"id": sid, "title": title, "expected": expected, "observed": str(observed), "status": status})
        return fn
    return deco


def load(name, **kw):
    m = S.SynthSegNet(**kw)
    m.load_state_dict(torch.load(os.path.join(stage, f"{name}.pt"), map_location="cpu"))
    return m.eval()


# The user's own trained models (here: weights trained on the synthetic task).
model = load("seg_A")
model_b, model_c = load("seg_B"), load("seg_C")
overfit = load("seg_overfit", width=64, dropout_p=0.0)
imgs, gts = S.make_batch(40, rng=np.random.default_rng(11))
image = imgs[0]


# ---------------------------------------------------------------------------
@scenario("S01", "Installation looks like a normal pip install",
          "imported from site-packages (not a source checkout); version matches pip metadata; `pip check` clean; import is quick")
def _():
    f = seggnosis.__file__
    assert "site-packages" in f or "dist-packages" in f, f"imported from {f}"
    assert os.path.realpath(f).startswith(os.path.realpath(sys.prefix)), f"not inside this environment: {f}"
    direct = importlib.metadata.distribution("seggnosis").read_text("direct_url.json") or ""
    assert '"editable": true' not in direct.replace(" ", "").replace('"editable":true', '"editable": true'), "editable install, not the wheel"
    assert importlib.metadata.version("seggnosis") == seggnosis.__version__
    chk = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stdout
    t = subprocess.run([sys.executable, "-c", "import time;t=time.time();import seggnosis;print(round(time.time()-t,2))"],
                       capture_output=True, text=True).stdout.strip()
    print("public API:", ", ".join(seggnosis.__all__))
    return f"v{seggnosis.__version__} from {os.path.dirname(f)}; pip check clean; cold import incl. torch {t}s"


@scenario("S02", "README quickstart: wrap(...).predict(image)",
          "Result has mask/probs/uncertainty_map/confidence with sane shapes, ranges and summary()")
def _():
    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20)
    r = trusted.predict(image)
    assert r.mask.shape == (32, 32) and r.probs.shape == (4, 32, 32) and r.uncertainty_map.shape == (32, 32)
    assert np.allclose(r.probs.sum(0), 1, atol=1e-4), "probs do not sum to 1"
    assert 0 <= r.confidence <= 1 and r.uncertainty_map.min() >= 0 and r.is_ood is None
    print(r.summary())
    return f"mask{r.mask.shape} probs{r.probs.shape}; {r.summary()}"


@scenario("S03", "Every method x every uncertainty type runs",
          "9/9 combinations return finite, in-range results")
def _():
    n = 0
    for method in ("mc_dropout", "tta", "ensemble"):
        for unc in ("entropy", "variance", "mutual_information"):
            kw = {"n_samples": 8} if method == "mc_dropout" else {}
            if method == "ensemble":
                kw["models"] = [model_b, model_c]
            r = seggnosis.wrap(model, method=method, uncertainty=unc, **kw).predict(image)
            assert np.isfinite(r.uncertainty_map).all() and 0 <= r.confidence <= 1
            assert r.uncertainty_map.min() > -1e-5
            n += 1
    return f"{n}/9 combinations OK"


@scenario("S04", "Batch of images", "(B,H,W) mask and a length-B confidence array; TTA batch == per-image calls")
def _():
    t = seggnosis.wrap(model, method="tta")
    r = t.predict(imgs[:5])
    assert r.mask.shape == (5, 32, 32) and r.confidence.shape == (5,)
    singles = np.stack([t.predict(imgs[i]).probs for i in range(5)])
    assert np.allclose(singles, r.probs, atol=1e-5)
    return f"mask{r.mask.shape} confidence{r.confidence.shape}; identical to per-image predictions"


@scenario("S05", "3-D volumes (spatial_dims=3)", "all three methods work on (C,D,H,W) and (B,C,D,H,W) volumes")
def _():
    class Vol(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.dp, self.c2 = nn.Conv3d(1, 8, 3, padding=1), nn.Dropout3d(0.3), nn.Conv3d(8, 3, 1)
        def forward(self, x):
            return self.c2(self.dp(torch.relu(self.c1(x))))
    v1, v2 = Vol().eval(), Vol().eval()
    vol = torch.rand(1, 8, 16, 16)
    shapes = []
    for method, kw in [("mc_dropout", {"n_samples": 4}), ("tta", {}), ("ensemble", {"models": [v2]})]:
        t = seggnosis.wrap(v1, method=method, spatial_dims=3, **kw)
        r = t.predict(vol)
        assert r.mask.shape == (8, 16, 16) and r.probs.shape == (3, 8, 16, 16)
        rb = t.predict(torch.rand(2, 1, 8, 16, 16))
        assert rb.mask.shape == (2, 8, 16, 16)
        shapes.append(method)
    return f"{', '.join(shapes)} OK on single and batched volumes"


@scenario("S06", "OOD detection through result.is_ood",
          "normal images rarely flagged (<=20%), 4x-noise images almost always (>=80%); save/load gives identical scores")
def _():
    fit, _ = S.make_batch(120, rng=np.random.default_rng(1))
    det = seggnosis.MahalanobisOOD(layer_name="conv2").fit(model, [fit[i:i + 20] for i in range(0, 120, 20)])
    trusted = seggnosis.wrap(model, method="tta").attach_ood_detector(det)
    normal = trusted.predict(S.make_batch(60, rng=np.random.default_rng(2))[0])
    shifted = trusted.predict(S.make_batch(60, rng=np.random.default_rng(3), noise_mult=4)[0])
    fa, tp = float(np.mean(normal.is_ood)), float(np.mean(shifted.is_ood))
    assert fa <= 0.20 and tp >= 0.80, (fa, tp)
    path = os.path.join(tempfile.mkdtemp(), "det.npz")
    det.save(path)
    again = seggnosis.MahalanobisOOD.load(path)
    x = S.make_batch(6, rng=np.random.default_rng(4), noise_mult=3)[0]
    assert np.allclose(det.score(model, x), again.score(model, x))
    return f"false alarms {fa:.0%}, shifted detected {tp:.0%}; save/load round-trip identical"


@scenario("S07", "Temperature scaling on an overconfident model",
          "ECE drops by >=30%, accuracy unchanged, save/load round-trips")
def _():
    cx, cy = S.make_batch(60, rng=np.random.default_rng(6), label_noise_p=0.1)
    tx, ty = S.make_batch(100, rng=np.random.default_rng(7), label_noise_p=0.1)
    sc = seggnosis.TemperatureScaler().fit(overfit, [(cx, cy)])
    with torch.no_grad():
        lg = overfit(tx)
        raw, cal = torch.softmax(lg, 1), sc.apply(lg)
    e0 = seggnosis.expected_calibration_error(raw.numpy(), ty.numpy())
    e1 = seggnosis.expected_calibration_error(cal.numpy(), ty.numpy())
    assert (e0 - e1) / e0 >= 0.30 and (raw.argmax(1) == cal.argmax(1)).all()
    p = os.path.join(tempfile.mkdtemp(), "t.json")
    sc.save(p)
    assert abs(seggnosis.TemperatureScaler.load(p).temperature - sc.temperature) < 1e-9
    # figure for the user's report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    seggnosis.visualize.plot_reliability_diagram(raw.numpy(), ty.numpy(), ax=ax[0]); ax[0].set_title(f"Before (ECE={e0:.3f})")
    seggnosis.visualize.plot_reliability_diagram(cal.numpy(), ty.numpy(), ax=ax[1]); ax[1].set_title(f"After scaling (ECE={e1:.3f})")
    fig.savefig(os.path.join(out, "images", "user_reliability.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    return f"T={sc.temperature:.2f}; ECE {e0:.4f} -> {e1:.4f} ({100 * (e0 - e1) / e0:.0f}% lower)"


@scenario("S08", "Conformal prediction (LAC / APS / RAPS)",
          "coverage >= target - 1pt for all three scores; helpful error if predict() is called before calibrate()")
def _():
    cal = [S.make_batch(20, rng=np.random.default_rng(300 + i)) for i in range(10)]
    ti, tm = S.make_batch(80, rng=np.random.default_rng(777))
    rows = []
    for score in ("lac", "aps", "raps"):
        t = seggnosis.wrap(model, method="conformal", alpha=0.1, score=score)
        try:
            t.predict(image)
            raise AssertionError("predict before calibrate did not raise")
        except RuntimeError as e:
            msg = str(e)
        t.calibrate(cal)
        ps = np.concatenate([t.predict(ti[i:i + 20]).raw["prediction_set"] for i in range(0, 80, 20)])
        cov = float(np.take_along_axis(ps, tm.numpy()[:, None], 1).mean())
        assert cov >= 0.9 - 0.01, (score, cov)
        rows.append(f"{score}: cov {cov:.3f}, size {ps.sum(1).mean():.2f}")
    return "; ".join(rows) + f" | error before calibrate: \"{msg[:70]}...\""


@scenario("S09", "Whole-image classification API", "classification() works for all methods incl. OOD attachment")
def _():
    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.bn, self.dp = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Dropout(0.3)
            self.fc = nn.Linear(8, 5)
        def forward(self, x):
            return self.fc(self.dp(torch.relu(self.bn(self.conv(x)))).mean((2, 3)))
    c1, c2 = Clf().eval(), Clf().eval()
    outs = []
    for method, kw in [("mc_dropout", {"n_samples": 8}), ("tta", {}), ("ensemble", {"models": [c2]})]:
        r = seggnosis.classification(c1, method=method, **kw).predict(image)
        assert isinstance(r.predicted_class, int) and r.probs.shape == (5,) and 0 <= r.confidence <= 1
        outs.append(f"{method}->class {r.predicted_class}")
    det = seggnosis.MahalanobisOOD(layer_name="conv").fit(c1, [torch.rand(4, 3, 32, 32) for _ in range(8)])
    r = seggnosis.classification(c1, method="tta").attach_ood_detector(det).predict(image)
    assert r.is_ood in (True, False)
    return "; ".join(outs) + "; OOD attached OK"


@scenario("S10", "Mistakes produce clear, specific error messages",
          "each misuse raises ValueError/RuntimeError with a message (not a cryptic internal error)")
def _():
    det_unfit = seggnosis.MahalanobisOOD(layer_name="conv2")
    cases = [
        ("predict a 1-D tensor", lambda: seggnosis.wrap(model).predict(torch.rand(5)), ValueError),
        ("predict a 5-D tensor with spatial_dims=2", lambda: seggnosis.wrap(model).predict(torch.rand(1, 1, 3, 8, 8)), ValueError),
        ("unknown method name", lambda: seggnosis.wrap(model, method="magic"), ValueError),
        ("ensemble with a single model", lambda: seggnosis.wrap(model, method="ensemble"), ValueError),
        ("unknown uncertainty type", lambda: seggnosis.wrap(model, uncertainty="vibes"), ValueError),
        ("OOD detector with a misspelled layer", lambda: seggnosis.MahalanobisOOD(layer_name="conv99").fit(model, [imgs[:4]]), ValueError),
        ("OOD score before fit()", lambda: det_unfit.score(model, imgs[:2]), RuntimeError),
        ("conformal alpha out of range", lambda: seggnosis.wrap(model, method="conformal", alpha=1.5), ValueError),
        ("conformal unknown score", lambda: seggnosis.wrap(model, method="conformal", score="x"), ValueError),
        ("TemperatureScaler.apply before fit()", lambda: seggnosis.TemperatureScaler().apply(torch.rand(1, 4, 8, 8)), RuntimeError),
        ("classification unknown method", lambda: seggnosis.classification(model, method="nope"), ValueError),
        ("float64 image into a float32 model", lambda: seggnosis.wrap(model).predict(image.double()), RuntimeError),
        ("4-channel image into a 3-channel model", lambda: seggnosis.wrap(model).predict(torch.rand(4, 32, 32)), RuntimeError),
    ]
    lines, bad = [], []
    for name, fn, exc in cases:
        try:
            fn()
            bad.append(f"{name}: did NOT raise")
        except exc as e:
            msg = str(e).strip().splitlines()[0]
            lines.append(f"{name:48s} -> {type(e).__name__}: {msg[:110]}")
            if len(msg) < 15:
                bad.append(f"{name}: message too terse: {msg!r}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: raised {type(e).__name__} (expected {exc.__name__}): {e}")
    print("\n".join(lines))
    assert not bad, "; ".join(bad)
    return f"{len(cases)}/{len(cases)} misuse cases raise a clear error"


@scenario("S11", "The user's model is left untouched by predict()",
          "train/eval mode, parameters and gradients unchanged after every method")
def _():
    def fingerprint(m):
        h = hashlib.sha256()
        for p in m.parameters():
            h.update(p.detach().numpy().tobytes())
        return h.hexdigest()
    cal = [S.make_batch(20, rng=np.random.default_rng(5))]
    for start_train in (False, True):
        for method, kw in [("mc_dropout", {"n_samples": 4}), ("tta", {}), ("ensemble", {"models": [model_b]}),
                           ("conformal", {})]:
            m = load("seg_A")
            m.train(start_train)
            fp = fingerprint(m)
            t = seggnosis.wrap(m, method=method, **kw)
            if method == "conformal":
                t.calibrate(cal)
            t.predict(image)
            assert m.training == start_train, f"{method}: training flag changed ({start_train} -> {m.training})"
            assert m.dropout.training == start_train, f"{method}: dropout layer left in wrong mode"
            assert fingerprint(m) == fp, f"{method}: weights changed"
            assert all(p.grad is None for p in m.parameters()), f"{method}: gradients created"
    return "mode, weights and grads unchanged for 4 methods x {eval, train} starting states"


@scenario("S12", "Determinism", "TTA/ensemble reproducible; MC-dropout stochastic but reproducible under a seed")
def _():
    t = seggnosis.wrap(model, method="tta")
    assert np.array_equal(t.predict(image).probs, t.predict(image).probs)
    e = seggnosis.wrap(model, method="ensemble", models=[model_b])
    assert np.array_equal(e.predict(image).probs, e.predict(image).probs)
    mc = seggnosis.wrap(model, method="mc_dropout", n_samples=6)
    a, b = mc.predict(image).probs, mc.predict(image).probs
    assert not np.array_equal(a, b), "MC dropout produced identical outputs: dropout not active?"
    torch.manual_seed(0); a = mc.predict(image).probs
    torch.manual_seed(0); b = mc.predict(image).probs
    assert np.array_equal(a, b)
    return "tta/ensemble identical across calls; mc_dropout differs between calls, identical with torch.manual_seed"


@scenario("S13", "Documentation is present", "every public name has a docstring")
def _():
    missing = [n for n in seggnosis.__all__ if n != "__version__" and not (inspect.getdoc(getattr(seggnosis, n)) or "").strip()]
    assert not missing, f"no docstring: {missing}"
    return f"all {len(seggnosis.__all__) - 1} public names documented"


@scenario("S14", "Visualization helpers produce figures",
          "overlay_uncertainty / plot_mask_comparison / reliability diagram run and save")
def _():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seggnosis.visualize import overlay_uncertainty, plot_mask_comparison
    r = seggnosis.wrap(model, method="ensemble", models=[model_b, model_c]).predict(imgs[3])
    rgb = imgs[3].permute(1, 2, 0).numpy()
    fig = plot_mask_comparison(rgb, r.mask, r.uncertainty_map, ground_truth=gts[3].numpy())
    fig.savefig(os.path.join(out, "images", "user_mask_comparison.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    overlay_uncertainty(rgb, r.uncertainty_map, ax=ax)
    fig.savefig(os.path.join(out, "images", "user_overlay.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    return "user_mask_comparison.png, user_overlay.png written"


@scenario("S15", "Typical latency on CPU (informational)", "completes; numbers recorded")
def _():
    rows = []
    for size in (32, 64):
        x = torch.rand(3, size, size)
        for method, kw in [("mc_dropout", {"n_samples": 20}), ("tta", {}), ("ensemble", {"models": [model_b, model_c]})]:
            t = seggnosis.wrap(model, method=method, **kw)
            t.predict(x)
            ts = []
            for _ in range(7):
                s = time.perf_counter(); t.predict(x); ts.append(time.perf_counter() - s)
            rows.append(f"{size}x{size} {method}: {1000 * np.median(ts):.0f} ms")
    print("\n".join(rows))
    return f"cuda available: {torch.cuda.is_available()}; " + "; ".join(rows)


# ---------------------------------------------------------------------------
n_pass = sum(r["status"] == "PASS" for r in RECORDS)
say(f"\n\nSUMMARY: {n_pass}/{len(RECORDS)} scenarios passed  (seggnosis {seggnosis.__version__} from {seggnosis.__file__})")
json.dump({"version": seggnosis.__version__, "location": seggnosis.__file__, "python": sys.version.split()[0],
           "torch": torch.__version__, "platform": sys.platform, "scenarios": RECORDS},
          open(os.path.join(out, "scenarios.json"), "w"), indent=2)
with open(os.path.join(out, "scenarios.md"), "w", encoding="utf-8") as f:
    f.write(f"# User-simulation scenarios (seggnosis {seggnosis.__version__}, installed from the built wheel)\n\n")
    f.write(f"**{n_pass}/{len(RECORDS)} passed.**  Python {sys.version.split()[0]}, torch {torch.__version__}, {sys.platform}.\n\n")
    f.write("| ID | Scenario | Result | What was observed |\n|---|---|---|---|\n")
    for r in RECORDS:
        obs = r["observed"].replace("|", "\\|").replace("\n", "<br>")
        f.write(f"| {r['id']} | {r['title']} | {'PASS' if r['status'] == 'PASS' else '**FAIL**'} | {obs} |\n")
sys.exit(0 if n_pass == len(RECORDS) else 1)
