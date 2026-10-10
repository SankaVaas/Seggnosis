"""Runs one fixed set of experiments against WHICHEVER seggnosis is importable
(selected via PYTHONPATH), and writes JSON. Used to compare old vs new."""
import json, sys, time
import numpy as np
import torch
import seggnosis
from common import S, auroc, get_model, rankdata

out_path = sys.argv[1]
res = {"version": seggnosis.__version__, "file": seggnosis.__file__}
A = get_model("seg_A")
imgs, masks = S.make_batch(60, rng=np.random.default_rng(2024))
gt = masks.numpy()

# --- T1: multi-image batch (README says (B,C,H,W) is accepted) -------------------
trusted = seggnosis.wrap(A, method="mc_dropout", n_samples=10)
torch.manual_seed(0)
r = trusted.predict(imgs[:8])
mask = np.asarray(r.mask)
n_returned = 1 if mask.ndim == 2 else mask.shape[0]
accs = [float((mask == gt[i]).mean()) if mask.ndim == 2 else float((mask[i] == gt[i]).mean()) for i in range(8)]
res["T1_batch"] = {"images_in": 8, "masks_returned": n_returned, "mask_shape": list(mask.shape),
                   "per_image_accuracy_vs_own_ground_truth": accs, "mean_accuracy": float(np.mean(accs))}

# --- T2: confidence for variance-based uncertainty (tta default) -------------------
def conf_study(method, unc):
    kw = {"n_samples": 10} if method == "mc_dropout" else {}
    t = seggnosis.wrap(A, method=method, uncertainty=unc, **kw)
    conf, acc = [], []
    for i in range(60):
        rr = t.predict(imgs[i]); conf.append(float(rr.confidence)); acc.append(float((rr.mask == gt[i]).mean()))
    conf, acc = np.array(conf), np.array(acc)
    low = acc < np.median(acc)
    return {"confidence": conf.tolist(), "accuracy": acc.tolist(), "conf_min": float(conf.min()), "conf_max": float(conf.max()),
            "conf_std": float(conf.std()),
            "auroc_low_confidence_flags_bad_images": auroc(1 - conf[low], 1 - conf[~low]),
            "spearman_conf_vs_acc": float(np.corrcoef(rankdata(conf), rankdata(acc))[0, 1])}
res["T2_confidence_variance"] = conf_study("tta", "variance")
res["T2_confidence_entropy_control"] = conf_study("tta", "entropy")

# --- T3: does predict() leave the user's model untouched? --------------------------
m = get_model("seg_A"); m.eval()
x = imgs[:1]
trusted = seggnosis.wrap(m, method="mc_dropout", n_samples=5)
trusted.predict(imgs[0])
with torch.no_grad():
    d = float((m(x) - m(x)).abs().max())
res["T3_model_state"] = {"dropout_layer_in_train_mode_after_predict": bool(m.dropout.training),
                         "model_in_train_mode_after_predict": bool(m.training),
                         "two_identical_forward_passes_max_abs_diff": d}

# --- T4: OOD detector with a batch -------------------------------------------------
fit, _ = S.make_batch(120, rng=np.random.default_rng(1))
det = seggnosis.MahalanobisOOD(layer_name="conv2").fit(A, [fit[i:i + 20] for i in range(0, 120, 20)])
mixed = torch.cat([S.make_batch(4, rng=np.random.default_rng(2))[0], S.make_batch(4, rng=np.random.default_rng(3), noise_mult=6)[0]])
sc = det.score(A, mixed)
res["T4_ood_batch"] = {"images_in": 8, "scores_returned": int(np.size(sc)),
                       "scores": np.atleast_1d(sc).tolist()}

# --- T5: speed of MC dropout --------------------------------------------------------
def bench(img, n_samples, reps=9, rounds=3):
    """Best of `rounds` medians-of-`reps`: robust to noisy shared CPUs."""
    t = seggnosis.wrap(A, method="mc_dropout", n_samples=n_samples)
    t.predict(img)
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            s0 = time.perf_counter(); t.predict(img); ts.append(time.perf_counter() - s0)
        meds.append(float(np.median(ts)))
    return min(meds)
res["T5_speed_seconds"] = {f"{s}x{s}, 20 samples": bench(torch.rand(3, s, s), 20) for s in (32, 64, 128, 256)}
json.dump(res, open(out_path, "w"), indent=2)
print("done", res["version"])
