"""Shared helpers for the seggnosis validation suite."""
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(HERE, ".cache")
RESULTS = os.environ.get("SEGGNOSIS_RESULTS_DIR", os.path.join(REPO, "results"))
IMAGES = os.path.join(RESULTS, "images")
DATA = os.path.join(RESULTS, "data")
# Work from a plain repo checkout: use the repo's seggnosis only if none is importable
# (never override an explicit PYTHONPATH -- the old-vs-new comparison depends on that).
try:
    import seggnosis  # noqa: F401
except ImportError:
    sys.path.insert(0, REPO)
for d in (CACHE, IMAGES, DATA):
    os.makedirs(d, exist_ok=True)

import synthetic_shapes as S  # noqa: E402

# name -> training recipe. All trained on the synthetic shapes task.
MODEL_SPECS = {
    # Well-trained models trained on a stream of fresh data (calibrated-ish).
    "seg_A": dict(seed=0, n_steps=500),
    "seg_B": dict(seed=1, n_steps=500),
    "seg_C": dict(seed=2, n_steps=500),
    # Deliberately overconfident: big, no dropout, memorizes 32 noisy-label images.
    "seg_overfit": dict(seed=10, n_steps=800, batch_size=32, width=64, dropout_p=0.0, overfit=True),
}


def _overfit_set():
    return S.make_batch(32, rng=np.random.default_rng(5), label_noise_p=0.10)


def get_model(name):
    path = os.path.join(CACHE, f"{name}.pt")
    spec = dict(MODEL_SPECS[name])
    width, dropout_p = spec.get("width", 24), spec.get("dropout_p", 0.25)
    if os.path.exists(path):
        m = S.SynthSegNet(width=width, dropout_p=dropout_p)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.eval()
    overfit = spec.pop("overfit", False)
    if overfit:
        spec["fixed_set"] = _overfit_set()
    print(f"[common] training {name} ...", flush=True)
    m = S.train_model(**spec)
    torch.save(m.state_dict(), path)
    return m


def rankdata(a):
    """Average ranks (1-based), handling ties -- scipy-free."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    # boundaries of tie groups
    boundaries = np.flatnonzero(np.diff(sorted_a)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(a)]])
    for s_, e_ in zip(starts, ends):
        ranks[order[s_:e_]] = (s_ + 1 + e_) / 2.0
    return ranks


def auroc(positive_scores, negative_scores):
    """P(random positive scores higher than random negative); ties count 1/2."""
    pos, neg = np.asarray(positive_scores, float), np.asarray(negative_scores, float)
    ranks = rankdata(np.concatenate([pos, neg]))
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_json(name, obj):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    return plt


def savefig(fig, name):
    path = os.path.join(IMAGES, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {os.path.relpath(path, REPO)}", flush=True)
    return path
