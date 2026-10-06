"""
Empirical validation of seggnosis against ground truth -- not just API
contract tests. Trains a small real model on a synthetic-but-structured
segmentation task (so there's genuine signal to detect) and checks:

1. Uncertainty quality: does higher uncertainty actually predict higher
   pixel-wise error, for mc_dropout / tta / ensemble?
2. OOD detection: does MahalanobisOOD actually separate in-distribution
   data from out-of-distribution data (both a "far" and a "near" OOD case)?
3. Calibration: does temperature scaling actually reduce Expected
   Calibration Error?
4. Conformal prediction: does empirical coverage on a REAL trained model's
   predictions hold at the target rate (the unit tests only checked this
   with random, unlearnable labels)?

Run with:  python benchmarks/validate_library.py
"""

import numpy as np
import torch

import seggnosis
from synthetic_shapes import N_CLASSES, SynthSegNet, make_batch, mean_iou, train_model


def auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (no sklearn dependency).
    'positive' = the class we want HIGH scores for (here: OOD)."""
    all_scores = np.concatenate([positive_scores, negative_scores])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    # ranks is aligned with the concatenation order: positives first, then negatives.
    sum_ranks_pos = ranks[:n_pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    section("Training a real model on the synthetic shapes task")
    model = train_model(seed=0, n_steps=500)
    model2 = train_model(seed=1, n_steps=400)  # for the ensemble check

    rng = np.random.default_rng(123)
    test_imgs, test_masks = make_batch(300, rng=rng)
    with torch.no_grad():
        preds = model(test_imgs).argmax(dim=1).numpy()
    iou = mean_iou(preds, test_masks.numpy())
    random_baseline = 1.0 / N_CLASSES
    print(f"Held-out mean IoU: {iou:.3f}  (random baseline ~ {random_baseline:.3f})")
    if iou < 0.4:
        print("WARNING: model barely learned the task -- downstream checks "
              "below would not be meaningful. Increase n_steps.")

    # ------------------------------------------------------------------
    section("1. Uncertainty quality: does it track real pixel-wise error?")
    # ------------------------------------------------------------------
    rng = np.random.default_rng(7)
    eval_imgs, eval_masks = make_batch(40, rng=rng)

    for method, kwargs in [
        ("mc_dropout", {"n_samples": 15, "uncertainty": "entropy"}),
        ("tta", {"uncertainty": "variance"}),
        ("ensemble", {"models": [model2], "uncertainty": "mutual_information"}),
    ]:
        trusted = seggnosis.wrap(model, method=method, **kwargs)
        all_uncertainty, all_correct = [], []
        for i in range(eval_imgs.shape[0]):
            result = trusted.predict(eval_imgs[i])
            correct = (result.mask == eval_masks[i].numpy()).astype(np.float32)
            all_uncertainty.append(result.uncertainty_map.ravel())
            all_correct.append(correct.ravel())
        u = np.concatenate(all_uncertainty)
        c = np.concatenate(all_correct)

        overall_acc = c.mean()
        # bin pixels into uncertainty quartiles, report accuracy per bin
        order = np.argsort(u)
        c_sorted = c[order]
        quartiles = np.array_split(c_sorted, 4)
        bin_accs = [float(q.mean()) for q in quartiles]
        monotonic = all(bin_accs[i] >= bin_accs[i + 1] - 0.02 for i in range(3))
        if u.std() == 0 or c.std() == 0:
            corr = float("nan")  # degenerate: no variance to correlate (e.g. 100% accuracy)
        else:
            corr = float(np.corrcoef(u, c)[0, 1])  # expect negative

        print(f"\n[{method}] overall accuracy: {overall_acc:.3f}")
        print(f"  accuracy by uncertainty quartile (lowest->highest uncertainty): "
              + ", ".join(f"{a:.3f}" for a in bin_accs))
        print(f"  monotonically decreasing (allowing 2% slack): {monotonic}")
        print(f"  correlation(uncertainty, correctness): {corr:+.3f}  (expect negative)")

    # ------------------------------------------------------------------
    section("1b. Entropy vs mutual information: aleatoric vs epistemic")
    # ------------------------------------------------------------------
    # Shape-overlap pixels have randomized, irreducible label noise by
    # construction (see synthetic_shapes.py) -- no amount of model skill
    # can predict them above chance. "entropy" (total uncertainty) should
    # still fire there, since the model genuinely can't be confident.
    # "mutual_information" (epistemic only, via an ensemble) is supposed to
    # isolate disagreement *caused by the models not knowing the answer*
    # from disagreement caused by the label itself being unresolvable --
    # so it should fire noticeably LESS on these specific pixels than
    # entropy does, relative to each measure's own typical pixel.
    trusted_ent = seggnosis.wrap(model, method="ensemble", models=[model2], uncertainty="entropy")
    trusted_mi = seggnosis.wrap(model, method="ensemble", models=[model2], uncertainty="mutual_information")
    rng = np.random.default_rng(321)
    noisy_imgs, _ = make_batch(30, rng=rng, n_shapes_range=(2, 3))

    ent_ratio, mi_ratio = [], []
    for i in range(noisy_imgs.shape[0]):
        r_ent = trusted_ent.predict(noisy_imgs[i])
        r_mi = trusted_mi.predict(noisy_imgs[i])
        # crude proxy for "ambiguous-by-construction" pixels: where the two
        # ensemble members' top-1 predictions disagree
        probs = r_ent.probs  # mean probs, same prediction either way
        pred = probs.argmax(axis=0)
        # use high-raw-entropy pixels as the ambiguous-region proxy
        thresh = np.percentile(r_ent.uncertainty_map, 90)
        ambiguous = r_ent.uncertainty_map >= thresh
        if ambiguous.sum() == 0:
            continue
        ent_ratio.append(r_ent.uncertainty_map[ambiguous].mean() / (r_ent.uncertainty_map.mean() + 1e-8))
        mi_ratio.append(r_mi.uncertainty_map[ambiguous].mean() / (r_mi.uncertainty_map.mean() + 1e-8))

    print(f"On the top-10%-most-ambiguous pixels (by entropy), relative to each "
          f"measure's own image-wide mean:")
    print(f"  entropy ratio:            {np.mean(ent_ratio):.2f}x")
    print(f"  mutual_information ratio: {np.mean(mi_ratio):.2f}x")
    print("  (entropy picking up relatively more there than MI does would support "
          "entropy ~ total uncertainty and MI ~ epistemic-only, as designed)")

    # ------------------------------------------------------------------
    section("2. OOD detection: does it actually separate distributions?")
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    fit_imgs, _ = make_batch(200, rng=rng)
    in_dist_loader = [fit_imgs[i:i + 20] for i in range(0, 200, 20)]
    detector = seggnosis.MahalanobisOOD(layer_name="conv2").fit(model, in_dist_loader)

    rng = np.random.default_rng(99)
    id_test, _ = make_batch(100, rng=rng)
    far_ood, _ = make_batch(100, rng=rng, corrupt="pure_noise")
    near_ood, _ = make_batch(100, rng=rng, corrupt="heavy_noise")
    color_ood, _ = make_batch(100, rng=rng, corrupt="color_shift")

    id_scores = detector.score(model, id_test)
    far_scores = detector.score(model, far_ood)
    near_scores = detector.score(model, near_ood)
    color_scores = detector.score(model, color_ood)

    print(f"Mean OOD score -- in-distribution: {id_scores.mean():.2f}, "
          f"far-OOD (pure noise): {far_scores.mean():.2f}, "
          f"near-OOD (heavy noise): {near_scores.mean():.2f}, "
          f"near-OOD (color shift): {color_scores.mean():.2f}")
    print(f"AUROC in-dist vs far-OOD (pure noise):   {auroc(far_scores, id_scores):.3f}  (expect close to 1.0)")
    print(f"AUROC in-dist vs near-OOD (heavy noise): {auroc(near_scores, id_scores):.3f}  (expect > 0.5, likely < far-OOD case)")
    print(f"AUROC in-dist vs near-OOD (color shift): {auroc(color_scores, id_scores):.3f}  (expect > 0.5)")

    # ------------------------------------------------------------------
    section("3. Calibration: does temperature scaling reduce ECE?")
    # ------------------------------------------------------------------
    rng = np.random.default_rng(55)
    calib_data = [make_batch(16, rng=rng) for _ in range(10)]
    scaler = seggnosis.TemperatureScaler().fit(model, calib_data)

    rng = np.random.default_rng(56)
    ece_imgs, ece_masks = make_batch(100, rng=rng)
    with torch.no_grad():
        logits = model(ece_imgs)
        raw_probs = torch.softmax(logits, dim=1)
        calibrated_probs = scaler.apply(logits)

    ece_before = seggnosis.expected_calibration_error(raw_probs, ece_masks)
    ece_after = seggnosis.expected_calibration_error(calibrated_probs, ece_masks)
    print(f"Learned temperature: {scaler.temperature:.3f}")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}  "
          f"({'improved' if ece_after <= ece_before else 'did not improve'})")

    # ------------------------------------------------------------------
    section("4. Conformal prediction: coverage on a REAL trained model")
    # ------------------------------------------------------------------
    for alpha in (0.1, 0.2):
        trusted = seggnosis.wrap(model, method="conformal", alpha=alpha)
        rng = np.random.default_rng(777)
        calib_batches = [make_batch(16, rng=rng) for _ in range(15)]
        trusted.calibrate(calib_batches)

        rng = np.random.default_rng(778)
        test_imgs2, test_masks2 = make_batch(15, rng=rng)
        covered, total, set_sizes = 0, 0, []
        for i in range(test_imgs2.shape[0]):
            result = trusted.predict(test_imgs2[i])
            pred_set = result.raw["prediction_set"][0]  # (C, H, W)
            y = test_masks2[i].numpy()
            in_set = np.take_along_axis(pred_set, y[None, ...], axis=0)[0]
            covered += int(in_set.sum())
            total += in_set.size
            set_sizes.append(pred_set.sum(axis=0).mean())

        empirical_coverage = covered / total
        target = 1 - alpha
        print(f"\nalpha={alpha} (target coverage {target:.0%}): "
              f"empirical coverage = {empirical_coverage:.1%}, "
              f"avg prediction-set size = {np.mean(set_sizes):.2f} / {N_CLASSES} classes")


if __name__ == "__main__":
    main()
