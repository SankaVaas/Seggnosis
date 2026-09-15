"""
Conformal prediction: pixel-wise class sets with a coverage guarantee,
calibrated on held-out labeled data.

Run with:  python examples/conformal_prediction.py
"""

import numpy as np
import torch
import torch.nn as nn

import seggnosis


class TinyUNetLike(nn.Module):
    def __init__(self, in_channels=3, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(16, n_classes, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        return self.conv2(x)


def fake_labeled_batches(n_batches, batch_size, size, n_classes, seed):
    """Stand-in for a real held-out labeled DataLoader."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = torch.rand(batch_size, 3, size, size, generator=g)
        y = torch.randint(0, n_classes, (batch_size, size, size), generator=g)
        yield x, y


def main():
    torch.manual_seed(0)
    model = TinyUNetLike().eval()

    # 1. Wrap with method="conformal" and pick a target miscoverage rate.
    #    alpha=0.1 targets ~90% of pixels having their true class inside
    #    the predicted set.
    trusted = seggnosis.wrap(model, method="conformal", alpha=0.1)

    # 2. Calibrate on held-out LABELED data (distinct from training data).
    trusted.calibrate(fake_labeled_batches(60, 8, 32, n_classes=4, seed=1))

    # 3. Predict: every pixel gets a *set* of plausible classes, not just
    #    a single top prediction.
    image = torch.rand(3, 32, 32)
    result = trusted.predict(image)
    prediction_set = result.raw["prediction_set"][0]  # (B=1, C, H, W) -> (C, H, W)

    avg_set_size = prediction_set.sum(axis=0).mean()
    print(f"Target coverage: {result.raw['target_coverage']:.0%}")
    print(f"Average prediction-set size per pixel: {avg_set_size:.2f} "
          f"(out of {prediction_set.shape[0]} classes)")
    print(result.summary())

    # 4. Sanity-check the coverage guarantee on a fresh held-out batch.
    covered, total = 0, 0
    for x, y in fake_labeled_batches(20, 8, 32, n_classes=4, seed=2):
        r = trusted.predict(x)
        pred_set = r.raw["prediction_set"]  # (B, C, H, W)
        y_np = y.numpy()
        in_set = np.take_along_axis(pred_set, y_np[:, None, ...], axis=1)
        covered += int(in_set.sum())
        total += in_set.size
    print(f"\nEmpirical coverage on held-out data: {covered / total:.1%} "
          f"(target: {1 - trusted.alpha:.0%})")


if __name__ == "__main__":
    main()
