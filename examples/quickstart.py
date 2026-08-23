"""
Quickstart: wrap a segmentation model, get an uncertainty-aware prediction,
attach an OOD detector, and check calibration.

Run with:  python examples/quickstart.py
"""

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


def main():
    torch.manual_seed(0)
    model = TinyUNetLike().eval()
    image = torch.rand(3, 64, 64)

    # 1. Wrap the model -- no retraining needed.
    trusted = seggnosis.wrap(model, method="mc_dropout", n_samples=20, uncertainty="entropy")

    # 2. Predict with a trust signal attached.
    result = trusted.predict(image)
    print("Without OOD detector:", result.summary())
    print("  mask shape:", result.mask.shape)
    print("  uncertainty map shape:", result.uncertainty_map.shape)

    # 3. Fit an OOD detector on "in-distribution" data and attach it.
    in_dist_loader = [torch.rand(4, 3, 64, 64) for _ in range(10)]
    detector = seggnosis.MahalanobisOOD(layer_name="conv1").fit(model, in_dist_loader)
    trusted.attach_ood_detector(detector)

    result = trusted.predict(image)
    print("\nWith OOD detector:", result.summary())

    shifted_image = torch.rand(3, 64, 64) * 5.0 + 3.0  # simulate a badly out-of-domain input
    ood_result = trusted.predict(shifted_image)
    print("Shifted/OOD input:  ", ood_result.summary())

    # 4. Calibrate confidence with temperature scaling on held-out labeled data.
    calib_data = [
        (torch.rand(2, 3, 64, 64), torch.randint(0, 4, (2, 64, 64))) for _ in range(10)
    ]
    scaler = seggnosis.TemperatureScaler().fit(model, calib_data)
    print(f"\nLearned temperature: {scaler.temperature:.3f}")


if __name__ == "__main__":
    main()
