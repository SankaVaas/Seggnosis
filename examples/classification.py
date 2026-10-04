"""
seggnosis.classification(): the same uncertainty/OOD machinery as
seggnosis.wrap(), for whole-image classifiers instead of per-pixel
segmentation models.

Run with:  python examples/classification.py
"""

import torch
import torch.nn as nn

import seggnosis


class TinyClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main():
    torch.manual_seed(0)
    model = TinyClassifier(n_classes=5).eval()
    image = torch.rand(3, 32, 32)

    # Same three methods as seggnosis.wrap(), applied to (B, C) logits.
    trusted = seggnosis.classification(model, method="mc_dropout", n_samples=20)
    result = trusted.predict(image)
    print(f"Predicted class: {result.predicted_class}")
    print(result.summary())

    # Attach an OOD detector exactly like the segmentation API.
    in_dist_loader = [torch.rand(4, 3, 32, 32) for _ in range(10)]
    detector = seggnosis.MahalanobisOOD(layer_name="conv").fit(model, in_dist_loader)
    trusted.attach_ood_detector(detector)

    result = trusted.predict(image)
    print(f"\nWith OOD detector attached: {result.summary()}")

    shifted = torch.rand(3, 32, 32) * 5.0 + 3.0  # clearly out-of-distribution
    result = trusted.predict(shifted)
    print(f"Shifted/OOD input:          {result.summary()}")

    # seggnosis.segmentation is just seggnosis.wrap under a clearer name.
    print(f"\nseggnosis.segmentation is seggnosis.wrap: "
          f"{seggnosis.segmentation is seggnosis.wrap}")


if __name__ == "__main__":
    main()
