"""
Quickstart for volumetric data (CT/MRI-style), using spatial_dims=3.

Run with:  python examples/quickstart_3d.py
"""

import torch
import torch.nn as nn

import seggnosis


class Tiny3DUNetLike(nn.Module):
    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.dropout = nn.Dropout3d(p=0.3)
        self.conv2 = nn.Conv3d(8, n_classes, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        return self.conv2(x)


def main():
    torch.manual_seed(0)
    model = Tiny3DUNetLike().eval()

    # A stand-in for one CT/MRI volume: (C, D, H, W)
    volume = torch.rand(1, 16, 64, 64)

    trusted = seggnosis.wrap(
        model, method="mc_dropout", n_samples=20, spatial_dims=3
    )
    result = trusted.predict(volume)

    print("3D volume prediction:", result.summary())
    print("  mask shape (D, H, W):        ", result.mask.shape)
    print("  probs shape (C, D, H, W):    ", result.probs.shape)
    print("  uncertainty map shape:       ", result.uncertainty_map.shape)

    # Per-slice mean uncertainty -- useful for flagging which slices of a
    # scan need a radiologist's closer look.
    per_slice_uncertainty = result.uncertainty_map.mean(axis=(1, 2))
    worst_slice = int(per_slice_uncertainty.argmax())
    print(f"\n  Most uncertain slice: {worst_slice} "
          f"(mean uncertainty {per_slice_uncertainty[worst_slice]:.3f})")


if __name__ == "__main__":
    main()
