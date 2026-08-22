import torch
import torch.nn as nn
import pytest


class TinySegModel(nn.Module):
    """A minimal conv-net segmentation model with dropout, just big enough
    to exercise seggnosis's wrappers in tests without needing real data."""

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


@pytest.fixture
def tiny_model():
    model = TinySegModel()
    model.eval()
    return model


@pytest.fixture
def sample_image():
    torch.manual_seed(0)
    return torch.rand(3, 32, 32)
