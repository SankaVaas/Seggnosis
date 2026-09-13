import numpy as np
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


def test_mc_dropout_classification():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    image = torch.rand(3, 32, 32)

    trusted = seggnosis.classification(model, method="mc_dropout", n_samples=10)
    result = trusted.predict(image)

    assert isinstance(result.predicted_class, int)
    assert result.probs.shape == (5,)
    assert 0.0 <= result.confidence <= 1.0
    assert result.uncertainty >= 0.0


def test_tta_classification():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    image = torch.rand(3, 32, 32)

    trusted = seggnosis.classification(model, method="tta")
    result = trusted.predict(image)
    assert result.probs.shape == (5,)


def test_ensemble_classification():
    torch.manual_seed(0)
    m1, m2 = TinyClassifier(), TinyClassifier()
    image = torch.rand(3, 32, 32)

    trusted = seggnosis.classification(m1, method="ensemble", models=[m2])
    result = trusted.predict(image)
    assert result.probs.shape == (5,)
    assert result.raw["samples"].shape[0] == 2


def test_ood_detector_on_classification():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    image = torch.rand(3, 32, 32)

    in_dist_loader = [torch.rand(2, 3, 32, 32) for _ in range(6)]
    detector = seggnosis.MahalanobisOOD(layer_name="conv").fit(model, in_dist_loader, max_batches=6)

    trusted = seggnosis.classification(model, method="mc_dropout", n_samples=5)
    trusted.attach_ood_detector(detector)
    result = trusted.predict(image)

    assert result.is_ood in (True, False)
    assert result.ood_score is not None


def test_segmentation_alias_unchanged():
    assert seggnosis.segmentation is seggnosis.wrap


def test_invalid_classification_method_raises():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    try:
        seggnosis.classification(model, method="not_real")
        assert False, "expected ValueError"
    except ValueError:
        pass