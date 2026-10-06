"""
A small synthetic segmentation task with genuine learnable structure, used
to validate seggnosis against ground truth rather than just checking API
shapes. Classes: 0=background, 1=circle, 2=square, 3=triangle.
"""

import numpy as np
import torch


N_CLASSES = 4
SIZE = 32

_CLASS_COLORS = {
    1: np.array([0.9, 0.2, 0.2]),   # circle: red
    2: np.array([0.2, 0.8, 0.2]),   # square: green
    3: np.array([0.2, 0.3, 0.9]),   # triangle: blue
}


def _circle_mask(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _square_mask(yy, xx, cy, cx, r):
    return (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)


def _triangle_mask(yy, xx, cy, cx, r):
    # Apex at (cy - r, cx), base corners at (cy + r, cx - r) and (cy + r, cx + r).
    y0, x0 = cy - r, cx
    y1, x1 = cy + r, cx - r
    y2, x2 = cy + r, cx + r

    def sign(py, px, ay, ax, by, bx):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    d1 = sign(yy, xx, y0, x0, y1, x1)
    d2 = sign(yy, xx, y1, x1, y2, x2)
    d3 = sign(yy, xx, y2, x2, y0, x0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


_SHAPE_FNS = {1: _circle_mask, 2: _square_mask, 3: _triangle_mask}


def _box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Vectorized box blur via an integral image, no scipy dependency.
    img: (C, H, W). Softens shape edges so boundary pixels are genuinely
    ambiguous -- real segmentation models never see perfectly crisp edges
    either (sensor resolution, anti-aliasing, partial-volume effects)."""
    pad = k // 2
    padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    ii = np.cumsum(np.cumsum(padded, axis=1), axis=2)
    ii = np.pad(ii, ((0, 0), (1, 0), (1, 0)), mode="constant")
    H, W = img.shape[1], img.shape[2]
    y0, y1 = np.arange(H)[:, None], np.arange(H)[:, None] + k
    x0, x1 = np.arange(W)[None, :], np.arange(W)[None, :] + k
    total = ii[:, y1, x1] - ii[:, y0, x1] - ii[:, y1, x0] + ii[:, y0, x0]
    return total / (k * k)


def make_batch(
    batch_size: int,
    size: int = SIZE,
    rng: np.random.Generator = None,
    n_shapes_range=(2, 3),
    noise_std: float = 0.06,
    corrupt: str = None,
):
    """
    Generate a batch of (image, label_mask) pairs.

    corrupt : None, "heavy_noise", "color_shift", or "pure_noise"
        None: normal in-distribution data.
        "heavy_noise": same shapes/labels, much higher pixel noise -- a
            "near OOD" case (same task, degraded sensor/domain shift).
        "color_shift": same shapes/labels, class colors permuted/inverted --
            another "near OOD" case (same structure, wrong color statistics).
        "pure_noise": no shapes at all, just random noise images -- a "far
            OOD" case (completely different distribution, unrelated task).
    """
    rng = rng or np.random.default_rng()
    imgs = np.zeros((batch_size, 3, size, size), dtype=np.float32)
    masks = np.zeros((batch_size, size, size), dtype=np.int64)

    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    if corrupt == "pure_noise":
        imgs = rng.uniform(0.0, 1.0, size=(batch_size, 3, size, size)).astype(np.float32)
        return torch.from_numpy(imgs), torch.from_numpy(masks)  # masks unused for far-OOD

    for b in range(batch_size):
        bg = rng.uniform(0.05, 0.25, size=3).astype(np.float32)
        img = np.broadcast_to(bg[:, None, None], (3, size, size)).copy()
        mask = np.zeros((size, size), dtype=np.int64)

        # Larger shapes + overlap allowed (centers not kept apart) -> a
        # genuine population of hard boundary/overlap pixels, not just
        # interior pixels every method would get right.
        n_shapes = rng.integers(n_shapes_range[0], n_shapes_range[1] + 1)
        for _ in range(n_shapes):
            cls = int(rng.integers(1, 4))
            r = int(rng.integers(size // 6, size // 3))
            cy = int(rng.integers(r + 1, size - r - 1))
            cx = int(rng.integers(r + 1, size - r - 1))
            region = _SHAPE_FNS[cls](yy, xx, cy, cx, r)

            color = _CLASS_COLORS[cls].copy()
            if corrupt == "color_shift":
                color = 1.0 - color  # invert the class/color association

            for c in range(3):
                img[c][region] = color[c]  # new shape visually drawn on top, as normal

            # Label assignment: on NEW territory the new shape's class wins
            # outright, same as before. On territory that overlaps an
            # already-labeled shape, flip a coin per pixel instead of
            # deterministically overwriting. This injects genuine,
            # irreducible label ambiguity -- like real inter-annotator
            # disagreement at occluded/overlapping boundaries -- that no
            # amount of model capacity or training can resolve, which is
            # exactly the case uncertainty estimation needs to be tested
            # against (as opposed to error that's merely "hard but
            # learnable", which more training would eventually fix).
            overlap = region & (mask != 0)
            new_territory = region & ~overlap
            mask[new_territory] = cls
            if overlap.any():
                coin = rng.random(size=mask.shape) < 0.5
                mask[overlap & coin] = cls
                # else: pixel keeps its prior label, by design (image still
                # shows the new shape's color there -- the ambiguity is in
                # the label, not the pixel, same as real annotation noise)

        img = _box_blur(img, k=3)  # soften edges: real boundary ambiguity
        std = noise_std if corrupt != "heavy_noise" else noise_std * 6
        img = img + rng.normal(0, std, size=img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        imgs[b] = img
        masks[b] = mask

    return torch.from_numpy(imgs), torch.from_numpy(masks)


class SynthSegNet(torch.nn.Module):
    """Small enough to train in seconds on CPU, with Dropout2d for MC Dropout."""

    def __init__(self, n_classes: int = N_CLASSES, dropout_p: float = 0.25):
        super().__init__()
        import torch.nn as nn
        self.conv1 = nn.Conv2d(3, 24, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv3 = nn.Conv2d(32, 24, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, n_classes, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)


def train_model(seed: int, n_steps: int = 500, batch_size: int = 16, lr: float = 2e-3):
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SynthSegNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(n_steps):
        imgs, masks = make_batch(batch_size, rng=rng)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def mean_iou(pred_mask: np.ndarray, true_mask: np.ndarray, n_classes: int = N_CLASSES) -> float:
    ious = []
    for c in range(n_classes):
        pred_c = pred_mask == c
        true_c = true_mask == c
        union = (pred_c | true_c).sum()
        if union == 0:
            continue
        inter = (pred_c & true_c).sum()
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0
