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


def _ring_mask(yy, xx, cy, cx, r):
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 <= r ** 2) & (d2 >= (0.55 * r) ** 2)


def _cross_mask(yy, xx, cy, cx, r):
    t = max(1, r // 3)
    return ((np.abs(yy - cy) <= t) & (np.abs(xx - cx) <= r)) | (
        (np.abs(xx - cx) <= t) & (np.abs(yy - cy) <= r))


_NOVEL_FNS = [_ring_mask, _cross_mask]
_NOVEL_COLOR = np.array([0.9, 0.85, 0.1])  # yellow: a color/shape never seen in training


def make_batch(
    batch_size: int,
    size: int = SIZE,
    rng: np.random.Generator = None,
    n_shapes_range=(2, 3),
    noise_std: float = 0.06,
    corrupt: str = None,
    blur_k: int = 3,
    color_alpha: float = 0.0,
    noise_mult: float = 1.0,
    label_noise_p: float = 0.0,
    return_ambiguity: bool = False,
    ambiguity_kind: str = "all",
):
    """
    Generate a batch of (image, label_mask) pairs (optionally + an
    "irreducibly ambiguous pixel" mask).

    Distribution-shift knobs (all default to "in-distribution"):
      noise_mult   : multiply pixel-noise std (1 = in-distribution).
      blur_k       : box-blur kernel size (3 = in-distribution; larger = blurrier).
      color_alpha  : 0..1 interpolation from true class colors to inverted ones.
      corrupt      : shortcuts -- "heavy_noise" (noise_mult=6), "color_shift"
                     (color_alpha=1), "pure_noise" (no structure at all),
                     "novel_shape" (rings/crosses in a never-seen color).
    label_noise_p  : fraction of pixels whose label is replaced by a random
                     class (simulated annotation noise).
    return_ambiguity : also return a (B, H, W) bool mask of pixels whose label
                     is irreducibly random.
    ambiguity_kind : "all" (shape overlaps + label noise) or "overlap" (only the
                     visible, predictable shape-overlap regions).
    """
    rng = rng or np.random.default_rng()
    imgs = np.zeros((batch_size, 3, size, size), dtype=np.float32)
    masks = np.zeros((batch_size, size, size), dtype=np.int64)
    amb = np.zeros((batch_size, size, size), dtype=bool)
    ovl = np.zeros((batch_size, size, size), dtype=bool)

    if corrupt == "heavy_noise" and noise_mult == 1.0:
        noise_mult = 6.0
    if corrupt == "color_shift":
        color_alpha = 1.0

    if corrupt == "pure_noise":
        imgs = rng.uniform(0.0, 1.0, size=(batch_size, 3, size, size)).astype(np.float32)
        out = (torch.from_numpy(imgs), torch.from_numpy(masks))
        return out + (torch.from_numpy(amb),) if return_ambiguity else out

    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    for b in range(batch_size):
        bg = rng.uniform(0.05, 0.25, size=3).astype(np.float32)
        img = np.broadcast_to(bg[:, None, None], (3, size, size)).copy()
        mask = np.zeros((size, size), dtype=np.int64)

        n_shapes = rng.integers(n_shapes_range[0], n_shapes_range[1] + 1)
        for _ in range(n_shapes):
            r = int(rng.integers(size // 6, size // 3))
            cy = int(rng.integers(r + 1, size - r - 1))
            cx = int(rng.integers(r + 1, size - r - 1))

            if corrupt == "novel_shape":
                region = _NOVEL_FNS[int(rng.integers(0, 2))](yy, xx, cy, cx, r)
                for c in range(3):
                    img[c][region] = _NOVEL_COLOR[c]
                continue

            cls = int(rng.integers(1, 4))
            region = _SHAPE_FNS[cls](yy, xx, cy, cx, r)
            color = (1.0 - color_alpha) * _CLASS_COLORS[cls] + color_alpha * (1.0 - _CLASS_COLORS[cls])
            for c in range(3):
                img[c][region] = color[c]  # new shape visually drawn on top

            # Label assignment: on NEW territory the new shape's class wins.
            # Where it overlaps an already-labeled shape, flip a coin per
            # pixel: genuine, irreducible label ambiguity (like inter-annotator
            # disagreement at occlusions) that no model can resolve.
            overlap = region & (mask != 0)
            new_territory = region & ~overlap
            mask[new_territory] = cls
            if overlap.any():
                coin = rng.random(size=mask.shape) < 0.5
                mask[overlap & coin] = cls
                amb[b] |= overlap
                ovl[b] |= overlap

        if label_noise_p > 0:
            flip = rng.random(size=mask.shape) < label_noise_p
            mask[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))
            amb[b] |= flip

        img = _box_blur(img, k=blur_k)
        img = img + rng.normal(0, noise_std * noise_mult, size=img.shape).astype(np.float32)
        imgs[b] = np.clip(img, 0.0, 1.0)
        masks[b] = mask

    out = (torch.from_numpy(imgs), torch.from_numpy(masks))
    chosen = ovl if ambiguity_kind == "overlap" else amb
    return out + (torch.from_numpy(chosen),) if return_ambiguity else out


class SynthSegNet(torch.nn.Module):
    """Small enough to train in seconds on CPU, with Dropout2d for MC Dropout."""

    def __init__(self, n_classes: int = N_CLASSES, dropout_p: float = 0.25, width: int = 24):
        super().__init__()
        import torch.nn as nn
        w2 = int(width * 4 / 3)
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, w2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(w2)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv3 = nn.Conv2d(w2, width, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width)
        self.conv4 = nn.Conv2d(width, n_classes, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)


def train_model(seed: int, n_steps: int = 500, batch_size: int = 16, lr: float = 2e-3,
                width: int = 24, dropout_p: float = 0.25, fixed_set=None):
    """Train on a stream of fresh synthetic images (default), or -- if
    `fixed_set=(imgs, masks)` is given -- on that finite set only, which lets a
    big model memorize it (and become overconfident on new data)."""
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SynthSegNet(width=width, dropout_p=dropout_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_steps):
        if fixed_set is None:
            imgs, masks = make_batch(batch_size, rng=rng)
        else:
            idx = torch.from_numpy(rng.choice(len(fixed_set[0]), size=min(batch_size, len(fixed_set[0])), replace=False))
            imgs, masks = fixed_set[0][idx], fixed_set[1][idx]
        loss = loss_fn(model(imgs), masks)
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
