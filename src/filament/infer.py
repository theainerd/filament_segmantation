"""Full-disk tiled inference with raised-cosine blending and dihedral TTA."""
from __future__ import annotations

import numpy as np
import torch


def _window(t: int) -> np.ndarray:
    w = np.hanning(t + 2)[1:-1]
    w = np.outer(w, w).astype(np.float32)
    return np.maximum(w, 1e-3)


def _tta(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def _untta(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    if k >= 4:
        x = x.flip(-1)
    return x


@torch.no_grad()
def predict_full(model, planes: np.ndarray, tile: int = 512, overlap: int = 128,
                 tta: int = 1, batch: int = 8, device="cuda", amp: bool = True) -> np.ndarray:
    """planes: (C,H,W) float32 -> probability map (H,W) float32.

    tta: number of dihedral transforms (1, 2, 4 or 8).
    """
    model.eval()
    _, h, w = planes.shape
    x_full = torch.from_numpy(planes)[None]
    tile = min(tile, h)
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    win = torch.from_numpy(_window(tile)).to(device)
    acc = torch.zeros((h, w), device=device)
    wacc = torch.zeros((h, w), device=device)
    coords = [(y, x) for y in ys for x in xs]
    ks = {1: [0], 2: [0, 4], 4: [0, 1, 2, 3], 8: list(range(8))}[tta]
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = torch.cat([x_full[:, :, y:y + tile, x:x + tile] for y, x in cb]).to(device)
        out = torch.zeros((len(cb), tile, tile), device=device)
        for k in ks:
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                enabled=amp and "cuda" in str(device)):
                logit = model(_tta(xb, k))
            out += _untta(torch.sigmoid(logit.float()), k)[:, 0]
        out /= len(ks)
        for j, (y, x) in enumerate(cb):
            acc[y:y + tile, x:x + tile] += out[j] * win
            wacc[y:y + tile, x:x + tile] += win
    return (acc / wacc).cpu().numpy()
