"""Probability map -> non-overlapping instance label image.

Steps (all tunable against OOF PQ):
  1. hysteresis: components of (p > t_low) that contain at least one p > t_high
  2. optional morphological closing / hole fill
  3. fragment rejoining: components whose gap < join_px are merged
     (implemented via dilation-based grouping, original pixels kept)
  4. min-area / min-peak filter
  5. optional dilation by `dilate` px, assigned without overlap
  6. disk mask (r/R < rmax)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class PPConfig:
    t_high: float = 0.7
    t_low: float = 0.4
    close: int = 0
    fill_holes: bool = True
    join_px: int = 0
    min_area: int = 300
    min_peak: float = 0.0
    dilate: int = 0
    rmax: float = 1.02

    def to_dict(self):
        return asdict(self)


def _disk_kernel(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def hysteresis(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    low = p > lo
    lab, n = ndi.label(low, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(low)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[p > hi])] = True
    keep[0] = False
    return keep[lab]


def instances(p: np.ndarray, cfg: PPConfig, rr: np.ndarray | None = None) -> np.ndarray:
    m = hysteresis(p, cfg.t_low, cfg.t_high)
    if rr is not None:
        m &= rr < cfg.rmax
    if cfg.close > 0:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _disk_kernel(cfg.close)).astype(bool)
    if cfg.fill_holes:
        m = ndi.binary_fill_holes(m)
    st = np.ones((3, 3))
    if cfg.join_px > 0:
        grown = cv2.dilate(m.astype(np.uint8), _disk_kernel(max(1, cfg.join_px // 2))).astype(bool)
        lab, n = ndi.label(grown, structure=st)
        lab = lab * m
    else:
        lab, n = ndi.label(m, structure=st)
    if n == 0:
        return lab.astype(np.int32)
    idx = np.arange(1, n + 1)
    area = ndi.sum_labels(np.ones_like(lab), lab, idx)
    peak = ndi.maximum(p, lab, idx) if cfg.min_peak > 0 else np.ones(n)
    good = (area >= cfg.min_area) & (peak >= cfg.min_peak)
    remap = np.zeros(n + 1, np.int32)
    remap[idx[good]] = np.arange(1, good.sum() + 1)
    lab = remap[lab]
    if cfg.dilate > 0 and lab.max() > 0:
        # grow each instance into background only, nearest-label assignment
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        grown = lab[iy, ix]
        grown[dist > cfg.dilate] = 0
        if rr is not None:
            grown[rr >= cfg.rmax] = 0
        lab = np.where(lab > 0, lab, grown)
    return lab.astype(np.int32)
