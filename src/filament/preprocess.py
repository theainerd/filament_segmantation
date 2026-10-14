"""Solar-disk fitting and limb-darkening flat-field for GONG H-alpha frames.

Outputs three float32 planes used as model input:
  0. contrast   = I / I_limb(r) - 1   (radial-profile flat field), clipped
  1. intensity  = I / disk median     (raw, normalised)
  2. r/R        = normalised radius (0 at centre, 1 at limb, >1 off-disk)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


def fit_disk(img: np.ndarray) -> Disk:
    """Otsu threshold + largest component + least-squares circle on its boundary."""
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    s = img.shape[0] / 512.0
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        h, w = img.shape
        return Disk(w / 2, h / 2, 0.45 * w)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    # algebraic circle fit (Kasa)
    x, y = c[:, 0], c[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(d + cx ** 2 + cy ** 2)
    return Disk(cx * s + (s - 1) / 2, cy * s + (s - 1) / 2, r * s)


def radius_map(shape, disk: Disk) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.hypot(xx - disk.cx, yy - disk.cy) / disk.r


def radial_profile(img: np.ndarray, rr: np.ndarray, nbins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Robust (median) intensity vs r/R inside the disk."""
    inside = rr < 1.0
    idx = np.clip((rr[inside] * nbins).astype(int), 0, nbins - 1)
    vals = img[inside].astype(np.float32)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    edges = np.searchsorted(idx, np.arange(nbins + 1))
    prof = np.full(nbins, np.nan, np.float32)
    for i in range(nbins):
        a, b = edges[i], edges[i + 1]
        if b - a > 20:
            prof[i] = np.median(vals[a:b])
    good = ~np.isnan(prof)
    centres = (np.arange(nbins) + 0.5) / nbins
    prof = np.interp(centres, centres[good], prof[good]).astype(np.float32)
    # light smoothing
    k = np.ones(7) / 7
    prof = np.convolve(np.pad(prof, 3, mode="edge"), k, mode="valid").astype(np.float32)
    return centres, prof


def flat_field(img: np.ndarray, disk: Disk | None = None, clip: float = 1.0):
    img = img.astype(np.float32)
    disk = disk or fit_disk(img.astype(np.uint8))
    rr = radius_map(img.shape, disk)
    centres, prof = radial_profile(img, rr)
    limb = np.interp(np.minimum(rr, 0.999), centres, prof)
    limb = np.maximum(limb, 1.0)
    contrast = np.clip(img / limb - 1.0, -clip, clip)
    contrast[rr >= 1.0] = 0.0
    med = float(np.median(img[rr < 0.9])) or 1.0
    inten = np.clip(img / med, 0, 2.0) - 1.0
    planes = np.stack([contrast, inten, np.minimum(rr, 1.5)], 0).astype(np.float32)
    return planes, disk


CLIP = 0.5


def compact(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, Disk]:
    """uint8 cache: (contrast_u8, intensity_u8, disk). Memory: 2 x 4 MB per frame."""
    planes, disk = flat_field(img, clip=CLIP)
    c8 = np.clip((planes[0] / CLIP + 1.0) * 127.5, 0, 255).astype(np.uint8)
    i8 = np.clip((planes[1] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return c8, i8, disk


def planes_from_compact(c8: np.ndarray, i8: np.ndarray, disk: Disk,
                        y0: int = 0, x0: int = 0, h: int | None = None, w: int | None = None) -> np.ndarray:
    """Model input (3,h,w) float32 for a crop of the cached frame."""
    h = h or c8.shape[0]; w = w or c8.shape[1]
    c = c8[y0:y0 + h, x0:x0 + w].astype(np.float32) / 127.5 - 1.0     # in [-1,1] = contrast/CLIP
    i = i8[y0:y0 + h, x0:x0 + w].astype(np.float32) / 127.5 - 1.0
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    rr = np.hypot(xx - disk.cx, yy - disk.cy) / disk.r
    return np.stack([c * 2.0, i, np.minimum(rr, 1.5) - 0.5], 0).astype(np.float32)
