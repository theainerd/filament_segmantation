"""Submission RLE helpers.

Contract (to be re-verified against the hosts' self-evaluation notebook):
  * CSV columns ``filament_id,segmentation_rle``; one row per predicted instance.
  * ``filament_id = <image_stem>_<n>``.
  * ``segmentation_rle`` = pycocotools *compressed* RLE ``counts`` string of a
    2048x2048 Fortran-order uint8 mask (no size field).
  * No overlapping pixels between instances of the same image; no empty masks.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

H = W = 2048


def encode(mask: np.ndarray) -> str:
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    return rle["counts"].decode("ascii")


def decode(counts: str, h: int = H, w: int = W) -> np.ndarray:
    return mask_utils.decode({"size": [h, w], "counts": counts.encode("ascii")}).astype(bool)


def to_rle(counts: str, h: int = H, w: int = W) -> dict:
    return {"size": [h, w], "counts": counts.encode("ascii")}


def labels_to_rows(stem: str, labels: np.ndarray) -> list[tuple[str, str]]:
    """Label image (0=bg, 1..K instances) -> submission rows."""
    rows = []
    ids = [i for i in np.unique(labels) if i != 0]
    for n, i in enumerate(ids):
        m = labels == i
        if m.any():
            rows.append((f"{stem}_{n}", encode(m)))
    return rows


def write_csv(rows, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)


def read_csv(path: str | Path) -> dict[str, list[str]]:
    """-> {stem: [counts, ...]}"""
    out: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            stem = row["filament_id"].rsplit("_", 1)[0]
            out[stem].append(row["segmentation_rle"])
    return dict(out)


def validate_csv(path: str | Path, known_stems: set[str] | None = None) -> dict:
    """Decode every mask, check emptiness/overlap/ids/round-trip. Returns a report dict."""
    errors = []
    n_rows = 0
    seen_ids = set()
    with open(path) as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["filament_id", "segmentation_rle"]:
            errors.append(f"bad header {r.fieldnames}")
        per_image: dict[str, list[str]] = defaultdict(list)
        for row in r:
            n_rows += 1
            fid = row["filament_id"]
            if fid in seen_ids:
                errors.append(f"duplicate id {fid}")
            seen_ids.add(fid)
            per_image[fid.rsplit("_", 1)[0]].append(row["segmentation_rle"])
    for stem, cl in per_image.items():
        if known_stems is not None and stem not in known_stems:
            errors.append(f"unknown stem {stem}")
        acc = np.zeros((H, W), np.uint8)
        for c in cl:
            m = decode(c)
            if not m.any():
                errors.append(f"empty mask in {stem}")
            if encode(m) != c:
                errors.append(f"non-canonical RLE in {stem}")
            acc += m
        if (acc > 1).any():
            errors.append(f"overlap in {stem}: {(acc > 1).sum()} px")
    return {"rows": n_rows, "images": len(per_image), "errors": errors[:50], "n_errors": len(errors)}
