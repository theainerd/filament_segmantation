"""Ensemble inference: load trained fold models and average their tiled probability maps.

`load_runs` reads every run directory (args.json + best.pth) matching a glob pattern.
`ensemble_map` predicts one full-disk frame with all loaded models (512 px tiles, dihedral TTA)
and returns the mean probability map and the fitted solar disk.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from .infer import predict_full
from .preprocess import compact, planes_from_compact
from .train import make_model


def load_runs(pattern: str, ckpt: str, device):
    """Load the models of all run directories matching `pattern` (comma-separated globs)."""
    runs = []
    for d in sorted({p for pat in pattern.split(",") for p in glob.glob(pat, recursive=True)}):
        d = Path(d)
        if not (d / "args.json").exists() or not (d / f"{ckpt}.pth").exists():
            continue
        a = json.load(open(d / "args.json"))
        m = make_model(a["encoder"], pretrained=False)
        m.load_state_dict(torch.load(d / f"{ckpt}.pth", map_location=device))
        runs.append((d, m.to(device).eval(), a.get("tile", 512)))
        print("loaded", d, a["encoder"], flush=True)
    return runs


def ensemble_map(runs, img, device, tta):
    """Mean probability map of all models for one frame, and the fitted solar disk."""
    c8, i8, disk = compact(img)
    planes = planes_from_compact(c8, i8, disk)
    acc = np.zeros(img.shape, np.float32)
    for _, model, tile in runs:
        acc += predict_full(model, planes, tile=tile, tta=tta, device=device)
    return acc / len(runs), disk
