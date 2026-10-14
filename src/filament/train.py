"""Train one tiled, native-resolution U-Net (EfficientNet encoder, scSE decoder attention) for filament masks.

Loss: BCE + Tversky, optionally + BiCC component loss (`--bicc-w`). Runs on one GPU (about 1.6 h per fold on an
NVIDIA A10) or on a CPU (`--smoke` for a short sanity run).

    python -m filament.train --fold 0 --out runs/f0 --encoder efficientnet-b3 [--bicc-w 0.3]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu
from torch.utils.data import DataLoader, Dataset

from . import data as D
from . import metrics as M
from .infer import predict_full
from .losses import bicc_loss, voronoi_cells
from .postprocess import PPConfig, instances
from .preprocess import compact, planes_from_compact


# ----------------------------------------------------------------- caching
def _cache_one(args):
    stem, root = args
    c8, i8, disk = compact(D.load_image(stem, "train", Path(root)))
    return stem, c8, i8, disk


def _target_one(args):
    """Soft target: the fraction of annotator readings that mark each pixel (uint8, 0..255)."""
    stem, readings = args
    acc = np.zeros((D.H, D.W), np.float32)
    for r in readings:
        union = np.zeros((D.H, D.W), bool)
        for gr in r.rles():
            union |= mu.decode(gr).astype(bool)
        acc += union
    acc /= max(len(readings), 1)
    return stem, (acc * 255 + 0.5).astype(np.uint8)


def build_cache(stems, root, workers):
    with Pool(workers) as p:
        out = p.map(_cache_one, [(s, str(root)) for s in stems], chunksize=4)
    return {r[0]: r[1:] for r in out}                   # stem -> (contrast u8, intensity u8, disk)


# ----------------------------------------------------------------- dataset
def random_disk_center(disk):
    """Uniform random point inside the solar disk, as (y, x)."""
    angle = random.uniform(0, 2 * math.pi)
    radius = disk.r * math.sqrt(random.random())
    return disk.cy + radius * math.sin(angle), disk.cx + radius * math.cos(angle)


class TileDS(Dataset):
    """Random 512 px training tiles: 75 % centred near a filament pixel, 25 % anywhere on the disk."""

    def __init__(self, stems, cache, targets, weights, tile=512, n=4000, pos_frac=0.75, aug=True, bicc=False):
        self.stems, self.cache, self.targets = stems, cache, targets
        w = np.asarray([weights[s] for s in stems], np.float64)
        self.p = w / w.sum()
        self.tile, self.n, self.pos_frac, self.aug, self.bicc = tile, n, pos_frac, aug, bicc
        self.pos = {}
        for s in stems:
            yy, xx = np.nonzero(targets[s] > 0)
            if len(yy) > 4000:
                k = np.random.default_rng(0).choice(len(yy), 4000, replace=False)
                yy, xx = yy[k], xx[k]
            self.pos[s] = (yy, xx)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        s = self.stems[np.random.choice(len(self.stems), p=self.p)]
        c8, i8, disk = self.cache[s]
        H, W = c8.shape
        t = self.tile
        yy, xx = self.pos[s]
        if len(yy) and random.random() < self.pos_frac:
            j = random.randrange(len(yy))
            cy = yy[j] + random.randint(-t // 3, t // 3)
            cx = xx[j] + random.randint(-t // 3, t // 3)
        else:
            cy, cx = random_disk_center(disk)
        y0 = int(np.clip(cy - t // 2, 0, H - t)); x0 = int(np.clip(cx - t // 2, 0, W - t))
        x = planes_from_compact(c8, i8, disk, y0, x0, t, t)
        y = self.targets[s][y0:y0 + t, x0:x0 + t].astype(np.float32)[None] / 255.0
        if self.aug:
            k = random.randrange(8)
            if k >= 4:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            x, y = np.rot90(x, k % 4, (1, 2)), np.rot90(y, k % 4, (1, 2))
            # photometric jitter on the contrast and intensity planes
            x = x.copy()
            x[0] = x[0] * random.uniform(0.75, 1.3) + random.uniform(-0.1, 0.1)
            x[1] = x[1] * random.uniform(0.8, 1.2) + random.uniform(-0.15, 0.15)
            if random.random() < 0.3:
                sig = random.uniform(0.5, 1.5)
                x[0] = cv2.GaussianBlur(x[0], (0, 0), sig); x[1] = cv2.GaussianBlur(x[1], (0, 0), sig)
            if random.random() < 0.3:
                x[:2] += np.random.normal(0, random.uniform(0.01, 0.06), x[:2].shape).astype(np.float32)
        y = np.ascontiguousarray(y)
        m = y[0] >= 0.5
        vor = voronoi_cells(m).astype(np.float32)[None] if self.bicc else np.zeros_like(y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(vor)


# ----------------------------------------------------------------- model / loss
def make_model(encoder, pretrained=True):
    """U-Net with scSE decoder attention; 3 input planes, 1 output logit."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1, decoder_attention_type="scse")


def loss_fn(logit, y, vor, alpha=0.5, beta=0.5, bicc_w=0.0, bicc_alpha=0.5):
    """BCE + Tversky loss (+ BiCC). alpha weights false positives, beta false negatives."""
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
    loss = bce + 1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)
    if bicc_w > 0:
        loss = loss + bicc_w * bicc_loss(logit, y, vor, bicc_alpha)
    return loss


# ----------------------------------------------------------------- eval
def evaluate(model, stems, cache, gt_by_stem, device, tta=1, cfg=None, save_dir=None):
    """PQ counts of the validation images with the default post-processing rule."""
    cfg = cfg or PPConfig()
    tot = M.PQCounts()
    for s in stems:
        c8, i8, disk = cache[s]
        p = predict_full(model, planes_from_compact(c8, i8, disk), tta=tta, device=device)
        if save_dir is not None:
            np.savez_compressed(Path(save_dir) / f"{s}.npz", p=(p * 255 + 0.5).astype(np.uint8))
        pr = M.labels_to_rles(instances(p, cfg))
        for _, g in gt_by_stem[s]:
            c, _ = M.match_counts(pr, g)
            tot += c
    return tot


def partition_stems(stems, splits, fold):
    """Train on the other folds, validate on `fold`. Frames of one capture date never cross the boundary."""
    tr = [s for s in stems if splits[s] != fold]
    va = [s for s in stems if splits[s] == fold]
    if not tr or not va:
        raise ValueError("Empty train or validation partition")
    if {s[:8] for s in tr} & {s[:8] for s in va}:
        raise ValueError("Capture date crosses training/validation boundary")
    return tr, va


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(D.data_root()))
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--fold", type=int, default=0, help="validation fold; the other 4 folds train")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500, help="steps per epoch")
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--pct-start", type=float, default=0.05, help="OneCycle warm-up fraction")
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--val-max", type=int, default=0, help="limit validation images during training (0 = all)")
    ap.add_argument("--tversky-beta", type=float, default=0.5, help="Tversky weight of false negatives")
    ap.add_argument("--bicc-w", type=float, default=0.0, help="BiCC component loss weight (0 = off)")
    ap.add_argument("--bicc-alpha", type=float, default=0.5, help="annotation-side vs prediction-side balance")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to test the code")
    ap.add_argument("--require-cuda", action="store_true", help="fail before training when CUDA is unavailable")
    a = ap.parse_args(argv)
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.require_cuda and device != "cuda":
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(a.fold); np.random.seed(a.fold); random.seed(a.fold)

    readings, _ = D.load_train(Path(a.root))
    by_stem = D.readings_by_stem(readings)
    splits = json.load(open(a.splits))       # {stem: fold}
    stems = sorted(by_stem)
    tr, va = partition_stems(stems, splits, a.fold)
    if a.smoke:
        tr, va = tr[:6], va[:3]
        a.steps, a.epochs, a.bs, a.val_every = 4, 1, 2, 1
    json.dump({"train": tr, "validation": va, "excluded": sorted(set(stems) - set(tr) - set(va))},
              open(out / "partition.json", "w"), indent=1)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    print(f"device={device} train={len(tr)} val={len(va)}", flush=True)

    cache = build_cache(tr + va, a.root, a.workers)
    with Pool(a.workers) as p:
        targets = dict(p.map(_target_one, [(s, by_stem[s]) for s in tr], chunksize=4))
    weights = {s: len(by_stem[s]) for s in tr}   # the metric pools over readings
    gt_va = {s: [(r.reading_id, r.rles()) for r in by_stem[s]] for s in va}
    print(f"cache built {time.time() - t_start:.0f}s", flush=True)

    ds = TileDS(tr, cache, targets, weights, tile=a.tile, n=a.steps * a.bs, bicc=a.bicc_w > 0)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers if not a.smoke else 0,
                    pin_memory=True, drop_last=True, persistent_workers=not a.smoke)
    model = make_model(a.encoder, pretrained=not a.smoke).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=a.pct_start)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    alpha = 1 - a.tversky_beta
    best, hist = -1.0, []
    va_eval = va[: a.val_max] if a.val_max else va
    for ep in range(a.epochs):
        model.train(); tl = 0.0; t0 = time.time()
        for x, y, vor in dl:
            x, y, vor = x.to(device, non_blocking=True), y.to(device, non_blocking=True), vor.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logit = model(x)
            loss = loss_fn(logit.float(), y, vor, alpha, a.tversky_beta, a.bicc_w, a.bicc_alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        rec = {"epoch": ep, "loss": tl / a.steps, "sec": time.time() - t0}
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            c = evaluate(model, va_eval, cache, gt_va, device)
            rec.update(c.summary())
            if c.pq > best:                      # keep the checkpoint with the best validation PQ
                best = c.pq
                torch.save(model.state_dict(), out / "best.pth")
        hist.append(rec); print(json.dumps(rec), flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
    torch.save(model.state_dict(), out / "last.pth")
    model.load_state_dict(torch.load(out / "best.pth", map_location=device))
    od = out / "oof"; od.mkdir(exist_ok=True)
    c = evaluate(model, va, cache, gt_va, device, tta=4, save_dir=od)
    json.dump({"best_val_pq_tta1": best, "oof_tta4": c.summary()}, open(out / "final.json", "w"), indent=1)
    print("final", json.dumps(c.summary()), flush=True)


if __name__ == "__main__":
    main()
