import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from pycocotools import mask as mu

from filament import rle, metrics
from filament.postprocess import PPConfig, instances


def _rle(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    m = np.zeros((2048, 2048), bool)
    m[100:300, 400:420] = True
    m[rng.integers(0, 2048, 500), rng.integers(0, 2048, 500)] = True
    c = rle.encode(m)
    assert (rle.decode(c) == m).all()


def test_pq_basic():
    a = np.zeros((64, 64), bool); a[10:20, 10:20] = True
    b = np.zeros((64, 64), bool); b[10:20, 12:20] = True     # IoU 0.8 with a
    c = np.zeros((64, 64), bool); c[40:50, 40:50] = True     # unmatched GT
    d = np.zeros((64, 64), bool); d[0:5, 50:60] = True       # unmatched pred
    gt = {"A-x": ("x", [_rle(a), _rle(c)]), "B-x": ("x", [_rle(a)])}
    pred = {"x": [_rle(b), _rle(d)]}
    tot = metrics.score(pred, gt)
    # reading A: TP=1 (0.8) FP=1 FN=1 ; reading B: TP=1 (0.8) FP=1 FN=0
    assert (tot.tp, tot.fp, tot.fn) == (2, 2, 1)
    assert abs(tot.pq - 1.6 / (2 + 1 + 0.5)) < 1e-9


def test_strict_threshold():
    a = np.zeros((10, 10), bool); a[:, :4] = True
    b = np.zeros((10, 10), bool); b[:, :2] = True            # IoU exactly 0.5
    c, _ = metrics.match_counts([_rle(b)], [_rle(a)])
    assert c.tp == 0 and c.fp == 1 and c.fn == 1


def test_postprocess_no_overlap():
    p = np.zeros((256, 256), np.float32)
    p[50:60, 20:200] = 0.9
    p[63:70, 20:200] = 0.9
    lab = instances(p, PPConfig(min_area=10, dilate=3))
    assert lab.max() == 2
    lab2 = instances(p, PPConfig(min_area=10, join_px=6))
    assert lab2.max() == 1


def test_losses():
    import torch
    from filament.losses import bicc_loss, voronoi_cells
    from filament.train import loss_fn
    m = np.zeros((128, 128), bool); m[40:46, 10:120] = True; m[90:95, 90:96] = True
    cells = voronoi_cells(m)
    assert cells.max() == 2 and (cells[m] > 0).all()                     # one Voronoi cell per annotated component
    vor = torch.from_numpy(cells.astype(np.float32))[None, None]
    y = torch.from_numpy(m[None, None].astype(np.float32))
    good = torch.where(y > 0, 5.0, -5.0); bad = -good
    assert bicc_loss(good, y, vor) < bicc_loss(bad, y, vor)
    assert loss_fn(good, y, vor, bicc_w=0.3) < loss_fn(bad, y, vor, bicc_w=0.3)
    assert loss_fn(good, y, vor) < 0.1


def test_exact_counts():
    import pandas as pd
    from filament.rescoring import exact_counts
    df = pd.DataFrame(dict(tp_pairs=[1, 0], n_read=[2, 2], matched_reads=[1, 0], iou_sum=[0.8, 0.0]))
    c = exact_counts(df, [True, False], n_gt=3)                           # keep only the first candidate
    assert (c.tp, c.fp, c.fn) == (1, 1, 2) and abs(c.iou_sum - 0.8) < 1e-9


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
