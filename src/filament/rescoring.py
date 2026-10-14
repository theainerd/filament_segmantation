"""Per-candidate features for the rescorer, which decides whether a candidate filament is kept (see pipeline.py)."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .metrics import PQCounts

PERMISSIVE = dict(t_high=0.5, t_low=0.3, close=0, fill_holes=False, join_px=8, min_area=30, min_peak=0.0, dilate=0, rmax=1.02)
FEATS = ["log_area", "mean_p", "max_p", "q50", "q90", "f50", "f70", "f90", "r", "elong", "fill", "perim_ratio",
         "log_nn_dist", "n_cands_img", "img_mean_p_on", "minor_len"]


def comp_features(p, lab, rr):
    """16 probability and shape features of every component of the label image `lab` (rows follow label ids 1..n)."""
    n = lab.max()
    if n == 0:
        return pd.DataFrame(columns=FEATS)
    idx = np.arange(1, n + 1)
    objs = ndi.find_objects(lab)
    rows = []
    cents = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx))
    if n > 1:
        d = np.sqrt(((cents[:, None] - cents[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf); nn = d.min(1)
    else:
        nn = np.array([2048.0])
    on = p[lab > 0].mean()
    for k, sl in enumerate(objs):
        m = lab[sl] == k + 1
        v = p[sl][m]
        a = m.sum()
        ys, xs = np.nonzero(m)
        cov = np.cov(np.vstack([ys, xs])) if a > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]; ev = np.maximum(ev, 1e-3)
        er = ndi.binary_erosion(m)
        perim = a - er.sum()
        cy, cx = cents[k]
        rows.append({"log_area": np.log(a), "mean_p": v.mean(), "max_p": v.max(), "q50": np.quantile(v, 0.5),
                     "q90": np.quantile(v, 0.9), "f50": (v > 0.5).mean(), "f70": (v > 0.7).mean(), "f90": (v > 0.9).mean(),
                     "r": rr[int(cy), int(cx)], "elong": np.sqrt(ev[0] / ev[1]),
                     "fill": a / (m.shape[0] * m.shape[1]), "perim_ratio": perim / a,
                     "log_nn_dist": np.log1p(nn[k]), "n_cands_img": n, "img_mean_p_on": on,
                     "minor_len": 4 * np.sqrt(ev[1])})
    return pd.DataFrame(rows)


def exact_counts(df, emit, n_gt):
    """Exact PQ counts when the candidates marked in `emit` (boolean mask over the rows of `df`) are kept.

    Candidates are disjoint, so TP pairs, matched readings and IoU sums add up per candidate.
    A kept candidate that matches no filament in a reading is a false positive in that reading.
    """
    e = df.loc[np.asarray(emit, dtype=bool)]
    tp = int(e.tp_pairs.sum())
    return PQCounts(tp, int((e.n_read - e.matched_reads).sum()), int(n_gt - tp), float(e.iou_sum.sum()))
