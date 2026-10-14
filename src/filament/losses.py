"""Bidirectional connected-component (BiCC) loss.

Follows the idea of BiCC (arXiv 2609.30223): every annotated component and every false-positive
predicted component contributes one loss term of equal weight, independent of its size.

* Annotation side: the tile is split into Voronoi cells around the annotated components.
  Each cell gives one error, 1 - soft Dice, between the prediction and the target inside the cell.
* Prediction side: a connected component of (p > 0.5) that does not touch the annotation is a
  false-positive blob. Each blob gives one error, the mean predicted probability inside the blob.

This is our implementation of the idea, not the authors' code. The loss is added to BCE + Tversky in `train.py`.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi


def voronoi_cells(mask: np.ndarray) -> np.ndarray:
    """int16 map: id of the nearest annotated component (1..K); 0 if the tile has no annotation."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(mask.shape, np.int16)
    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    return lab[iy, ix].astype(np.int16)


def _region_mean(values, ids, n_ids):
    s = torch.zeros(n_ids, device=values.device, dtype=values.dtype).index_add_(0, ids, values)
    c = torch.zeros(n_ids, device=values.device, dtype=values.dtype).index_add_(0, ids, torch.ones_like(values))
    return s, c


def bicc_loss(logit: torch.Tensor, y: torch.Tensor, vor: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-object error rate, analogous to PQ's recognition term.

    logit, y: (B,1,H,W); vor: (B,1,H,W) Voronoi cell ids (0 = no annotation in the tile).
    Each annotated component k contributes e_k = 1 - softDice(p, y | cell k)  (in [0,1]).
    Each false-positive predicted blob j contributes f_j = mean p over the blob (in [0.5,1]).
    loss = 2 * (alpha * sum_k e_k + (1 - alpha) * sum_j f_j) / (K + J)
    """
    B = logit.shape[0]
    p = torch.sigmoid(logit)
    pf, yf = p.flatten(), y.flatten()
    # --- annotation side
    ids = vor.long().flatten(1)
    K = int(ids.max().item()) + 1
    gid = (ids + torch.arange(B, device=ids.device)[:, None] * K).flatten()
    n = B * K
    inter, _ = _region_mean(pf * yf, gid, n)
    psum, _ = _region_mean(pf, gid, n)
    ysum, _ = _region_mean(yf, gid, n)
    valid = (torch.arange(n, device=ids.device) % K != 0) & (ysum > 0)
    e = (1 - (2 * inter + 1) / (psum + ysum + 1))[valid]
    # --- prediction side (components found on CPU, no gradient through labelling)
    with torch.no_grad():
        pb = (p > 0.5).squeeze(1).cpu().numpy()
        yb = (y >= 0.5).squeeze(1).cpu().numpy()
        fp = np.zeros(pb.shape, np.int32); nfp = 0
        for b in range(B):
            lab, nc = ndi.label(pb[b], structure=np.ones((3, 3)))
            if nc == 0:
                continue
            touched = np.unique(lab[yb[b] & (lab > 0)])
            keep = np.ones(nc + 1, bool); keep[0] = False; keep[touched] = False
            remap = np.zeros(nc + 1, np.int32); remap[keep] = np.arange(1, keep.sum() + 1) + nfp
            fp[b] = remap[lab]; nfp += int(keep.sum())
    if nfp:
        fid = torch.from_numpy(fp).to(logit.device).long().flatten()
        s, c = _region_mean(pf, fid, nfp + 1)
        f = s[1:] / c[1:]
    else:
        f = pf[:0]
    nobj = e.numel() + f.numel()
    if nobj == 0:
        return logit.sum() * 0
    return 2 * (alpha * e.sum() + (1 - alpha) * f.sum()) / nobj
