"""Panoptic Quality as used by the competition (to be cross-checked with the
hosts' self-evaluation notebook).

    PQ = sum_{TP} IoU / (|TP| + 0.5|FP| + 0.5|FN|),  TP iff IoU > 0.5 (strict)

Ground truth is keyed by annotator reading; predictions by image stem.  Each
reading of an image is scored against the same prediction set, and counts are
pooled (micro) over all readings.  Readings for stems with no prediction count
all their GT as FN; prediction stems without any reading are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from pycocotools import mask as mask_utils

THR = 0.5


@dataclass
class PQCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    def __iadd__(self, o: "PQCounts"):
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn; self.iou_sum += o.iou_sum
        return self

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d else 0.0

    def summary(self) -> dict:
        return {**asdict(self), "pq": self.pq, "sq": self.sq, "rq": self.rq}


def iou_matrix(pred_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    if not pred_rles or not gt_rles:
        return np.zeros((len(pred_rles), len(gt_rles)))
    return np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)))


def match_counts(pred_rles: list[dict], gt_rles: list[dict]) -> tuple[PQCounts, np.ndarray]:
    ious = iou_matrix(pred_rles, gt_rles)
    c = PQCounts()
    if ious.size:
        hit = ious > THR
        c.tp = int(hit.sum())
        c.iou_sum = float(ious[hit].sum())
        c.fp = int((~hit.any(1)).sum())
        c.fn = int((~hit.any(0)).sum())
    else:
        c.fp = len(pred_rles)
        c.fn = len(gt_rles)
    return c, ious


def labels_to_rles(labels: np.ndarray) -> list[dict]:
    ids = [i for i in np.unique(labels) if i != 0]
    return [mask_utils.encode(np.asfortranarray((labels == i).astype(np.uint8))) for i in ids]


def score(pred: dict[str, list[dict]], gt: dict[str, tuple[str, list[dict]]],
          per_reading: bool = False):
    """pred: {stem: [rle,...]}; gt: {reading_id: (stem, [rle,...])}."""
    tot = PQCounts()
    rows = []
    for rid, (stem, g) in gt.items():
        c, _ = match_counts(pred.get(stem, []), g)
        tot += c
        if per_reading:
            rows.append({"reading_id": rid, "stem": stem, **asdict(c)})
    return (tot, rows) if per_reading else tot
