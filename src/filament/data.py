"""MAGFiLO Kaggle-2026 data access.

Expected layout (root = MAGFiLO_1.0_Kaggle_2026):
    train/train_images/<stem>.jpeg
    train/MAGFiLO_1.0_Annotations_kaggle2026_train.json   (COCO)
    test/test_images/<stem>.jpeg
COCO ``images[].id`` is "<annotator>-<stem>" -> one entry per annotator reading.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

H = W = 2048


def data_root() -> Path:
    return Path(os.environ.get("MAGFILO_ROOT", "data/MAGFiLO_1.0_Kaggle_2026"))


@dataclass
class Reading:
    reading_id: str          # "<annotator>-<stem>"
    stem: str
    annotator: str
    height: int
    width: int
    anns: list = field(default_factory=list)   # raw COCO annotation dicts

    def rles(self) -> list[dict]:
        out = []
        for a in self.anns:
            seg = a["segmentation"]
            if isinstance(seg, list):
                r = mask_utils.merge(mask_utils.frPyObjects(seg, self.height, self.width))
            elif isinstance(seg["counts"], list):
                r = mask_utils.frPyObjects(seg, self.height, self.width)
            else:
                r = seg
            out.append(r)
        return out

    def masks(self) -> np.ndarray:
        rl = self.rles()
        if not rl:
            return np.zeros((0, self.height, self.width), bool)
        return mask_utils.decode(rl).transpose(2, 0, 1).astype(bool)

    def union(self) -> np.ndarray:
        m = self.masks()
        return m.any(0) if len(m) else np.zeros((self.height, self.width), bool)


def stem_of(file_name: str) -> str:
    return Path(file_name).stem


def load_train(root: Path | None = None) -> tuple[dict[str, Reading], dict]:
    root = root or data_root()
    jp = next((root / "train").glob("*.json"))
    coco = json.load(open(jp))
    readings: dict[str, Reading] = {}
    for im in coco["images"]:
        rid = str(im["id"])
        stem = stem_of(im["file_name"])
        ann = rid.split("-", 1)[0] if "-" in rid else "0"
        readings[rid] = Reading(rid, stem, ann, im.get("height", H), im.get("width", W))
    for a in coco["annotations"]:
        readings[str(a["image_id"])].anns.append(a)
    return readings, coco


def readings_by_stem(readings: dict[str, Reading]) -> dict[str, list[Reading]]:
    d = defaultdict(list)
    for r in readings.values():
        d[r.stem].append(r)
    return dict(d)


def image_path(stem: str, split: str = "train", root: Path | None = None) -> Path:
    root = root or data_root()
    return root / split / f"{split}_images" / f"{stem}.jpeg"


def load_image(stem: str, split: str = "train", root: Path | None = None) -> np.ndarray:
    im = Image.open(image_path(stem, split, root))
    return np.asarray(im.convert("L"))


def test_stems(root: Path | None = None) -> list[str]:
    root = root or data_root()
    return sorted(p.stem for p in (root / "test" / "test_images").glob("*.jp*g"))
