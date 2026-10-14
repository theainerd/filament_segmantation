"""Pipeline steps called by notebooks/fsc26_pipeline.ipynb.

Each function is one step of the method: fold split, training, out-of-fold (OOF) blend,
candidate extraction, rescorer fit, final mask rule, cross-validation and test inference.
"""
from __future__ import annotations

import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import data as D
from . import metrics as M
from .postprocess import PPConfig, instances
from .preprocess import fit_disk, radius_map
from .rescoring import FEATS, PERMISSIVE, comp_features, exact_counts

FAMILIES = {"b3": [], "bicc": ["--bicc-w", "0.3"]}            # B3 U-Net, and B3 U-Net + BiCC loss
TRAIN_ARGS = ["--encoder", "efficientnet-b3", "--epochs", "30", "--steps", "500", "--bs", "12", "--lr", "4e-4"]
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
CORE_T = 0.5                                                   # final masks keep pixels with p > CORE_T
_G: dict = {}


# ----------------------------------------------------------------- folds
def make_splits(by_stem: dict, k: int = 5, seed: int = 2026) -> dict:
    """k folds grouped by capture date; groups balanced by reading count within each year."""
    groups = defaultdict(list)
    for s in by_stem:
        groups[s[:8]].append(s)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda g: -sum(len(by_stem[s]) for s in groups[g]))
    load, year_load, assign = [0] * k, defaultdict(lambda: [0] * k), {}
    for g in keys:
        w = sum(len(by_stem[s]) for s in groups[g])
        f = min(range(k), key=lambda i: (year_load[g[:4]][i], load[i]))
        for s in groups[g]:
            assign[s] = f
        load[f] += w
        year_load[g[:4]][f] += w
    return assign


# ----------------------------------------------------------------- training
def train_fold(family: str, fold: int, out: Path, splits_path: Path, data_root: Path,
               workers: int = 4, smoke: bool = False) -> Path:
    """Train one model on 4 folds; writes best.pth and the held-out fold's TTA-4 maps to out/oof."""
    from .train import main
    argv = ["--fold", str(fold), "--out", str(out), "--splits", str(splits_path), "--root", str(data_root),
            "--workers", str(workers), *TRAIN_ARGS, *FAMILIES[family]] + (["--smoke"] if smoke else [])
    main(argv)
    return out


def load_map(path: Path) -> np.ndarray:
    a = np.load(path)
    return (a["p"] if path.suffix == ".npz" else a).astype(np.float32) / 255


def map_path(d: Path, stem: str) -> Path | None:
    for e in (".npz", ".npy"):
        if (Path(d) / f"{stem}{e}").exists():
            return Path(d) / f"{stem}{e}"
    return None


def u8(p: np.ndarray) -> np.ndarray:
    """8-bit rounding used for every saved map, so saved and in-memory maps give the same masks."""
    return (p * 255 + 0.5).astype(np.uint8).astype(np.float32) / 255


def blend_oof(b3_dirs: list[Path], bicc_dirs: list[Path], out: Path) -> int:
    """Mean of the two families' OOF maps for every image that both families predicted."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for a, b in zip(b3_dirs, bicc_dirs):
        for f in sorted(Path(a).glob("*.np[yz]")):
            g = map_path(b, f.stem)
            if g is None:
                continue
            p = (load_map(f) + load_map(g)) / 2
            np.savez_compressed(out / f"{f.stem}.npz", p=(p * 255 + 0.5).astype(np.uint8))
            n += 1
    return n


# ----------------------------------------------------------------- candidates and rescorer
def _init(g):
    _G.update(g)


def _extract(stem):
    p = load_map(map_path(_G["maps"], stem))
    rr = radius_map(p.shape, fit_disk(D.load_image(stem, "train", _G["root"])))
    lab = instances(p, PPConfig(**PERMISSIVE), rr)
    df = comp_features(p, lab, rr)
    pr, gts = M.labels_to_rles(lab), _G["gt"][stem]
    gain, pairs, matched = np.zeros(len(pr)), np.zeros(len(pr), int), np.zeros(len(pr))
    for g in gts:
        iou = M.iou_matrix(pr, g)
        if iou.size:
            gain += (iou * (iou > 0.5)).sum(1)
            pairs += (iou > 0.5).sum(1)
            matched += iou.max(1) > 0.5
    df = df.assign(gain=gain / max(len(gts), 1), iou_sum=gain, tp_pairs=pairs, matched_reads=matched,
                   n_read=len(gts), stem=stem)
    return df, sum(len(g) for g in gts)


def extract_candidates(maps: Path, data_root: Path, by_stem: dict, stems: list[str], workers: int = 4):
    """Permissive candidates of every OOF map, with 16 map features and the per-reading PQ gain target."""
    gt = {s: [r.rles() for r in by_stem[s]] for s in stems}
    with Pool(workers, initializer=_init, initargs=(dict(maps=Path(maps), root=Path(data_root), gt=gt),)) as pool:
        res = pool.map(_extract, stems, chunksize=2)
    return pd.concat([r[0] for r in res], ignore_index=True), {s: r[1] for s, r in zip(stems, res)}


def fit_rescorer(cands: pd.DataFrame, n_gt: dict, taus=np.round(np.linspace(0, 0.6, 61), 2)):
    """Fit the gain regressor; choose the keep threshold tau that maximises pooled PQ on stem-grouped OOF predictions."""
    X, y, grp = cands[FEATS].values, cands.gain.values, cands.stem.values
    oof = np.zeros(len(cands))
    for tr, te in GroupKFold(5).split(X, y, grp):
        oof[te] = HistGradientBoostingRegressor(**HGB).fit(X[tr], y[tr]).predict(X[te])
    total = sum(n_gt.values())
    curve = pd.Series([exact_counts(cands, oof > t, total).pq for t in taus], index=taus)
    model = HistGradientBoostingRegressor(**HGB).fit(X, y)
    return {"model": model, "feats": FEATS, "tau": float(curve.idxmax()), "permissive": PERMISSIVE}, curve


def final_labels(p: np.ndarray, rr: np.ndarray, bundle: dict, model=None) -> np.ndarray:
    """Final instance labels: permissive candidates -> rescorer keep/drop -> keep the p > CORE_T core of each kept filament."""
    lab = instances(p, PPConfig(**bundle["permissive"]), rr)
    if lab.max() == 0:
        return lab
    keep = (model or bundle["model"]).predict(comp_features(p, lab, rr)[bundle["feats"]].values) > bundle["tau"]
    remap = np.zeros(lab.max() + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
    core = lab * (p > CORE_T)
    for i in np.setdiff1d(np.unique(lab), np.unique(core)):     # a filament with no core pixel keeps its full mask
        core[lab == i] = i
    return core


# ----------------------------------------------------------------- cross-validation
def _cv_one(stem):
    p = load_map(map_path(_G["maps"], stem))
    rr = radius_map(p.shape, fit_disk(D.load_image(stem, "train", _G["root"])))
    pr = M.labels_to_rles(final_labels(p, rr, _G["bundle"], _G["models"][_G["splits"][stem]]))
    tot = M.PQCounts()
    for g in _G["gt"][stem]:
        tot += M.match_counts(pr, g)[0]
    return stem, tot


def cross_validate(maps: Path, data_root: Path, by_stem: dict, splits: dict, cands: pd.DataFrame, bundle: dict,
                   stems: list[str], workers: int = 4, n_boot: int = 2000):
    """Out-of-fold PQ of the full pipeline. For each fold the rescorer is refitted without that fold."""
    fold = cands.stem.map(splits)
    models = {f: HistGradientBoostingRegressor(**HGB).fit(cands.loc[fold != f, FEATS].values, cands.loc[fold != f, "gain"].values)
              for f in sorted(set(splits[s] for s in stems))}
    g = dict(maps=Path(maps), root=Path(data_root), gt={s: [r.rles() for r in by_stem[s]] for s in stems},
             bundle=bundle, models=models, splits=splits)
    with Pool(workers, initializer=_init, initargs=(g,)) as pool:
        per = dict(pool.map(_cv_one, stems, chunksize=2))
    rows = pd.DataFrame([dict(stem=s, fold=splits[s], tp=c.tp, fp=c.fp, fn=c.fn, iou=c.iou_sum) for s, c in per.items()])
    pq = lambda r: r.iou.sum() / (r.tp.sum() + r.fp.sum() / 2 + r.fn.sum() / 2)
    rng = np.random.default_rng(0)
    boot = [pq(rows.iloc[rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    summary = dict(pq=pq(rows), ci95=np.percentile(boot, [2.5, 97.5]).round(4).tolist(), tp=int(rows.tp.sum()),
                   fp=int(rows.fp.sum()), fn=int(rows.fn.sum()), sq=rows.iou.sum() / rows.tp.sum(),
                   per_fold=rows.groupby("fold").apply(pq, include_groups=False).round(4).to_dict())
    return summary, rows


# ----------------------------------------------------------------- inference
def blend_map(b3_models, bicc_models, img: np.ndarray, device: str, tta: int = 4):
    """Probability map of one image: mean over each family's 5 folds (TTA), then mean of the two families."""
    from .predict import ensemble_map
    p_b3, disk = ensemble_map(b3_models, img, device, tta)
    p_bicc, _ = ensemble_map(bicc_models, img, device, tta)
    return (u8(p_b3) + u8(p_bicc)) / 2, disk


def fetch_release(url: str, dest: Path) -> None:
    """Download the GitHub release archive and unpack it into `dest` (weights/ and oof_blend/)."""
    import tempfile, urllib.request, zipfile
    with tempfile.TemporaryDirectory() as t:
        z = Path(t) / "release.zip"
        urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(dest)
