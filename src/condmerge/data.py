"""Splits and pixel caches.

Split convention (identical to the earlier ICA project, so the same images are
used): for each task, the official *test* split is permuted with
``np.random.default_rng(0)``; the first 20 % (rounded) is **dev** (all method
selection), the remaining 80 % is **test** (frozen configurations only).
The experts were fine-tuned on the official train splits, so neither dev nor
test images were seen in training.

Pixels are cached as uint8 after CLIP's resize + center crop (exactly what
``CLIPImageProcessor`` produces before rescale/normalize; verified max abs
difference 2.4e-7 after normalization). Normalization happens on the GPU.

A small *calibration* pool can be drawn from each task's official *train* split
(unlabeled use only; labels are stored solely for diagnostics and must never be
read by a method that claims label-free access).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paths import CACHE, DATASETS

DEV_FRACTION = 0.2
SPLIT_SEED = 0
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pixel_dir() -> Path:
    d = CACHE / "pixels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_hf_split(task: str, split: str):
    from datasets import load_dataset

    spec = DATASETS[task]
    return load_dataset(spec["path"], spec.get("name"), split=split)


def dev_test_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_dev = int(round(DEV_FRACTION * n))
    return perm[:n_dev], perm[n_dev:]


def calib_indices(n: int, n_calib: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n_calib, n), replace=False))


def cache_paths(task: str, split: str) -> tuple[Path, Path, Path]:
    d = pixel_dir()
    return d / f"{task}_{split}_x.npy", d / f"{task}_{split}_y.npy", d / f"{task}_{split}_idx.npy"


def classnames(task: str) -> list[str]:
    ds = load_hf_split(task, "test")
    return list(ds.features["label"].names)


def build_cache(task: str, split: str, n_calib: int = 512, workers: int = 16, batch: int = 256) -> dict:
    """Decode + resize + crop once, store uint8 arrays. ``split`` in {dev,test,calib}."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPImageProcessor

    xp, yp, ip = cache_paths(task, split)
    if xp.exists() and yp.exists() and ip.exists():
        return {"task": task, "split": split, "n": int(np.load(yp).shape[0]), "cached": True}
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if split in ("dev", "test"):
        ds = load_hf_split(task, "test")
        dev, test = dev_test_indices(len(ds))
        idx = dev if split == "dev" else test
    elif split == "calib":
        ds = load_hf_split(task, "train")
        idx = calib_indices(len(ds), n_calib)
    else:
        raise ValueError(split)
    sub = ds.select(idx.tolist())
    key = "image" if "image" in sub.features else next(k for k in sub.features if k != "label")

    def collate(rows):
        px = proc([r[key].convert("RGB") for r in rows], return_tensors="pt",
                  do_rescale=False, do_normalize=False)["pixel_values"]
        assert px.dtype == torch.uint8, px.dtype
        return px, torch.tensor([int(r["label"]) for r in rows])

    loader = DataLoader(sub, batch_size=batch, num_workers=workers, collate_fn=collate)
    n = len(sub)
    x = np.lib.format.open_memmap(str(xp) + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 224, 224))
    y = np.empty(n, dtype=np.int64)
    pos = 0
    for px, lb in loader:
        b = px.shape[0]
        x[pos:pos + b] = px.numpy()
        y[pos:pos + b] = lb.numpy()
        pos += b
    assert pos == n
    x.flush()
    del x
    Path(str(xp) + ".tmp.npy").rename(xp)
    np.save(yp, y)
    np.save(ip, idx)
    meta = {"task": task, "split": split, "n": n, "n_classes": len(ds.features["label"].names)}
    (pixel_dir() / f"{task}_{split}_meta.json").write_text(json.dumps(meta))
    return meta


def load_cache(task: str, split: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xp, yp, _ = cache_paths(task, split)
    x = np.load(xp, mmap_mode="r" if mmap else None)
    y = np.load(yp)
    return x, y
