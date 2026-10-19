"""Pixel caches for the uncertainty diagnostic (protocol 02 sec. 9): disjoint TRAIN images per task.

From each task's official train split, excluding the 512-image calibration pool of the main study
(data.calib_indices, seed 1), a seed-7 permutation gives three disjoint sets:
  uref   1024 images  large-calibration reference (diagnosis only; never used by a small-calibration method)
  upool   640 images  pool from which the small calibration subsets are drawn (disjoint subsets)
  uind    256 images  independent activations on which correction errors are measured
Labels are stored for bookkeeping only and are never read by the diagnostic.
Usage: build_cache.py [--procs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from condmerge.paths import TASKS8, offline_env  # noqa: E402

offline_env()

SIZES = {"uref": 1024, "upool": 640, "uind": 256}


def job(args):
    task, workers = args
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPImageProcessor

    from condmerge.data import cache_paths, calib_indices, load_hf_split, pixel_dir

    ds = load_hf_split(task, "train")
    n = len(ds)
    calib = set(calib_indices(n, 512).tolist())
    rest = np.asarray([i for i in range(n) if i not in calib])
    perm = rest[np.random.default_rng(7).permutation(len(rest))]
    start = 0
    out = {}
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for split, size in SIZES.items():
        idx = np.sort(perm[start:start + size])
        start += size
        xp, yp, ip = cache_paths(task, split)
        if xp.exists() and yp.exists() and ip.exists():
            out[split] = "cached"
            continue
        sub = ds.select(idx.tolist())
        key = "image" if "image" in sub.features else next(k for k in sub.features if k != "label")

        def collate(rows):
            px = proc([r[key].convert("RGB") for r in rows], return_tensors="pt", do_rescale=False,
                      do_normalize=False)["pixel_values"]
            return px, torch.tensor([int(r["label"]) for r in rows])

        x = np.zeros((len(idx), 3, 224, 224), np.uint8)
        y = np.zeros(len(idx), np.int64)
        pos = 0
        for px, lb in DataLoader(sub, batch_size=128, num_workers=workers, collate_fn=collate):
            x[pos:pos + px.shape[0]] = px.numpy()
            y[pos:pos + px.shape[0]] = lb.numpy()
            pos += px.shape[0]
        assert pos == len(idx)
        np.save(str(xp) + ".tmp.npy", x)
        Path(str(xp) + ".tmp.npy").rename(xp)
        np.save(yp, y)
        np.save(ip, idx)
        (pixel_dir() / f"{task}_{split}_meta.json").write_text(json.dumps({"task": task, "split": split, "n": len(idx),
                                                                          "source": "train minus calib, seed-7 perm"}))
        out[split] = len(idx)
    return task, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.procs) as ex:
        for task, out in ex.map(job, [(t, args.workers) for t in TASKS8]):
            print(task, out, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
