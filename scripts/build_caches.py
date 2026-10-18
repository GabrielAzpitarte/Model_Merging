"""Build uint8 pixel caches for dev/test (and an optional train calibration pool)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import TASKS8, offline_env  # noqa: E402

offline_env()


def job(args):
    task, split, n_calib, workers = args
    from condmerge.data import build_cache

    meta = build_cache(task, split, n_calib=n_calib, workers=workers)
    print("done", meta, flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS8))
    ap.add_argument("--splits", default="dev,test,calib")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    jobs = [(t, s, args.n_calib, args.workers) for s in args.splits.split(",") for t in args.tasks.split(",")]
    with ProcessPoolExecutor(args.procs) as ex:
        for _ in ex.map(job, jobs):
            pass
    print("ALL DONE")


if __name__ == "__main__":
    main()
