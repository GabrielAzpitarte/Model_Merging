"""Cap the BLAS thread pools. Import this **before** numpy or torch, always.

Why this file exists rather than a line in each script: numpy, OpenBLAS, MKL
and torch each read their thread-count environment variable exactly once, at
import time, and default it to "one thread per core". On a 128-core shared
machine that means every process claims all 128 -- so N concurrent processes
ask for 128 N threads and spend their time in the scheduler instead of in
BLAS.

Measured on this project, five concurrent evaluation processes: 387 threads
each, load average 392, and three of them completed **zero** work units in 55
minutes. Capped, the same processes used 2.3 cores each and ran twice as fast.
The machine's other users were paying for it too.

The right cap is not one number, it is one per *shape of parallelism*:

* ``WORKER_POOL`` -- the script already forks one worker per module or per
  shard, so each worker should take a single thread and let the pool provide
  the parallelism. Anything above 1 multiplies.
* ``FEW_PROCESSES`` -- a handful of long-lived processes, one per GPU. Measured
  need: 2.3 cores during the grid (the CPU work per evaluation is 84 ms of
  reconstruction), and joint diagonalisation does **not** scale with threads at
  all -- 36.6 s on one against 31.9 s on sixteen, because it is a Jacobi loop
  over tiny einsums and is bound by interpreter latency, not by BLAS. So this
  is headroom over the 2.3, not a target.
* ``SINGLE`` -- one process doing the whole job.

An environment variable already set by the caller always wins, so a one-off
run can still override from the command line.
"""

from __future__ import annotations

import os

__all__ = ["cap_threads", "WORKER_POOL", "FEW_PROCESSES", "SINGLE"]

WORKER_POOL = 1
FEW_PROCESSES = 4
SINGLE = 8

_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cap_threads(n: int = FEW_PROCESSES) -> int:
    """Set every BLAS thread variable to ``n`` unless already set. Returns it."""
    if n < 1:
        raise ValueError(f"Thread cap must be at least 1; got {n}")
    for var in _VARS:
        os.environ.setdefault(var, str(n))
    return int(os.environ[_VARS[0]])
