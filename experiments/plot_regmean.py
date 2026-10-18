"""Figure: RegMean (budget C) vs number of calibration images per task, alpha = 1, dev.

Lines: plain solve, RegMean shrinkage a = 0.95, kappa cap 1e5 (ridge to the mean); band = range
over 3 calibration subsets. Reference lines: uniform (alpha tuned) and plain RegMean at n = 512.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [(("shrink", 1.0, None), "plain RegMean (a = 1)", "#e34948"),
          (("shrink", 0.95, None), "RegMean shrinkage a = 0.95", "#eb6834"),
          (("kcap_mean", 1.0, 1e5), "kappa cap 1e5 (ridge to mean)", "#2a78d6")]


def main() -> None:
    recs = [json.loads(line) for line in (RESULTS / "regmean" / "instability_C_dev.jsonl").read_text().splitlines()]
    tab = defaultdict(lambda: defaultdict(list))
    for r in recs:
        c = r["cfg"]
        if c["alpha"] == 1.0:
            tab[(c["method"], c["shrink"], c["kappa_target"])][c["n"]].append(100 * r["metrics"]["mean_norm"])
    ns = [16, 64, 512]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(77.65, ls="--", lw=1.5, color=MUTED, label="uniform average (alpha tuned)")
    for key, label, col in SERIES:
        d = tab[key]
        mean = np.array([np.mean(d[n]) for n in ns])
        lo = np.array([np.min(d[n]) for n in ns])
        hi = np.array([np.max(d[n]) for n in ns])
        ax.fill_between(ns, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(ns, mean, marker="o", ms=6, lw=2, color=col, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("calibration images per task (budget C, labels unused)", color=INK)
    ax.set_ylabel("mean normalised dev acc (%)", color=INK)
    ax.set_title("RegMean solve: instability with noisy statistics, fixed by any regulariser", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = RESULTS / "figures" / "regmean_instability.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
