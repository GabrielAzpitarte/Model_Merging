"""Figure: mean normalised dev accuracy vs temperature multiplier b for the literal rule and controls.

Panels (small multiples, same metric): alpha = 1 (literal normalised average) and
best alpha per b (matched alpha grid); bottom row: coefficient concentration n_eff.
Shuffled-score null = 5 seeds (band = min..max, line = mean).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_sweep import load, table  # noqa: E402

from condmerge.paths import RESULTS  # noqa: E402

COLORS = {"kappa": "#2a78d6", "kappa_rev": "#eb6834", "logkappa": "#1baf7a", "kappa_z": "#6250d6",
          "dnorm_rev": "#008300", "kappa_strict": "#e87ba4", "dnorm": "#e34948", "global:kappa_z": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def series(d: dict, bs: list[float], alpha: float | None, key=lambda r: r["metrics"]["mean_norm"]):
    ys = []
    for b in bs:
        rs = [r for (bb, a), r in d.items() if bb == b and (alpha is None or a == alpha)]
        ys.append(max(key(r) for r in rs) if rs else np.nan)
    return np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "sweeps" / "rule_sweep_dev.jsonl"))
    ap.add_argument("--rules", default="kappa,kappa_rev,logkappa,kappa_z,dnorm_rev")
    ap.add_argument("--null", default="kappa")
    ap.add_argument("--out", default=str(RESULTS / "figures" / "pilot_rule_vs_b.png"))
    args = ap.parse_args()
    by = table(load(Path(args.file)))
    bs = sorted({k[0] for r in by.values() for k in r if k[0] > 0}, key=lambda x: (math.isinf(x), x))
    xs = np.arange(len(bs))
    xt = ["inf" if math.isinf(b) else f"{b:g}" for b in bs]
    uni = by["uniform"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for col, alpha in enumerate((1.0, None)):
        ax = axes[0, col]
        u = uni[(0.0, 1.0)]["metrics"]["mean_norm"] if alpha == 1.0 else max(r["metrics"]["mean_norm"] for r in uni.values())
        ax.axhline(100 * u, ls="--", lw=1.5, color=MUTED, label="uniform (beta=0)" + ("" if alpha else ", best alpha"))
        shuf = [r for r in by if r.startswith("shuf") and r.endswith(":" + args.null)]
        if shuf:
            m = np.stack([series(by[r], bs, alpha) for r in shuf])
            ax.fill_between(xs, 100 * m.min(0), 100 * m.max(0), color="#b9b8b2", alpha=0.45, lw=0,
                            label=f"shuffled {args.null} ({len(shuf)} seeds, min-max)")
            ax.plot(xs, 100 * m.mean(0), color="#8a8984", lw=1.5)
        for rule in args.rules.split(","):
            if rule in by:
                ax.plot(xs, 100 * series(by[rule], bs, alpha), marker="o", ms=5, lw=2, color=COLORS.get(rule, INK), label=rule)
        ax.set_title("alpha = 1 (normalised average)" if alpha == 1.0 else "best alpha per b (shared alpha grid)",
                     fontsize=11, color=INK)
        ax.set_ylabel("mean normalised dev acc (%)", color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        axn = axes[1, col]
        for rule in args.rules.split(","):
            if rule in by:
                ne = series(by[rule], bs, 1.0, key=lambda r: r["coeff_summary"].get("n_eff_mean", np.nan))
                axn.plot(xs, ne, marker="o", ms=4, lw=1.5, color=COLORS.get(rule, INK))
        axn.set_ylabel("n_eff (of 8)", color=INK)
        axn.set_xticks(xs, xt)
        axn.set_xlabel("temperature multiplier b  (beta = b / median within-map score sd)", color=INK)
        axn.grid(color=GRID, lw=0.8)
        axn.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Literal condition-number rule vs controls, 8 FusionBench ViT-B/32 experts (dev)", fontsize=12, color=INK)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
