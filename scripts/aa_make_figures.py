"""Figures for docs/aa/AA_SIDE_REPORT.md (static PNGs, light theme; values are read from the result files).

fig_shared_target.png  mean own-task dev accuracy of same-task soups per rule, three pools (small multiples)
fig_tail_stress.png    literal-rule lambda and merged accuracy before/after a function-null tail edit
fig_oracle_heatmap.png full-FT oracle lambda (73 maps x 8 experts) with the argmin-kappa expert marked
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS, TASKS8  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY, GRAY_D = "#2a78d6", "#eb6834", "#a3a29c", "#6b6a65"
OUT = RESULTS / "aa" / "figures"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def soup_means(fn: str) -> dict[str, float]:
    rows = [json.loads(l) for l in (RESULTS / "aa" / "soup" / f"{fn}.jsonl").read_text().splitlines() if l.strip()]
    tasks = list(dict.fromkeys(r["task"] for r in rows))

    def get(t, rule, b=None):
        for r in rows:
            if r["task"] == t and r["rule"] == rule and (b is None or r["b"] == b or
                                                          (isinstance(b, float) and math.isinf(b) and isinstance(r["b"], float) and math.isinf(r["b"]))):
                return r["acc"]
        return float("nan")

    oracle = "oracle_invvar" if any(r["rule"] == "oracle_invvar" for r in rows) else "oracle_quality_softmax100"
    spec = [("uniform", "uniform", None), ("oracle", oracle, None),
            ("literal κ, b=1", "kappa", 1), ("literal κ, b=16", "kappa", 16), ("literal κ, argmin", "kappa", math.inf),
            ("GD inverse variance (tuning-free)", "mp_invvar", None), ("update stable rank, b=4", "dsrank", 4),
            ("update norm, b=4", "dnorm", 4)]
    return {lab: 100 * float(np.nanmean([get(t, rule, b) for t in tasks])) for lab, rule, b in spec}


def fig_shared_target():
    pools = [("partA_controlled_noise_strong_v2", "A: isotropic noise"),
             ("partA_controlled_noise_strong_normmatched", "A-nm: noise, norms matched"),
             ("partB_lora_variants_v2", "B: 13 real LoRA variants")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    for ax, (fn, title) in zip(axes, pools):
        m = soup_means(fn)
        labs = list(m)[::-1]
        vals = [m[k] for k in labs]
        cols = [BLUE if "κ" in k else ORANGE if ("GD" in k or "stable" in k) else GRAY_D if k == "oracle" else GRAY for k in labs]
        lo = min(vals) - 2
        ax.barh(range(len(labs)), [v - lo for v in vals], left=lo, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.axvline(m["uniform"], color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs if ax is axes[0] else [""] * len(labs), fontsize=8, color=INK)
        ax.set_xlim(lo, max(vals) + 1.8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mean own-task dev accuracy (%), 8 tasks", fontsize=8, color=INK2)
        style(ax)
    fig.suptitle("Same-task soups: the literal κ rule is anti-inverse-variance; update-spectrum shape works only under isotropic noise",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shared_target.png", dpi=160, facecolor=SURFACE)


def fig_tail_stress():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=SURFACE)
    rows = []
    for e in ("svhn", "sun397"):
        p = RESULTS / "aa" / f"tail_stress_{e}.json"
        if p.exists():
            d = json.loads(p.read_text())
            for b in (4, 16, "inf"):
                o = next(r for r in d["rule"] if str(r["b"]) in (str(b), "Infinity" if b == "inf" else str(b)) and r["alpha"] == 1.6 and r["version"] == "original") if b != "inf" else \
                    next(r for r in d["rule"] if isinstance(r["b"], float) and math.isinf(r["b"]) and r["version"] == "original")
                t = next(r for r in d["rule"] if r["b"] == o["b"] and r["alpha"] == o["alpha"] and r["version"] == "tail-edited")
                rows.append((f"{e.upper()}  b={'∞' if b == 'inf' else b}", o, t))
    labs = [r[0] for r in rows][::-1]
    for ax, key, xlab in ((axes[0], "mean_lambda_j", "literal-rule mean λ of the edited expert"),
                          (axes[1], "avg_acc", "merged dev accuracy (%)")):
        scale = 100 if key == "avg_acc" else 1
        for i, (lab, o, t) in enumerate(rows[::-1]):
            a, b = o[key] * scale, t[key] * scale
            ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
            ax.scatter([a], [i], s=40, color=GRAY, zorder=2, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([b], [i], s=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(b, i + 0.28, f"{b:.2f}" if scale == 1 else f"{b:.1f}", fontsize=7.5, color=INK, ha="center")
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs if ax is axes[0] else [""] * len(labs), fontsize=8, color=INK)
        ax.set_xlabel(xlab, fontsize=8, color=INK2)
        style(ax)
    axes[0].scatter([], [], color=GRAY, label="original expert")
    axes[0].scatter([], [], color=BLUE, label="tail-edited (accuracy unchanged, edit = 0.5% of ‖Δ‖)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    fig.suptitle("A function-null spectral-tail edit swings the literal rule (α = 1.6)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_tail_stress.png", dpi=160, facecolor=SURFACE)


def fig_oracle_heatmap():
    rec = json.loads((RESULTS / "aa" / "oracle_lambda" / "layerwise_alpha_seed0.json").read_text())
    st = np.load(CACHE / "aa" / "fullft_stats.npz")
    names = list(st["names"])
    order = sorted(range(len(names)), key=lambda i: (-1 if "patch" in names[i] else int(names[i].split(".")[2]),
                                                     ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch"].index(
                                                         next(k for k in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch"] if k in names[i]))))
    lam = np.array([rec["lambda"][names[i]] for i in order])
    amin = np.array([st["kappa_W"][i].argmin() for i in order])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue1", ["#f7f9fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(13, 3.2), facecolor=SURFACE)
    im = ax.imshow(lam.T, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.scatter(np.arange(len(order)), amin, s=9, color=ORANGE, marker="o", label="argmin-κ expert (literal rule, b=∞)")
    short = ["patch" if "patch" in names[i] else f"L{names[i].split('.')[2]}.{[k for k in ['q','k','v','out','fc1','fc2'] if (k + '_proj' if k in 'qkv' or k == 'out' else k) in names[i]][0]}" for i in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short, rotation=90, fontsize=5.5, color=INK2)
    ax.set_yticks(range(8))
    ax.set_yticklabels(TASKS8, fontsize=8, color=INK)
    ax.set_title(f"Full-FT oracle allocation (fit on dev labels; α={rec['alpha']:.2f}, test {100 * rec['test']['avg_acc']:.1f}%): attention maps go to one expert; κ points elsewhere",
                 fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.set_label("oracle λ", fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.18), labelcolor=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_oracle_heatmap.png", dpi=170, facecolor=SURFACE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_shared_target()
    fig_tail_stress()
    fig_oracle_heatmap()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
