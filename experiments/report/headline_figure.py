"""Headline figure: the professor's literal rule vs tuned uniform and shuffled-score nulls, on every bank tested.

One row per bank and split; the x axis is the difference to tuned uniform merging IN THAT BANK'S OWN UNITS (marked on
the row: normalised points for full FT and LLMs, absolute accuracy points for the CLIP LoRA bank). Blue = literal
kappa with its own dev-selected (b, alpha) and 95 % paired-bootstrap CI where available; grey = shuffled-score nulls
(identical per-matrix concentration; tuned on dev or evaluated at kappa's frozen config).
Reads the frozen-test records of the CLIP study and results/llm/summary/S*/summary.json; missing rows are skipped.
Writes results/figures/fig_headline_kappa_vs_nulls.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def rows() -> list[dict]:
    out = []
    tc = load(R / "freeze" / "test_comparison.json")
    if tc:
        r = tc["results"]
        k = r["kappa|tuned"]
        out.append({"label": "CLIP ViT-B/32 full FT, test", "unit": "norm", "d": 100 * k["diff_vs_ref"],
                    "ci": [100 * x for x in k["ci95"]],
                    "nulls": [100 * v["diff_vs_ref"] for t, v in r.items() if t.startswith("shuf") and t.endswith(":kappa|tuned")]})
    xf = load(R / "b16" / "freeze" / "transfer_comparison.json")
    if xf:
        r = xf["results"]
        k = r["b16xfer|kappa|tuned"]
        out.append({"label": "CLIP ViT-B/16 full FT, test\n(pre-registered transfer)", "unit": "norm",
                    "d": 100 * k["diff_vs_ref"], "ci": [100 * x for x in k["ci95"]],
                    "nulls": [100 * v["diff_vs_ref"] for t, v in r.items() if t.startswith("b16xfer|shuf") and "@kappa|tuned" in t]})
    lt = R / "aa" / "lora" / "test_freeze.jsonl"
    if lt.exists():
        recs = {json.loads(l)["label"]: json.loads(l) for l in lt.read_text().splitlines() if l.strip()}
        u = recs["uniform"]["avg_acc"]
        out.append({"label": "CLIP LoRA (KnOTS release), test\n(not replicated on 3 other banks, dev)", "unit": "acc",
                    "d": 100 * (recs["kappa_b128"]["avg_acc"] - u), "ci": None,
                    "nulls": [100 * (recs[f"shuf{s}:kappa_b128"]["avg_acc"] - u) for s in range(3) if f"shuf{s}:kappa_b128" in recs]})
    names = {"S1": "Llama-3.1-8B, 8 experts", "S2": "Llama-3.2-3B, 8 experts", "S3": "Llama-3.2-3B, 20 experts",
             "S4": "Llama-3.1-8B, 20 experts"}
    for s in ("S1", "S2", "S3", "S4"):
        o = load(R / "llm" / "summary" / s / "summary.json")
        if not o:
            continue
        for group, gl in (("frozen", "own-dev selection"), ("transfer", "S1 config transferred")):
            t = o.get(group, {})
            k = t.get("kappa|tuned", {})
            for sp in ("test", "fresh"):
                v = k.get(sp, {}).get("vs_uniform_tuned")
                if not v:
                    continue
                nulls = [100 * x[sp]["vs_uniform_tuned"]["diff"] for tag, x in t.items()
                         if tag.startswith("shuf") and tag.endswith(":kappa|tuned") and x.get(sp, {}).get("vs_uniform_tuned")]
                ur = t.get("uniform|tuned", {}).get(sp, {}).get("norm")
                if sp == "test" and group == "frozen" and ur:
                    nulls += [100 * (m - ur) for m in o.get("shuffle_rank", {}).get("test", {}).get("matched_shuffles", [])]
                out.append({"label": f"{names[s]}, {sp}\n({gl})", "unit": "norm", "d": 100 * v["diff"],
                            "ci": [100 * v["lo"], 100 * v["hi"]], "nulls": nulls})
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "legend.frameon": False})
    rs = rows()
    if not rs:
        print("no rows")
        return
    fig, ax = plt.subplots(figsize=(12.5, 0.78 * len(rs) + 2.2))
    for i, r in enumerate(rs):
        for x in r["nulls"]:
            ax.plot(x, i, "o", color=MUTED, ms=6.5, mec=SURF, mew=1.0, alpha=0.9, zorder=2)
        if r["ci"]:
            ax.plot(r["ci"], [i, i], color=BLUE, lw=2.2, zorder=3, solid_capstyle="round")
        ax.plot(r["d"], i, "o", color=BLUE, ms=10, mec=SURF, mew=1.6, zorder=4)
        ax.annotate(f"{r['d']:+.2f} {r['unit']} pts", (r["d"], i), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK2, fontsize=10.5)
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([r["label"] for r in rs])
    ax.set_ylim(len(rs) - 0.4, -0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("difference vs tuned uniform merging (each bank in its own units; 0 = no effect)")
    fig.suptitle("Literal condition-number weighting, lambda = softmax(-beta kappa(W)),\n"
                 "vs tuned uniform merging and shuffled-score nulls", x=0.01, ha="left", fontsize=13.5,
                 fontweight="semibold", color=INK)
    h1 = ax.plot([], [], "o", color=BLUE, ms=10)[0]
    h2 = ax.plot([], [], "o", color=MUTED, ms=7)[0]
    fig.legend([h1, h2], ["literal kappa rule (dev-selected), 95 % CI where available",
                          "shuffled-kappa nulls (same per-matrix concentration)"], loc="lower center", ncol=2, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out = R / "figures" / "fig_headline_kappa_vs_nulls.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print("written", out)


if __name__ == "__main__":
    main()
