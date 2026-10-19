"""Follow-up figure: does conditioning-based adaptation of the inverse solve, or uncertainty gating, beat its strongest
FAIR comparator? Only equal-budget comparisons are drawn (the large RegMean-vs-uniform gain is deliberately absent: it
is an established method's strength, not evidence for a conditioning method).

Panel A: kappa-capped ridge minus the equally tuned fixed relative ridge (and minus tuned RegMean shrinkage), frozen
test, every setting in its own units: full fine-tuning in normalised points (results/repro/clip_repro_regmean.json, n =
16 x 3 calibration subsets and n = 512), LoRA in absolute accuracy points with 95 % paired-bootstrap CIs
(results/inverse/lora_ridge_equal_test.json, fp32 evaluation).
Panel B: the uncertainty gate minus tuned ridge / kappa cap / its shuffled-uncertainty null, per calibration subset
(results/uncert/gate_test.json), or the reason it was not run (results/uncert/diagnostic_summary.json).
Writes results/figures/fig_followup_vs_fair_comparators.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def inverse_rows() -> list[dict]:
    out = []
    rr = load(R / "repro" / "clip_repro_regmean.json")
    if rr:
        tab = {}
        for c in rr["configs"]:
            tab.setdefault((c["n"], c["seed"]), {})[c["cell"]] = c["mean_norm"]
        for (n, sd), v in sorted(tab.items()):
            if {"kcap_ridge", "rel_ridge", "fixed_shrink"} <= set(v):
                out.append({"label": f"CLIP B/32 full FT, n = {n}/task" + (f", subset {sd}" if n == 16 else ""),
                            "unit": "norm", "fixed": (100 * (v["kcap_ridge"] - v["rel_ridge"]), None),
                            "shrink": (100 * (v["kcap_ridge"] - v["fixed_shrink"]), None)})
    le = load(R / "inverse" / "lora_ridge_equal_test.json")
    if le:
        for lab, rep in le.items():
            f, s = rep["kcap_minus_fixed"], rep["kcap_minus_shrink"]
            out.append({"label": f"CLIP LoRA {lab}", "unit": "acc",
                        "fixed": (f["diff_pt"], (f["lo_pt"], f["hi_pt"])), "shrink": (s["diff_pt"], (s["lo_pt"], s["hi_pt"]))})
    return out


def gate_status() -> tuple[dict | None, str]:
    g = load(R / "uncert" / "gate_test.json")
    d = load(R / "uncert" / "diagnostic_summary.json")
    if g and "status" not in g:
        return g, ""
    if g and "status" in g:
        return None, g["status"]
    if d is not None:
        return None, ("diagnostic PASSED; gate stage pending" if d.get("diagnostic_passed")
                      else "not run: the diagnostic failed its pre-registered criterion")
    return None, "pending (diagnostic not finished)"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "legend.frameon": False})
    rows = inverse_rows()
    gate, why = gate_status()
    if not rows and gate is None:
        print("no rows")
        return
    n_g = 0 if gate is None else sum(3 for _ in gate)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(0.62 * max(len(rows), n_g, 4) + 2.6, 5.0)),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for i, r in enumerate(rows):
        for key, col, dy, mk in (("fixed", BLUE, -0.13, "o"), ("shrink", MUTED, 0.13, "s")):
            v, ci = r[key]
            if ci:
                ax.plot(ci, [i + dy, i + dy], color=col, lw=2.2, zorder=2, solid_capstyle="round")
            ax.plot(v, i + dy, mk, color=col, ms=8.5, mec=SURF, mew=1.4, zorder=3)
        ax.annotate(f"{r['fixed'][0]:+.2f} {r['unit']} pts", (r["fixed"][0], i - 0.13), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", color=INK2, fontsize=10)
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.7)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("difference on frozen test (each row in its own units; 0 = no gain)")
    ax.set_title("A. Inverse-solve conditioning: kappa-cap ridge minus equally tuned comparators", loc="left",
                 fontsize=12.5, fontweight="semibold", color=INK, pad=14)
    h1 = ax.plot([], [], "o", color=BLUE, ms=8.5)[0]
    h2 = ax.plot([], [], "s", color=MUTED, ms=8.5)[0]
    ax.legend([h1, h2], ["minus tuned fixed relative ridge (strongest fair comparator)", "minus tuned RegMean shrinkage"],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=10.5)
    ax = axes[1]
    if gate is None:
        ax.axis("off")
        ax.text(0.02, 0.6, "B. Uncertainty-gated correction", fontsize=12.5, fontweight="semibold", color=INK,
                transform=ax.transAxes)
        ax.text(0.02, 0.48, why, fontsize=12, color=INK2, transform=ax.transAxes, wrap=True)
    else:
        labels, i = [], 0
        for cfg, rep in gate.items():
            for key, lab, col in (("gate_minus_ridge_per_subset", "gate - tuned ridge", GREEN),
                                  ("gate_minus_kcap_per_subset", "gate - kappa cap", BLUE),
                                  ("gate_minus_shuffled_gate_per_subset", "gate - shuffled gate", ORANGE)):
                d = 100 * np.asarray(rep.get(key, []), float)
                if d.size:
                    ax.plot(d, np.full(d.size, i), "o", color=col, ms=6.5, mec=SURF, mew=1.0, alpha=0.85, zorder=2)
                    ax.plot(d.mean(), i, "D", color=INK, ms=7, zorder=3)
                labels.append(f"{cfg}: {lab}")
                i += 1
        ax.axvline(0, color=INK2, lw=1.2, zorder=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylim(len(labels) - 0.5, -0.7)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("frozen test, normalised points (dots = calibration subsets, diamond = mean)")
        ax.set_title("B. Uncertainty-gated correction vs its fair comparators", loc="left", fontsize=12.5,
                     fontweight="semibold", color=INK)
    fig.tight_layout()
    out = R / "figures" / "fig_followup_vs_fair_comparators.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print("written", out)


if __name__ == "__main__":
    main()
