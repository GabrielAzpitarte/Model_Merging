"""Summarise LoRA RegMean runs (results/aa/lora/regmean_<bank><suffix>_dev.jsonl) into a markdown table.

Per (budget, solve, regulariser): best dev accuracy over the alpha grid, the alpha at which it is reached, and the
accuracy at alpha = 1.3 (a stability indicator: unregularised solves blow up when the update is scaled).
Also per-map Gram conditioning (full 768-dim vs adapter subspace) from regmean_<bank><suffix>_maps.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    d = RESULTS / "aa" / "lora"
    lines = []
    for fp in sorted(d.glob("regmean_*_dev.jsonl")):
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        tag = fp.stem.replace("regmean_", "").replace("_dev", "")
        uni = [r for r in rows if r["method"] == "uniform"]
        ub = max(uni, key=lambda r: r["avg_acc"]) if uni else None
        lines.append(f"## {tag}\n")
        if ub:
            lines.append(f"uniform best {100 * ub['avg_acc']:.1f} (alpha {ub['alpha']}); uniform alpha 1.3: "
                         f"{100 * next(r['avg_acc'] for r in uni if r['alpha'] == 1.3):.1f}\n")
        lines.append("| budget | solve | regulariser | best dev acc (alpha) | acc at alpha 1.3 | median kappa(G) | median kappa(solve) |")
        lines.append("|---|---|---|---|---|---|---|")
        keyf = lambda r: (r["budget"], r["solve"], r["shrink"], r["kappa_target"], r.get("ridge_to"))  # noqa: E731
        groups = {}
        for r in rows:
            if r["method"] == "regmean":
                groups.setdefault(keyf(r), []).append(r)
        for k in sorted(groups, key=lambda k: (k[0], k[1], -k[2], k[3] or 0, k[4] or "")):
            rr = groups[k]
            best = max(rr, key=lambda r: r["avg_acc"])
            a13 = next((r["avg_acc"] for r in rr if r["alpha"] == 1.3), float("nan"))
            reg = f"shrink {k[2]}" if k[3] is None else f"kappa<= {k[3]:.0e} ridge->{k[4]}"
            lines.append(f"| {k[0]} | {k[1]} | {reg} | {100 * best['avg_acc']:.1f} ({best['alpha']}) | {100 * a13:.1f} | "
                         f"{best.get('median_kappa_G', float('nan')):.2e} | {best.get('median_kappa_solve', float('nan')):.2e} |")
        mp = d / f"regmean_{tag}_maps.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            lines.append("")
            for key, maps in m.items():
                ks = np.array([v["kappa_G"] for v in maps.values()])
                lines.append(f"- per-map kappa(G) {key}: median {np.median(ks):.2e}, min {ks.min():.2e}, max {ks.max():.2e}")
        lines.append("")
    (d / "regmean_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
