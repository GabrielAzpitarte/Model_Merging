"""Summarise the shared-target soup experiments into results/aa/soup/summary.md (+ .json).

For each file and task: uniform, oracles, the literal kappa rule at b in {0.25, 1, 4, 16, inf}, and the best b
of each other rule (best-of-grid is an upper bound for rules with a free beta; tuning-free rules are reported
as is). Mean over tasks and the number of tasks where each rule beats uniform are given at the bottom.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS  # noqa: E402

import numpy as np  # noqa: E402

FILES = ("partA_controlled_noise", "partA_controlled_noise_strong", "partA_controlled_noise_strong_normmatched",
         "partA_controlled_noise_strong_v2", "partB_lora_variants", "partB_lora_variants_v2")
FIXED_B = (0.25, 1, 4, 16, math.inf)


def isinf(b) -> bool:
    return isinstance(b, float) and math.isinf(b)


def main() -> None:
    d = RESULTS / "aa" / "soup"
    md, js = [], {}
    for fn in FILES:
        fp = d / f"{fn}.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        tasks = list(dict.fromkeys(r["task"] for r in rows))
        cols = ["uniform"] + [f"kappa b={b}" for b in FIXED_B] + ["best kappa", "best logkappa", "best kappa_rev",
                "best global:logkappa", "best dsrank", "best dsrank_rev", "best dnorm", "best dnorm_rev", "mp_invvar"]
        orc = sorted({r["rule"] for r in rows if r["rule"].startswith(("oracle", "cleanest", "noisiest"))})
        cols += orc
        table = {}
        for t in tasks:
            rr = [r for r in rows if r["task"] == t]
            row = {}
            for r in rr:
                if r["rule"] == "uniform":
                    row["uniform"] = r["acc"]
                if r["rule"] == "kappa":
                    for b in FIXED_B:
                        if r["b"] == b or (isinf(b) and isinf(r["b"])):
                            row[f"kappa b={b}"] = r["acc"]
                if r["rule"] in orc or r["rule"] == "mp_invvar":
                    row[r["rule"]] = r["acc"]
            for rule in ("kappa", "logkappa", "kappa_rev", "global:logkappa", "dsrank", "dsrank_rev", "dnorm", "dnorm_rev"):
                accs = [r["acc"] for r in rr if r["rule"] == rule]
                if accs:
                    row[f"best {rule}"] = max(accs)
            table[t] = row
        js[fn] = table
        md.append(f"## {fn} ({len(tasks)} tasks)\n")
        md.append("| task | " + " | ".join(cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for t, row in table.items():
            md.append(f"| {t} | " + " | ".join(f"{100 * row[c]:.1f}" if c in row else "–" for c in cols) + " |")
        means = {c: np.mean([row[c] for row in table.values() if c in row]) for c in cols if any(c in row for row in table.values())}
        wins = {c: sum(1 for row in table.values() if c in row and row[c] > row["uniform"] + 1e-12) for c in means}
        md.append("| **mean** | " + " | ".join(f"**{100 * means[c]:.1f}**" if c in means else "–" for c in cols) + " |")
        md.append("| beats uniform | " + " | ".join(f"{wins[c]}/{len(tasks)}" if c in wins else "–" for c in cols) + " |\n")
    (d / "summary.md").write_text("\n".join(md) + "\n")
    (d / "summary.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
