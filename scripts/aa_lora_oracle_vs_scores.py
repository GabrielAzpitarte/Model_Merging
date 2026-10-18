"""LoRA bank: does the oracle allocation (budget D) follow kappa or any update-spectrum statistic?

Same decomposition as aa_score_vs_targets.py, on the 48 adapted attention maps:
within-map Spearman (perm null), between-expert (8 pts), within-expert (perm null: shuffle maps per expert).
Scores from results/aa/lora/lora_stats.csv. Writes results/aa/lora/oracle_vs_scores.{json,md}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa_score_vs_targets import rk, rowcorr, within_exp  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    rows = list(csv.DictReader((RESULTS / "aa" / "lora" / "lora_stats.csv").open()))
    names = sorted({r["param"] for r in rows})
    tab = {(r["param"], r["expert"]): r for r in rows}
    col = lambda c: np.array([[float(tab[(n, t)][c]) for t in TASKS8] for n in names])  # noqa: E731
    scores = {"kappa_W": col("kappa2_W"), "smin_W": col("smin_W"), "fro_D": col("fro_D"),
              "kplus_D_0.01": col("kplus_D_0.01"), "stable_rank_D": col("stable_rank_D"),
              "entropy_rank_D": col("entropy_rank_D"), "kappa_A(gauge-dep)": col("kappa_A")}
    od = RESULTS / "aa" / "oracle_lambda_lora"
    out, lines = {}, ["| oracle | score | within-map rho (p) | between (8 pts) | within-expert (p) |", "|---|---|---|---|---|"]
    for fp in sorted(od.glob("layerwise*_seed*.json")):
        rec = json.loads(fp.read_text())
        T = np.array([rec["lambda"][n] for n in names])
        rt = rk(T)
        neff = float(np.mean(np.exp(-np.sum(T * np.log(np.maximum(T, 1e-12)), 1))))
        out[fp.stem] = {"alpha": rec["alpha"], "dev": rec["dev"]["avg_acc"], "test": rec["test"]["avg_acc"], "n_eff": neff,
                        "expert_mean_lambda": dict(zip(TASKS8, T.mean(0).round(3).tolist()))}
        for sn, S in scores.items():
            rs = rk(S)
            wm = float(rowcorr(rt, rs).mean())
            null = np.array([rowcorr(rt, np.take_along_axis(rs, np.argsort(rng.random(rs.shape), 1), 1)).mean() for _ in range(4000)])
            we = within_exp(rt, rs)
            nullw = np.array([within_exp(rt, rk(np.stack([S[rng.permutation(S.shape[0]), i] for i in range(8)], 1))) for _ in range(2000)])
            v = {"within_map": wm, "p": float(np.mean(np.abs(null) >= abs(wm))), "between": float(np.corrcoef(rt.mean(0), rs.mean(0))[0, 1]),
                 "within_exp": we, "p_within": float(np.mean(np.abs(nullw - nullw.mean()) >= abs(we - nullw.mean())))}
            out[fp.stem][sn] = v
            lines.append(f"| {fp.stem} | {sn} | {wm:+.3f} ({v['p']:.3f}) | {v['between']:+.2f} | {we:+.3f} ({v['p_within']:.3f}) |")
    (RESULTS / "aa" / "lora" / "oracle_vs_scores.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "aa" / "lora" / "oracle_vs_scores.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("alpha", "dev", "test", "n_eff", "expert_mean_lambda")} for k, v in out.items()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
