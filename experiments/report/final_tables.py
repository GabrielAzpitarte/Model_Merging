"""Concise results tables (CLIP + LLM + audits) generated straight from the records -> docs/REPORT_03_tables.md.

Missing pieces are printed as "pending". Units are explicit in every table:
  norm  = mean normalised accuracy (accuracy / own-expert accuracy, averaged over tasks), in %;
  acc   = mean absolute accuracy, in %;  joint = LLM accuracy with all tasks' answers competing (no task ID).
Usage: final_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, nd=2, sign=False):
    if x is None:
        return "pending"
    return f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"


def ci(d):
    if not d:
        return "pending"
    return f"{100 * d['diff']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


def clip_section() -> list[str]:
    L = ["## CLIP (ViT-B/32 and B/16 full fine-tunes; LoRA banks) - frozen test", ""]
    tc = load(R / "freeze" / "test_comparison.json")
    if tc:
        res = tc["results"]
        L += ["| B/32 test (norm %) | value | diff vs tuned uniform [95 % CI] |", "|---|---|---|"]
        for tag, lab in (("uniform|tuned", "uniform, tuned alpha"), ("kappa|tuned", "literal kappa, tuned"),
                         ("kappa_rev|tuned", "reversed kappa, tuned")):
            v = res.get(tag)
            if v:
                c = v.get("ci95") or [None, None]
                L.append(f"| {lab} | {pct(v['test_mean_norm'])} | {pct(v.get('diff_vs_ref'), sign=True)}"
                         + (f" [{pct(c[0], sign=True)}, {pct(c[1], sign=True)}] |" if c[0] is not None else " |"))
        sh = [res[t]["diff_vs_ref"] for t in res if t.startswith("shuf") and t.endswith(":kappa|tuned")]
        if sh:
            L.append(f"| 5 shuffled-kappa nulls, tuned | | {pct(min(sh), sign=True)} ... {pct(max(sh), sign=True)} |")
    xf = load(R / "b16" / "freeze" / "transfer_comparison.json")
    if xf:
        v = xf["results"].get("b16xfer|kappa|tuned")
        if v:
            L.append(f"| B/16 pre-registered transfer: literal kappa | {pct(v['test_mean_norm'])} | "
                     f"{pct(v['diff_vs_ref'], sign=True)} |")
    rp = load(R / "repro" / "clip_repro_rules.json")
    if rp:
        agree = [min(c["per_image_agreement_with_original"].values()) for c in rp["configs"].values()
                 if c.get("per_image_agreement_with_original")]
        L += ["", f"Reproduction audit: {len(rp['configs'])} frozen B/32 test configurations rerun; max |diff vs record| "
                  f"{max(abs(c['diff_vs_record_pt']) for c in rp['configs'].values()):.3f} pt; minimum per-task per-image "
                  f"agreement {min(agree):.4f}; non-uniform rules' coefficients differ from uniform by up to "
                  f"{max(c['max_abs_lambda_minus_uniform'] for c in rp['configs'].values()):.3f}."]
    return L


def inverse_section() -> list[str]:
    L = ["", "## Inverse-solve conditioning (RegMean-type merges)", ""]
    ca = load(R / "inverse" / "collapse_audit.json")
    if ca:
        L += ["**Collapse audit (LoRA, budget C, full solve; acc %):**", "",
              "| bank / calibration | config | fp16 eval | fp32 eval | max abs W (base max) | verdict |", "|---|---|---|---|---|---|"]
        for key, rep in ca.items():
            for c, v in rep["configs"].items():
                if c.startswith("plain") or c.startswith("kcap1e4@alpha1.0"):
                    L.append(f"| {rep['label']} | {c} | {pct(v['acc_fp16'])} | {pct(v['acc_fp32'])} | "
                             f"{v['max_abs_weight']:.3g} ({v['max_abs_base_weight']:.3g}) | {v['verdict']} |")
    cf = load(R / "inverse" / "collapse_audit_fullft.json")
    if cf:
        L += ["", "**Full FT, budget C (norm %, dev):**", "", "| calibration / config | fp16 | fp32 | max abs W |", "|---|---|---|---|"]
        for key, v in cf.items():
            L.append(f"| {key} | {pct(v['fp16']['mean_norm'])} | {pct(v['fp32']['mean_norm'])} | {v['max_abs_weight']:.3g} |")
    le = load(R / "inverse" / "lora_ridge_equal_test.json")
    if le:
        L += ["", "**Equal-budget LoRA comparison (fp32 evaluation, acc %, test; dev-selected strength and alpha):**", "",
              "| setting | uniform | plain-or-best shrink | tuned fixed ridge | kappa-cap ridge | kcap - fixed [95 % CI] | "
              "kcap - shrink [CI] | rank-16: kcap - fixed [CI] |", "|---|---|---|---|---|---|---|---|"]
        for lab, rep in le.items():
            t = rep["test"]
            f = lambda d: f"{d['diff_pt']:+.2f} [{d['lo_pt']:+.2f}, {d['hi_pt']:+.2f}]"
            L.append(f"| {lab} | {pct(t['uniform']['acc'])} | {pct(t['shrink']['acc'])} (a={t['shrink']['strength']}) | "
                     f"{pct(t['fixed']['acc'])} | {pct(t['kcap']['acc'])} | {f(rep['kcap_minus_fixed'])} | "
                     f"{f(rep['kcap_minus_shrink'])} | {f(rep['kcap_minus_fixed_r16'])} |")
    return L


def uncert_section() -> list[str]:
    L = ["", "## Uncertainty diagnostic (CLIP full FT)", ""]
    d = load(R / "uncert" / "diagnostic_summary.json")
    if not d:
        return L + ["pending"]
    L += ["| budget:n | Spearman(u_boot, r_fun), median over subsets | best competitor (kappa(G), 1/lambda_min, "
          "tr G^-1, ||F||) | within-map | passes |", "|---|---|---|---|---|"]
    for k, v in d["configs"].items():
        L.append(f"| {k} | {v['primary']:.3f} | {v['best_competitor']:.3f} | "
                 f"{v.get('within_map_spearman_u_rel_vs_r_fun_median')} | {v['passes']} |")
    L.append(f"\nPre-registered criterion: {d['criterion']}. **Diagnostic passed: {d['diagnostic_passed']}**")
    g = load(R / "uncert" / "gate_test.json")
    if g:
        L += ["", "Gate stage (test norm %, mean over subsets):", "```", json.dumps(g, indent=1)[:3000], "```"]
    return L


def llm_section() -> list[str]:
    L = ["", "## LLM banks (exact spectra, exact merges)", ""]
    for s in ("S1", "S2", "S3", "S4"):
        o = load(R / "llm" / "summary" / s / "summary.json")
        if not o:
            L.append(f"* {s}: pending")
            continue
        L += [f"### {s}: {o['bank']} T{o['T']}", ""]
        for group in ("frozen", "transfer"):
            t = o.get(group)
            if not t:
                continue
            L += [f"{group} (norm %, diff vs tuned uniform with paired 95 % CI; joint accuracy %)", "",
                  "| config | test norm | test diff | fresh norm | fresh diff | test joint |", "|---|---|---|---|---|---|"]
            for tag in ("uniform|tuned", "uniform|alpha1", "kappa|tuned", "kappa|alpha1", "kappa_rev|tuned",
                        "global:kappa|tuned", "logkappa|tuned", "kappa_res|tuned"):
                v = t.get(tag)
                if not v:
                    continue
                te, fr = v.get("test", {}), v.get("fresh", {})
                L.append(f"| {tag} | {pct(te.get('norm'))} | {ci(te.get('vs_uniform_tuned'))} | {pct(fr.get('norm'))} | "
                         f"{ci(fr.get('vs_uniform_tuned'))} | {pct(te.get('joint'))} |")
            sh = [v["test"]["vs_uniform_tuned"]["diff"] for k, v in t.items()
                  if k.startswith("shuf") and k.endswith(":kappa|tuned") and v.get("test", {}).get("vs_uniform_tuned")]
            if sh:
                L.append(f"| tuned kappa shuffles ({len(sh)}) | | {100 * min(sh):+.2f} ... {100 * max(sh):+.2f} | | | |")
            L.append("")
        if o.get("shuffle_rank"):
            for sp, v in o["shuffle_rank"].items():
                if v["n_null"]:
                    L.append(f"* kappa vs shuffles ({sp}): rank {v['rank_among_all']} of {v['n_null'] + 1}")
        if o.get("noise_floor"):
            L.append("* numerical-noise range: " + "; ".join(f"{k} {100 * v['norm_range']:.2f} pt" for k, v in o["noise_floor"].items()))
        if o.get("context", {}).get("isoc"):
            L.append("* Iso-C (weight-only context): " + "; ".join(f"{sp} {pct(v['norm'])}" for sp, v in o["context"]["isoc"].items()))
        if o.get("tail"):
            L.append("* tail edit (norm %): " + "; ".join(
                f"{k}: " + ", ".join(f"{c} {pct(v['norm'])}" for c, v in cells.items()) for k, cells in o["tail"].items()))
        if o.get("gradpref"):
            L.append("* allocation usefulness (median within-matrix Spearman with -dL/dlambda): " + "; ".join(
                f"{tag}: " + ", ".join(f"{k} {v['median_rho']:+.3f}" for k, v in g.items()) for tag, g in o["gradpref"].items()))
        L.append("")
    return L


def main() -> None:
    L = ["# Results tables (generated by experiments/report/final_tables.py from the raw records)", "",
         "Units: norm = mean normalised accuracy %, acc = mean absolute accuracy %, joint = LLM accuracy without task "
         "IDs (all tasks' answers compete). Never compare norm and acc directly.", ""]
    L += clip_section() + inverse_section() + uncert_section() + llm_section()
    out = ROOT / "docs" / "REPORT_03_tables.md"
    out.write_text("\n".join(L) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
