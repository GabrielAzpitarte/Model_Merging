"""Analysis of the LLM study: tables, paired bootstraps, shuffle ranks, tail-edit table, figures.

Reads results/llm/<bank>_T<T>/*/*.jsonl (+ saved predictions) and configs/llm/*.json; writes
results/llm/summary/<setting>/{dev_grid.csv, dev_best.json, frozen.json, summary.md, ...} and
results/llm/summary/figures/*.png. Works on partial results (missing pieces are reported as missing).
Normalised accuracy is always recomputed here from per-task local accuracy and the experts' own local accuracy
on the same split (protocol 02 sec. 2). Units: normalised points (x 100) unless stated.
Usage: analyze.py [--settings S1,S2,S3,S4] [--boot 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan  # noqa: E402
from condmerge.llm import banks as BK  # noqa: E402
from condmerge.llm import jobs as J  # noqa: E402

SUM = J.RESULTS / "summary"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ----------------------------------------------------------------------------------------------- loading
def records(setting: str) -> dict[str, list[dict]]:
    s = plan.SETTINGS[setting]
    out: dict[str, list[dict]] = {}
    for r in J.read_results(s["bank"], s["T"]):
        if r["cfg"].get("setting") != setting:
            continue
        out.setdefault(r["cfg"]["group"], []).append(r)
    return out


def refs(recs: dict, split: str) -> np.ndarray | None:
    for r in recs.get("experts", []):
        if r["cfg"]["split"] == split:
            return np.asarray(r["own_local"], np.float64)
    return None


def norm_of(r: dict, ref: np.ndarray | None) -> float:
    if ref is None:
        return float("nan")
    loc = np.asarray(r["metrics"]["local_per_task"], np.float64)
    ok = np.isfinite(loc) & np.isfinite(ref) & (ref > 0)
    return float(np.mean(loc[ok] / ref[ok])) if ok.any() else float("nan")


def per_task_norm(r: dict, ref: np.ndarray) -> np.ndarray:
    loc = np.asarray(r["metrics"]["local_per_task"], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return loc / ref


def preds(r: dict) -> dict | None:
    p = r.get("preds")
    if not p or not (ROOT / p).exists():
        return None
    z = np.load(ROOT / p)
    return {k: z[k] for k in z.files}


def boot_diff(px: dict, py: dict, ref: np.ndarray, B: int = 2000, seed: int = 0, what: str = "local") -> dict:
    """Paired, task-stratified bootstrap of (X - Y) in mean normalised accuracy (what='local') or mean joint
    accuracy (what='joint'), over the tasks present in both."""
    assert np.array_equal(px["index"], py["index"]) and np.array_equal(px["task_of"], py["task_of"])
    rng = np.random.default_rng(seed)
    cx, cy = px[f"correct_{what}"].astype(np.float64), py[f"correct_{what}"].astype(np.float64)
    tasks = [k for k in np.unique(px["task_of"]) if np.isfinite(ref[k]) and ref[k] > 0]
    w = np.array([1.0 / ref[k] if what == "local" else 1.0 for k in tasks])
    point = np.mean([w[i] * (cx[px["task_of"] == k].mean() - cy[px["task_of"] == k].mean()) for i, k in enumerate(tasks)])
    bs = np.zeros(B)
    per_task = {}
    for i, k in enumerate(tasks):
        sel = np.flatnonzero(px["task_of"] == k)
        d = cx[sel] - cy[sel]
        idx = rng.integers(0, len(sel), size=(B, len(sel)))
        bs += w[i] * d[idx].mean(1)
        per_task[int(k)] = float(w[i] * d.mean())
    bs /= len(tasks)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "n_tasks": len(tasks), "per_task": per_task,
            "frac_boot_positive": float(np.mean(bs > 0))}


def _b(b):
    return math.inf if b in ("inf", math.inf) else float(b)


# ----------------------------------------------------------------------------------------------- per setting
def analyze_setting(setting: str, B: int) -> dict:
    s = plan.SETTINGS[setting]
    recs = records(setting)
    out = {"setting": setting, "bank": s["bank"], "T": s["T"], "tasks": BK.tasks(s["bank"], s["T"]),
           "groups": {g: len(v) for g, v in recs.items()}}
    d = SUM / setting
    d.mkdir(parents=True, exist_ok=True)
    rv, rt, rf = refs(recs, "val"), refs(recs, "test"), refs(recs, "fresh")
    out["expert_ref"] = {"val": None if rv is None else rv.tolist(), "test": None if rt is None else rt.tolist(),
                         "fresh": None if rf is None else rf.tolist()}
    for r in recs.get("base", []):
        out.setdefault("base", {})[r["cfg"]["split"]] = {"norm": norm_of(r, refs(recs, r["cfg"]["split"])),
                                                          "joint": r["metrics"]["joint_macro"]}
    # dev grid
    dev = []
    for r in recs.get("dev", []):
        dev.append({"rule": r["cfg"]["rule"], "b": r["cfg"]["b"], "alpha": r["cfg"]["alpha"], "norm": norm_of(r, rv),
                    "joint": r["metrics"]["joint_macro"], "local": r["metrics"]["local_macro"],
                    "n_eff": r["rule_info"].get("n_eff_mean"), "dist_uniform": r["rule_info"].get("dist_from_uniform"),
                    "update_norm_total": r["check"]["update_norm_total"], "fingerprint": r["check"]["fingerprint"],
                    "warning": r["rule_info"].get("warning", "")})
    if dev:
        with (d / "dev_grid.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dev[0]))
            w.writeheader()
            w.writerows(sorted(dev, key=lambda x: (x["rule"], _b(x["b"]), x["alpha"])))
        best = {}
        for rule in sorted({x["rule"] for x in dev}):
            rows = [x for x in dev if x["rule"] == rule]
            bb = max(rows, key=lambda x: x["norm"])
            curve = {}
            for x in rows:
                k = str(x["b"])
                if k not in curve or x["norm"] > curve[k]["norm"]:
                    curve[k] = {"b": x["b"], "alpha": x["alpha"], "norm": x["norm"], "joint": x["joint"], "n_eff": x["n_eff"]}
            a1 = [x for x in rows if abs(x["alpha"] - 1.0) < 1e-9]
            best[rule] = {"best": bb, "best_alpha1": max(a1, key=lambda x: x["norm"]) if a1 else None,
                          "curve": sorted(curve.values(), key=lambda c: _b(c["b"])), "n": len(rows)}
        out["dev_best"] = best
        fp = {}
        for x in dev:  # identical fingerprints for different rules would mean the dispatcher ignored the rule
            fp.setdefault(round(x["fingerprint"], 9), set()).add((x["rule"], str(x["b"]), x["alpha"]))

        def canon(rule, b):  # at the argmin limit kappa and log kappa (same shuffle seed) select identically
            return (rule.replace("logkappa", "kappa"), b) if b == "inf" else (rule, b)
        out["fingerprint_collisions"] = [sorted(v) for v in fp.values()
                                         if len({canon(a, b2) for a, b2, _ in v}) > 1
                                         and not all(r_ == "uniform" or b2 in ("0", "0.0") for r_, b2, _ in v)]
        out["dispatcher_warnings"] = [x for x in dev if x["warning"]]
    # frozen + transfer evaluations
    for group in ("frozen", "transfer"):
        rows = recs.get(group, [])
        if not rows:
            continue
        table = {}
        for r in rows:
            sp = r["cfg"]["split"]
            ref = refs(recs, sp)
            tag = r["cfg"].get("tag", r["cfg"]["rule"])
            pt = per_task_norm(r, ref) if ref is not None else None
            table.setdefault(tag, {})[sp] = {
                "rule": r["cfg"]["rule"], "b": r["cfg"]["b"], "alpha": r["cfg"]["alpha"], "norm": norm_of(r, ref),
                "joint": r["metrics"]["joint_macro"], "local": r["metrics"]["local_macro"],
                "worst_task_norm": float(np.nanmin(pt)) if pt is not None else None,
                "per_task_norm": None if pt is None else [None if not np.isfinite(v) else float(v) for v in pt],
                "n_eff": r["rule_info"].get("n_eff_mean"), "key": r["key"]}
        ref_tag = "uniform|tuned"
        for sp in ("test", "fresh"):
            ref = refs(recs, sp)
            base = next((r for r in rows if r["cfg"].get("tag") == ref_tag and r["cfg"]["split"] == sp), None)
            pb = preds(base) if base else None
            if pb is None or ref is None:
                continue
            for r in rows:
                if r["cfg"]["split"] != sp or r is base:
                    continue
                pr = preds(r)
                if pr is None:
                    continue
                tag = r["cfg"].get("tag", r["cfg"]["rule"])
                table[tag][sp]["vs_uniform_tuned"] = boot_diff(pr, pb, ref, B)
                table[tag][sp]["vs_uniform_tuned_joint"] = boot_diff(pr, pb, ref, B, what="joint")
        out[group] = table
        (d / f"{group}.json").write_text(json.dumps(table, indent=1, default=float))
    # shuffle ranks on test (tuned shuffles + matched-config shuffles)
    fz = out.get("frozen", {})
    if fz:
        for sp in ("test", "fresh"):
            k = fz.get("kappa|tuned", {}).get(sp)
            if not k:
                continue
            tuned = [v[sp]["norm"] for t, v in fz.items() if t.startswith("shuf") and t.endswith(":kappa|tuned") and sp in v]
            matched = [norm_of(r, refs(recs, sp)) for r in recs.get("matched", []) if r["cfg"]["split"] == sp]
            allnull = tuned + matched
            out.setdefault("shuffle_rank", {})[sp] = {
                "kappa": k["norm"], "tuned_shuffles": tuned, "matched_shuffles": matched,
                "rank_among_all": int(1 + sum(x > k["norm"] for x in allnull)), "n_null": len(allnull),
                "null_min": min(allnull) if allnull else None, "null_max": max(allnull) if allnull else None,
                "null_mean": float(np.mean(allnull)) if allnull else None}
    # numerical noise floor
    nz = {}
    for r in recs.get("noise", []):
        sp = r["cfg"]["split"]
        nz.setdefault(f"{sp}@{r['cfg']['alpha']}", []).append({"norm": norm_of(r, refs(recs, sp)),
                                                               "joint": r["metrics"]["joint_macro"],
                                                               "jitter": r["cfg"].get("jitter_rel", 0),
                                                               "repeat": r["cfg"].get("repeat", 0)})
    if nz:
        out["noise_floor"] = {k: {"norm_range": float(max(x["norm"] for x in v) - min(x["norm"] for x in v)),
                                  "joint_range": float(max(x["joint"] for x in v) - min(x["joint"] for x in v)),
                                  "runs": v} for k, v in nz.items()}
    # context baselines
    for r in recs.get("context", []):
        sp = r["cfg"]["split"]
        out.setdefault("context", {}).setdefault(r["cfg"]["method"], {})[sp] = {
            "alpha": r["cfg"]["alpha"], "norm": norm_of(r, refs(recs, sp)), "joint": r["metrics"]["joint_macro"]}
    # tail edit
    tail = {}
    for r in recs.get("tail", []):
        c = r["cfg"]
        sp = c["split"]
        tail.setdefault(f"{c['config_tag']}|{sp}", {})[c["cell"]] = {
            "norm": norm_of(r, refs(recs, sp)), "joint": r["metrics"]["joint_macro"],
            "lambda_edited_mean": float(np.mean(r["lambda_edited_expert"])),
            "n_eff": r["rule_info"].get("n_eff_mean"), "edited_expert": r.get("edited_expert")}
    if tail:
        out["tail"] = tail
    for r in recs.get("tail_expert", []):
        out.setdefault("tail_expert", {})[r["cfg"]["split"]] = {k: r[k] for k in r if k in (
            "own_local", "kl_mean", "kl_max", "max_abs_dlogp", "rel_dhidden_mean", "rel_dhidden_max",
            "argmax_first_token_changed", "edited_expert")}
    # gradient preference vs scores (within-matrix Spearman)
    gp = {}
    for r in recs.get("gradpref", []):
        g = np.load(ROOT / r["grad_file"])
        gp[r["cfg"]["tag"]] = gradpref_stats(s["bank"], s["T"], g)
    if gp:
        out["gradpref"] = gp
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def gradpref_stats(bank: str, T: int, g: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    from condmerge.llm import rules

    pref = -g  # positive: increasing lambda_i[p] lowers the val loss
    out = {}
    tabs = {"kappa": -rules.score_table(bank, T, "kappa"), "logkappa": -rules.score_table(bank, T, "logkappa"),
            "kappa_res": -rules.score_table(bank, T, "kappa_res"),
            "dnorm": -np.stack([rules.spectra(bank)[m]["dnorm_f"][[rules.spectra(bank)[m]["versions"].index(t)
                                                                   for t in BK.tasks(bank, T)]]
                                for m in BK.module_names(bank)])}
    kinds = np.asarray([BK.kind_of(m) for m in BK.module_names(bank)])
    for name, t in tabs.items():
        rhos = []
        for j in range(len(t)):
            if np.std(t[j]) > 0 and np.std(pref[j]) > 0:
                rhos.append(spearmanr(t[j], pref[j]).statistic)
            else:
                rhos.append(np.nan)
        rhos = np.asarray(rhos)
        out[name] = {"median_rho": float(np.nanmedian(rhos)), "mean_rho": float(np.nanmean(rhos)),
                     "frac_positive": float(np.nanmean(rhos > 0)),
                     "by_kind": {k: float(np.nanmean(rhos[kinds == k])) for k in BK.PROJ}}
    return out


# ----------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
                         "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "semibold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "font.family": "sans-serif", "lines.linewidth": 2, "lines.solid_capstyle": "round"})
    return plt


BANK_LABEL = {"l31_8b": "Llama-3.1-8B-Instruct", "l32_3b": "Llama-3.2-3B-Instruct"}


def _title(sname: str, o: dict) -> str:
    return f"{sname}: {BANK_LABEL.get(o['bank'], o['bank'])}, {o['T']} experts"


def fig_test_diffs(all_out: dict, path: Path) -> None:
    """Original rule vs uniform and shuffles: test (circle) / fresh (square) difference vs tuned uniform, 95 % CIs."""
    plt = _style()
    panels = [(s, o) for s, o in all_out.items() if o.get("frozen")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.0), squeeze=False)
    order = [("kappa|tuned", "kappa (literal), tuned", CAT[0]), ("kappa|alpha1", "kappa (literal), alpha = 1", CAT[0]),
             ("kappa_rev|tuned", "reversed kappa", CAT[1]), ("global:kappa|tuned", "global kappa (geo-mean)", CAT[6]),
             ("logkappa|tuned", "R1: log kappa", CAT[2]), ("kappa_res|tuned", "R2: kappa_res (BF16 floor)", CAT[3]),
             ("uniform|alpha1", "uniform, alpha = 1", MUTED)]
    for ax, (sname, o) in zip(axes[0], panels):
        fz = o["frozen"]
        ys, labels, y = [], [], 0
        for tag, lab, col in order:
            if tag not in fz:
                continue
            for sp, mk, dy in (("test", "o", -0.14), ("fresh", "s", 0.14)):
                v = fz[tag].get(sp, {}).get("vs_uniform_tuned")
                if v is None:
                    continue
                ax.errorbar(100 * v["diff"], y + dy, xerr=[[100 * (v["diff"] - v["lo"])], [100 * (v["hi"] - v["diff"])]],
                            fmt=mk, color=col, ms=8, mec=SURF, mew=1.5, elinewidth=2, capsize=0, zorder=3)
            ys.append(y)
            labels.append(lab)
            y += 1
        shuf = [v for t, v in fz.items() if t.startswith("shuf") and t.endswith(":kappa|tuned")]
        if shuf:
            for v in shuf:
                for sp, mk, dy in (("test", "o", -0.14), ("fresh", "s", 0.14)):
                    vv = v.get(sp, {}).get("vs_uniform_tuned")
                    if vv is not None:
                        ax.plot(100 * vv["diff"], y + dy, mk, color=MUTED, ms=7, mec=SURF, mew=1.2, zorder=3)
            ys.append(y)
            labels.append(f"shuffled kappa, tuned ({len(shuf)})")
            y += 1
        sr = o.get("shuffle_rank", {}).get("test")
        u = fz.get("uniform|tuned", {}).get("test", {}).get("norm")
        if sr and sr.get("matched_shuffles") and u:
            for m in sr["matched_shuffles"]:
                ax.plot(100 * (m - u), y, "o", color=MUTED, ms=5, alpha=0.8, zorder=3)
            ys.append(y)
            labels.append(f"matched-config shuffles ({len(sr['matched_shuffles'])}, test)")
            y += 1
        ax.axvline(0, color=INK2, lw=1.2)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, color=INK2)
        ax.invert_yaxis()
        ax.set_xlabel("difference vs tuned uniform (normalised points)")
        ctx = o.get("context", {}).get("isoc", {}).get("test")
        sub = ""
        if ctx and u:
            sub = f"\nweight-only Iso-C (context): {100 * (ctx['norm'] - u):+.1f} on test"
        ax.set_title(_title(sname, o) + sub, loc="left")
        ax.grid(axis="y", visible=False)
    h1 = axes[0][0].plot([], [], "o", color=INK2, ms=8)[0]
    h2 = axes[0][0].plot([], [], "s", color=INK2, ms=8)[0]
    fig.legend([h1, h2], ["test, 95 % CI (paired bootstrap)", "fresh, 95 % CI"], loc="lower center", ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_selectivity(all_out: dict, path: Path) -> None:
    """Dev performance as weighting becomes more selective (best alpha per temperature)."""
    plt = _style()
    panels = [(s, o) for s, o in all_out.items() if o.get("dev_best")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.8), squeeze=False)
    for ax, (sname, o) in zip(axes[0], panels):
        db = o["dev_best"]
        K = len(o["tasks"])

        def curve(rule):
            c = db.get(rule, {}).get("curve", [])
            return [x["n_eff"] for x in c], [100 * x["norm"] for x in c]
        sh = [r for r in db if r.startswith("shuf") and r.endswith(":kappa")]
        if sh:
            xs = [curve(r)[0] for r in sh]
            ys_ = [curve(r)[1] for r in sh]
            n = min(len(x) for x in xs)
            if n:
                xs = np.asarray([x[:n] for x in xs])
                ys_ = np.asarray([y[:n] for y in ys_])
                ax.fill_between(xs.mean(0), ys_.min(0), ys_.max(0), color=MUTED, alpha=0.22, lw=0,
                                label=f"shuffled kappa, range of {len(sh)} seeds")
        for rule, lab, col in (("kappa", "kappa (literal)", CAT[0]), ("kappa_rev", "reversed kappa", CAT[1]),
                               ("logkappa", "R1: log kappa", CAT[2]), ("kappa_res", "R2: kappa_res", CAT[3])):
            x, y = curve(rule)
            if x:
                ax.plot(x, y, "-o", color=col, ms=7, mec=SURF, mew=1.5, label=lab, zorder=3)
        u = db.get("uniform", {}).get("best")
        if u:
            ax.axhline(100 * u["norm"], color=INK2, lw=1.5, zorder=2)
            ax.annotate(f"tuned uniform {100 * u['norm']:.1f}", (K + 0.2, 100 * u["norm"]), xytext=(0, 4),
                        textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=11)
        ax.set_xlim(K + 0.3, 0.7)
        ax.set_xlabel("selectivity: mean effective number of experts per matrix")
        ax.set_ylabel("dev mean normalised accuracy (%)")
        ax.set_title(_title(sname, o) + "\n(best alpha at each temperature; argmin at n_eff = 1)", loc="left")
        ax.legend(fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_mechanism(path: Path, banks=(("l31_8b", 8), ("l32_3b", 8))) -> None:
    """Per adapted matrix: sigma_min of the expert matrices relative to the base checkpoint's BF16 resolution floor
    (x) vs the across-expert spread of kappa relative to kappa0 (y), one panel per projection type."""
    plt = _style()
    spec_dir = ROOT / "results" / "llm" / "spectral"  # weights-only diagnostics (never redirected)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), sharey=True)
    any_data = False
    for bi, (bank, T) in enumerate(banks):
        p = spec_dir / f"{bank}_T{T}_diagnostics.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open()))
        for ax, kind in zip(axes, BK.PROJ):
            sel = [r for r in rows if r["kind"] == kind]
            x = [float(r["median_smin_over_tau"]) for r in sel]
            y = [max(float(r["rel_spread"]), 1e-5) for r in sel]
            ax.scatter(x, y, s=48, color=CAT[bi], edgecolor=SURF, linewidth=1.2,
                       label=f"{BANK_LABEL[bank]} ({T} experts)", zorder=3)
            any_data = True
    for ax, kind in zip(axes, BK.PROJ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color=INK2, lw=1.2)
        ax.set_title(kind, loc="left")
        ax.set_xlabel("sigma_min(W_i) / BF16 floor of W0")
        ax.text(1.15, 0.97, "resolved", transform=ax.get_xaxis_transform(), color=INK2, fontsize=10, va="top")
        ax.text(0.87, 0.97, "unresolved", transform=ax.get_xaxis_transform(), color=INK2, fontsize=10, va="top",
                ha="right")
    axes[0].set_ylabel("across-expert spread of kappa / kappa0")
    if any_data:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_tail(all_out: dict, path: Path) -> None:
    """Tail-edit intervention: effects vs cell A (normalised points), with the numerical-noise floor shaded."""
    plt = _style()
    keys = [(s, k) for s, o in all_out.items() for k in sorted(o.get("tail", {})) if k.startswith("kappa|")]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(6.4 * len(keys), 5.0), squeeze=False)
    for ax, (sname, k) in zip(axes[0], keys):
        o = all_out[sname]
        t = o["tail"][k]
        split = k.rsplit("|", 1)[1]
        u = o["tail"].get(f"uniform|tuned|{split}", {})
        rows = []
        if "A" in t:
            a = t["A"]["norm"]
            for c, lab in (("B", "edited weights,\noriginal lambda (B-A)"), ("C", "original weights,\nrecomputed lambda (C-A)"),
                           ("D", "edited weights,\nrecomputed lambda (D-A)")):
                if c in t:
                    rows.append((lab, 100 * (t[c]["norm"] - a), CAT[0]))
        if "U-orig" in u and "U-edited" in u:
            rows.append(("uniform merge:\nedited - original", 100 * (u["U-edited"]["norm"] - u["U-orig"]["norm"]), MUTED))
        nf = None
        for kk, v in o.get("noise_floor", {}).items():
            if kk.startswith(split):
                nf = max(nf or 0.0, 100 * v["norm_range"])
        if nf:
            ax.axvspan(-nf, nf, color=MUTED, alpha=0.18, lw=0)
        for i, (lab, v, col) in enumerate(rows):
            ax.plot([0, v], [i, i], color=col, lw=2, zorder=2)
            ax.plot(v, i, "o", color=col, ms=9, mec=SURF, mew=1.5, zorder=3)
            ax.annotate(f"{v:+.2f}", (v, i), xytext=(0, 9), textcoords="offset points", ha="center",
                        color=INK2, fontsize=11)
        vals = [r[1] for r in rows] + ([-nf, nf] if nf else [])
        lo_, hi_ = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.15 * (hi_ - lo_ + 0.2)
        ax.set_xlim(lo_ - pad, hi_ + pad)
        ax.set_ylim(len(rows) - 0.4, -0.7)
        ax.axvline(0, color=INK2, lw=1.2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=10.5)
        lam = ""
        if "A" in t and "C" in t:
            lam = (f"\nlambda of edited expert ({t['A'].get('edited_expert') or ''}): "
                   f"{t['A']['lambda_edited_mean']:.3f} -> {t['C']['lambda_edited_mean']:.3f}")
        ax.set_title(f"{sname} {k.rsplit('|', 1)[0]}, {split}{lam}", loc="left", fontsize=11.5)
        ax.set_xlabel("change in mean normalised accuracy (points)")
        ax.grid(axis="y", visible=False)
    h = plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.18)
    fig.legend([h], ["numerical-noise range (identical merges up to 1e-7 jitter)"], loc="lower center", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- markdown
def md_setting(o: dict) -> str:
    L = [f"## {o['setting']}: {o['bank']} T{o['T']}", "", f"tasks: {', '.join(o['tasks'])}", ""]
    if o.get("base"):
        L.append("base (zero-shot): " + ", ".join(f"{sp} norm {100 * v['norm']:.2f} / joint {100 * v['joint']:.2f}"
                                                for sp, v in o["base"].items()))
    pct = lambda x, nd=2: "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}"
    if o.get("dev_best"):
        L += ["", "### Dev (val): best configuration per rule (mean normalised accuracy, %)", "",
              "| rule | best b | best alpha | dev norm | dev joint | n_eff | best at alpha 1 |", "|---|---|---|---|---|---|---|"]
        for rule, v in sorted(o["dev_best"].items(), key=lambda kv: -kv[1]["best"]["norm"]):
            bb, b1 = v["best"], v["best_alpha1"]
            ne = "" if bb["n_eff"] is None else f"{bb['n_eff']:.2f}"
            a1 = "" if b1 is None else f"{pct(b1['norm'])} (b {b1['b']})"
            L.append(f"| {rule} | {bb['b']} | {bb['alpha']} | {pct(bb['norm'])} | {pct(bb['joint'])} | {ne} | {a1} |")
    for group in ("frozen", "transfer"):
        t = o.get(group)
        if not t:
            continue
        L += ["", f"### {group.capitalize()} configurations: test and fresh (mean normalised accuracy %, diff vs tuned "
                  "uniform with paired 95 % CI)", "",
              "| tag | b | alpha | test norm | test diff [CI] | test joint | worst task | fresh norm | fresh diff [CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for tag, v in sorted(t.items()):
            te, fr = v.get("test", {}), v.get("fresh", {})

            def dd(x):
                d_ = x.get("vs_uniform_tuned")
                return "" if not d_ else f"{100 * d_['diff']:+.2f} [{100 * d_['lo']:+.2f}, {100 * d_['hi']:+.2f}]"
            b_ = te.get("b", fr.get("b", ""))
            a_ = te.get("alpha", fr.get("alpha", ""))
            L.append(f"| {tag} | {b_} | {a_} | {pct(te.get('norm'))} | {dd(te)} | {pct(te.get('joint'))} | "
                     f"{pct(te.get('worst_task_norm'), 1)} | {pct(fr.get('norm'))} | {dd(fr)} |")
    if o.get("shuffle_rank"):
        L += ["", "### kappa vs shuffled-score nulls", ""]
        for sp, v in o["shuffle_rank"].items():
            L.append(f"* {sp}: kappa {100 * v['kappa']:.2f}; rank {v['rank_among_all']} of {v['n_null'] + 1} "
                     f"(null range {100 * v['null_min']:.2f}-{100 * v['null_max']:.2f}, mean {100 * v['null_mean']:.2f})"
                     if v["n_null"] else f"* {sp}: no nulls yet")
    if o.get("noise_floor"):
        L += ["", "### Numerical-noise floor (uniform with 1e-7 coefficient jitter, and an exact repeat)", ""]
        for k, v in o["noise_floor"].items():
            L.append(f"* {k}: normalised range {100 * v['norm_range']:.2f} pt, joint range {100 * v['joint_range']:.2f} pt "
                     f"({len(v['runs'])} runs)")
    if o.get("context"):
        L += ["", "### Context baseline (weight-only Iso-C, source project's alpha)", ""]
        for m, v in o["context"].items():
            L.append(f"* {m}: " + ", ".join(f"{sp} norm {100 * x['norm']:.2f} / joint {100 * x['joint']:.2f}"
                                           for sp, x in v.items()))
    if o.get("tail"):
        L += ["", "### Tail-edit intervention (mean normalised accuracy %; lambda = edited expert's mean coefficient)", "",
              "| config / split | A | B | C | D | uniform orig | uniform edited |", "|---|---|---|---|---|---|---|"]
        for k, v in o["tail"].items():
            cell = lambda c: "" if c not in v else f"{100 * v[c]['norm']:.2f} (lam {v[c]['lambda_edited_mean']:.3f})"
            L.append(f"| {k} | {cell('A')} | {cell('B')} | {cell('C')} | {cell('D')} | {cell('U-orig')} | {cell('U-edited')} |")
    if o.get("tail_expert"):
        L += ["", "Edited expert alone (own task):"]
        for sp, v in o["tail_expert"].items():
            L.append(f"* {sp}: {v}")
    if o.get("gradpref"):
        L += ["", "### Allocation usefulness: within-matrix Spearman(score preference, -d loss / d lambda)", ""]
        for tag, v in o["gradpref"].items():
            L.append(f"* at {tag}: " + "; ".join(f"{k} median {x['median_rho']:+.3f} (frac>0 {x['frac_positive']:.2f})"
                                                 for k, x in v.items()))
    if o.get("fingerprint_collisions"):
        L += ["", f"**Dispatcher check: fingerprint collisions across rules: {o['fingerprint_collisions']}**"]
    else:
        L += ["", "Dispatcher check: no fingerprint collisions across distinct non-uniform configurations."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="S1,S2,S3,S4")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    SUM.mkdir(parents=True, exist_ok=True)
    (SUM / "figures").mkdir(parents=True, exist_ok=True)
    all_out = {}
    for sname in args.settings.split(","):
        if sname not in plan.SETTINGS:
            continue
        try:
            all_out[sname] = analyze_setting(sname, args.boot)
            print(f"{sname}: groups {all_out[sname]['groups']}", flush=True)
        except Exception as e:  # partial results must never kill the summary of the other settings
            import traceback

            print(f"{sname}: analysis error {e}\n{traceback.format_exc()}", flush=True)
    md = ["# LLM study: automatic summary", "",
          "Generated by experiments/llm/analyze.py from the raw records. Units: mean normalised accuracy in % (local "
          "accuracy / own-expert local accuracy, averaged over tasks) unless marked joint (argmax over all tasks' "
          "answers). Differences in normalised points.", ""]
    for o in all_out.values():
        md.append(md_setting(o))
    (SUM / "SUMMARY.md").write_text("\n".join(md))
    for fn, f in (("fig_llm_test_diffs.png", fig_test_diffs), ("fig_llm_selectivity.png", fig_selectivity),
                  ("fig_llm_tail_edit.png", fig_tail)):
        try:
            f(all_out, SUM / "figures" / fn)
        except Exception as e:
            print(f"figure {fn} failed: {e}", flush=True)
    try:
        fig_mechanism(SUM / "figures" / "fig_llm_mechanism.png")
    except Exception as e:
        print(f"figure mechanism failed: {e}", flush=True)
    print("analysis written to", SUM, flush=True)


if __name__ == "__main__":
    main()
