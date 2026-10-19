#!/usr/bin/env python
"""Generator for appendix tables C1-C6 (LoRA banks + RegMean) of the condition-number-merging report.

Every number in tables.md is computed here from the raw result files; nothing is typed by hand except the
"doc value" column of the cross-check table (copied from docs/AA_SIDE_REPORT.md, REPORT_02, RESEARCH_LOG).

READ-ONLY on /data/home/azpitarte/condmerge and /data/home/azpitarte/condmerge_aa.
Run:  OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /data/home/azpitarte/lora_merging_ica/.venv/bin/python make_tables.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# ----------------------------------------------------------------------------------------------------------------
# paths and constants
# ----------------------------------------------------------------------------------------------------------------
OUT_DIR = Path("/tmp/claude-1061/-data-home-azpitarte/3fa82598-2657-4652-97a2-9c24f2860353/scratchpad/report_build/C_lora_regmean")
AA = Path("/data/home/azpitarte/condmerge_aa/results/aa/lora")
MA = Path("/data/home/azpitarte/condmerge/results/regmean")
MASTER = Path("/data/home/azpitarte/condmerge")
AAREPO = Path("/data/home/azpitarte/condmerge_aa")
TASKS8 = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
INF = float("inf")
NAN = float("nan")

GAUGE_TOL = 1e-9    # threshold chosen here (not by the peer): product / spectrum unchanged by an exact gauge change
CORE_TOL = 1e-8     # threshold chosen here: QR-core spectrum vs dense SVD

LINES: list[str] = []          # output document
CHECKS: list[dict] = []        # cross-checks against docs
UNPARSED: list[str] = []       # anything that could not be parsed
MANIFEST: list[dict] = []      # source files used
VERIFY: list[dict] = []        # metric verification (absolute vs normalised) per file


def src_aa(name: str) -> str:
    return f"aa:results/aa/lora/{name}"


def src_ma(name: str) -> str:
    return f"master:results/regmean/{name}"


# ----------------------------------------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------------------------------------
def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, (np.floating,)) and np.isnan(x))


def pc(x, d: int = 2) -> str:
    """fraction -> percent string"""
    if isnan(x):
        return "-"
    return f"{100.0 * float(x):.{d}f}"


def nm(x, d: int = 2) -> str:
    if isnan(x):
        return "-"
    return f"{float(x):.{d}f}"


def sg(x, d: int = 2) -> str:
    """signed number"""
    if isnan(x):
        return "-"
    return f"{float(x):+.{d}f}"


def sci(x, d: int = 2) -> str:
    if isnan(x):
        return "-"
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    return f"{x:.{d}e}"


def gen(x, d: int = 4) -> str:
    """compact general format for parameter values (0.95, 10000.0 -> 1e4, ...)"""
    if x is None:
        return "None"
    if isinstance(x, str):
        return x
    if isnan(x):
        return "-"
    x = float(x)
    if math.isinf(x):
        return "inf"
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        return f"{x:.3g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{x:.{d}g}"


_NUMRE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?( ?(pt|%))?$|^-$|^inf$|^-inf$|^$")


def sset(vals) -> str:
    return "{" + ", ".join(gen(v) for v in vals) + "}"


def md_table(headers: list[str], rows: list[list], align: list[str] | None = None) -> str:
    n = len(headers)
    if align is None:
        align = []
        for j in range(n):
            col = [str(r[j]).strip() for r in rows]
            align.append("r" if col and all(_NUMRE.match(c) for c in col) else "l")
    assert len(align) == n, (len(align), n, headers)
    sep = []
    for a in align:
        sep.append({"l": ":---", "r": "---:", "c": ":---:"}[a])
    esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731  (a literal pipe inside a cell would split the column)
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        assert len(r) == n, (len(r), n, r)
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


COVER: list[dict] = []        # raw rows in a file vs data points shown in tables
TABLE_LOG: list[tuple] = []   # (section, table id, n rows, n cols)


def cover(file: str, raw: int, shown: int | None, where: str, how: str = "every row shown") -> None:
    COVER.append({"file": file, "raw": raw, "shown": shown, "where": where, "how": how})

CUR = {"sec": "", "id": ""}


def emit(s: str = "") -> None:
    LINES.append(s)


def h2(s: str) -> None:
    CUR["sec"] = s.split(".")[0]
    CUR["id"] = CUR["sec"]
    emit()
    emit(f"## {s}")
    emit()


def h3(s: str) -> None:
    CUR["id"] = s.split(".")[0]
    emit()
    emit(f"### {s}")
    emit()
    info = globals().get("SUBINFO", {}).get(CUR["id"])
    if info:
        emit(f"*What / split / metric / units.* {info[0]}  ")
        emit("*Source:* " + "; ".join(f"`{x}`" for x in info[1]))
        emit()


def para(s: str) -> None:
    m = re.match(r"\*\*(C\d+[a-z]?-?\d*)(: `([^`]+)`)?", s)
    if m:
        CUR["id"] = m.group(1) + (f" [{m.group(3)}]" if m.group(3) else "")
    emit(s)
    emit()


def tab(headers, rows, align=None) -> None:
    TABLE_LOG.append((CUR["sec"], CUR["id"], len(rows), len(headers)))
    emit(md_table(headers, rows, align))
    emit()


def bullets(items: list[str]) -> None:
    for it in items:
        emit(f"- {it}")
    emit()


def codeblock(s: str) -> None:
    emit("```")
    emit(s.rstrip("\n"))
    emit("```")
    emit()


def header_block(table_id: str, title: str, what: str, sources: list[str]) -> None:
    h2(f"{table_id}. {title}")
    para(f"**What is varied / split / metric / units.** {what}")
    para("**Source:** " + "; ".join(f"`{s}`" for s in sources))


# ----------------------------------------------------------------------------------------------------------------
# cross-check registry
# ----------------------------------------------------------------------------------------------------------------
def check(cid: str, doc: str, claim: str, doc_val, new_val, decimals: int | None = 1, note: str = "",
          text: bool = False, tol: float | None = None, sev: str | None = None) -> bool:
    """Register a doc-vs-data cross-check. Numeric: OK if |new - doc| <= 0.5 * 10**-decimals (+1e-9) or <= tol.
    A numeric miss is 'minor' when it is within 1.5 units of the last printed digit, else 'material'."""
    if text or decimals is None:
        ok = str(doc_val) == str(new_val)
        CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": gen(doc_val) if not isinstance(doc_val, str) else doc_val,
                       "new_val": gen(new_val) if not isinstance(new_val, str) else new_val,
                       "status": "OK" if ok else "DISCREPANCY", "sev": "" if ok else "wording/structure", "note": note})
        return ok
    thr = tol if tol is not None else 0.5 * 10 ** (-decimals) + 1e-9
    diff = abs(float(new_val) - float(doc_val))
    ok = diff <= thr
    sev_ = "" if ok else (sev if sev else ("minor" if diff <= 1.5 * 10 ** (-decimals) + 1e-9 else "material"))
    CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": f"{float(doc_val):.{max(decimals, 0)}f}",
                   "new_val": f"{float(new_val):.{decimals + 2 if decimals < 3 else decimals}f}",
                   "status": "OK" if ok else "DISCREPANCY", "sev": sev_, "note": note})
    return ok


def check_range(cid: str, doc: str, claim: str, doc_text: str, lo: float, hi: float, vals, decimals: int = 1, note: str = "") -> bool:
    """Doc quotes a range 'lo to hi'; OK if the recomputed values round into exactly that range."""
    vals = [float(v) for v in vals]
    ok = round(min(vals), decimals) == lo and round(max(vals), decimals) == hi
    CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": doc_text, "new_val": " / ".join(f"{v:.{decimals + 2}f}" for v in vals),
                   "status": "OK" if ok else "DISCREPANCY", "sev": "" if ok else "minor", "note": note})
    return ok


STASH: dict = {}


def check_array(cid: str, doc: str, claim: str, doc_arr, new_arr, decimals: int = 2, note: str = "", sev: str = "minor", sci_fmt: bool = False) -> bool:
    """A row of numbers quoted in a doc vs the recomputed row; OK if every entry rounds to the doc entry."""
    doc_arr = [float(x) for x in doc_arr]
    new_arr = [float(x) for x in new_arr]
    bad = [i for i, (a, b) in enumerate(zip(doc_arr, new_arr)) if abs(a - b) > 0.5 * 10 ** (-decimals) + 1e-9]
    ok = not bad and len(doc_arr) == len(new_arr)
    fmt = (lambda arr: " / ".join(f"{x:.3g}" for x in arr)) if sci_fmt else (lambda arr: " / ".join(f"{x:.{decimals}f}" for x in arr))  # noqa: E731
    CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": fmt(doc_arr), "new_val": fmt(new_arr),
                   "status": "OK" if ok else "DISCREPANCY", "sev": "" if ok else sev,
                   "note": note if ok else (note + ". " if note else "") + f"entries off (0-based): {bad}"})
    return ok


def check_bool(cid: str, doc: str, claim: str, doc_text: str, new_text: str, ok: bool, note: str = "", sev: str = "wording") -> bool:
    CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": doc_text, "new_val": new_text,
                   "status": "OK" if ok else "DISCREPANCY", "sev": "" if ok else sev, "note": note})
    return ok


def note_row(cid: str, doc: str, claim: str, doc_text: str, new_text: str, note: str = "") -> None:
    """Observation that qualifies a doc statement without contradicting a number (not counted as a discrepancy)."""
    CHECKS.append({"id": cid, "doc": doc, "claim": claim, "doc_val": doc_text, "new_val": new_text, "status": "NOTE", "sev": "", "note": note})


# ----------------------------------------------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------------------------------------------
def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def register(path: Path, label: str, n_rows: int | None = None) -> None:
    st = path.stat()
    MANIFEST.append({"label": label, "size": st.st_size,
                     "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                     "sha": sha12(path), "rows": n_rows})


def read_jsonl(path: Path, label: str) -> list[dict]:
    rows, bad = [], 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:  # pragma: no cover
                bad += 1
                UNPARSED.append(f"{label} line {i}: {e}")
    register(path, label, len(rows))
    return rows


def read_json(path: Path, label: str):
    key = ("json", str(path))
    if key in _CACHE:
        return _CACHE[key]
    try:
        d = json.loads(path.read_text())
    except Exception as e:  # pragma: no cover
        UNPARSED.append(f"{label}: {e}")
        raise
    register(path, label, None)
    _CACHE[key] = d
    return d


def tasks_mean(acc: dict) -> float:
    return float(np.mean([acc[t] for t in TASKS8]))


def order_key(vals, order):
    return [order.index(v) if v in order else 99 for v in vals]


_CACHE: dict = {}


# --- LoRA-side loaders -------------------------------------------------------------------------------------------
def lora_df(path: Path, label: str) -> pd.DataFrame:
    key = ("lora", str(path))
    if key in _CACHE:
        return _CACHE[key].copy()
    df = _lora_df(path, label)
    _CACHE[key] = df
    return df.copy()


def _lora_df(path: Path, label: str) -> pd.DataFrame:
    rows = read_jsonl(path, label)
    df = pd.DataFrame(rows)
    # verification that avg_acc is the plain mean of per-task absolute accuracies
    if "acc" in df.columns:
        diffs = [abs(r["avg_acc"] - tasks_mean(r["acc"])) for r in rows if isinstance(r.get("acc"), dict)]
        rng = [min(min(r["acc"].values()) for r in rows if isinstance(r.get("acc"), dict)),
               max(max(r["acc"].values()) for r in rows if isinstance(r.get("acc"), dict))]
        VERIFY.append({"file": label, "n": len(diffs), "max_diff_avg": max(diffs) if diffs else NAN,
                       "acc_min": rng[0], "acc_max": rng[1], "kind": "lora"})
    else:
        VERIFY.append({"file": label, "n": len(rows), "max_diff_avg": NAN, "acc_min": NAN, "acc_max": NAN,
                       "kind": "lora-noacc"})
    return df


# --- master-side loaders -----------------------------------------------------------------------------------------
_EXPERT_REF = None


def expert_ref():
    global _EXPERT_REF
    if _EXPERT_REF is None:
        p = MASTER / "results" / "experts" / "expert_accuracy.json"
        _EXPERT_REF = json.loads(p.read_text())["expert"]
        register(p, "master:results/experts/expert_accuracy.json")
    return _EXPERT_REF


def master_df(path: Path, label: str, split: str | None = None) -> pd.DataFrame:
    key = ("master", str(path))
    if key in _CACHE:
        return _CACHE[key].copy()
    df = _master_df(path, label, split)
    _CACHE[key] = df
    return df.copy()


def _master_df(path: Path, label: str, split: str | None = None) -> pd.DataFrame:
    """Flatten master run records: cfg keys + norm/acc/min_norm (percent) + per-task acc (percent) + extras."""
    rows = read_jsonl(path, label)
    out = []
    mx_norm, mx_acc = 0.0, 0.0
    for i, r in enumerate(rows):
        cfg, met = r["cfg"], r["metrics"]
        sp = cfg.get("split", split)
        ref = expert_ref()[sp]
        norm_re = float(np.mean([met["acc"][t] / ref[t] for t in TASKS8]))
        min_re = float(np.min([met["acc"][t] / ref[t] for t in TASKS8]))
        acc_re = tasks_mean(met["acc"])
        mx_norm = max(mx_norm, abs(norm_re - met["mean_norm"]), abs(min_re - met["min_norm"]))
        mx_acc = max(mx_acc, abs(acc_re - met["mean_acc"]))
        d = {"_i": i, **cfg}
        d["split"] = sp
        d["norm"] = 100 * met["mean_norm"]
        d["acc"] = 100 * met["mean_acc"]
        d["min_norm"] = 100 * met["min_norm"]
        for t in TASKS8:
            d[f"acc_{t}"] = 100 * met["acc"][t]
        for k, v in r.items():
            if k in ("cfg", "metrics", "key", "host"):
                continue
            if k == "n_images":
                d["n_images"] = json.dumps(v) if not isinstance(v, (int, float)) else v
                continue
            d[k] = v
        out.append(d)
    VERIFY.append({"file": label, "n": len(rows), "max_diff_norm": mx_norm, "max_diff_avg": mx_acc,
                   "kind": "master", "split": split or "cfg.split"})
    return pd.DataFrame(out)


# ====================================================================================================================
# SECTION FUNCTIONS (each appends to LINES)
# ====================================================================================================================


# ---- C1 -------------------------------------------------------------------------------------------------------------
def sec_C1():
    chk = read_json(AA / "lora_checks.json", src_aa("lora_checks.json"))
    st = pd.read_csv(AA / "lora_stats.csv")
    register(AA / "lora_stats.csv", src_aa("lora_stats.csv"), len(st))
    gauge = chk["gauge"]
    gs = chk["gauge_summary"]

    header_block("C1", "LoRA exactness and gauge checks (KnOTS release bank, rank 16)",
                 "Nothing is tuned here: each row is an algebraic or numerical check on the 8 KnOTS rank-16 adapters "
                 "(48 attention maps = 12 blocks x {q,k,v,out}_proj, 8 experts, 384 (map, expert) pairs; gamma = 1); "
                 "no split and no accuracy (spectral quantities only); values are unitless relative errors, ratios or counts.",
                 [src_aa("lora_checks.json"), src_aa("lora_stats.csv") + " (for the rows marked 'recomputed from lora_stats.csv')",
                  "aa:scripts/aa_lora_spectra.py (the script that wrote both files), aa:tests/test_aa_lora.py"])

    # recomputed gauge summary
    g_prod = max(x["prod_rel_err"] for x in gauge)
    g_core = max(x["core_spectrum_rel_err"] for x in gauge)
    ratA = {c: [x["kappa_A_after"] / x["kappa_A_before"] for x in gauge if x["R_cond"] == c] for c in (10.0, 1e4)}
    ratB = {c: [x["kappa_B_after"] / x["kappa_B_before"] for x in gauge if x["R_cond"] == c] for c in (10.0, 1e4)}
    med_A4 = float(np.median(ratA[1e4]))
    med_B4 = float(np.median(ratB[1e4]))
    same = (abs(g_prod - gs["max_prod_rel_err"]) <= 1e-30 and abs(g_core - gs["max_core_spectrum_rel_err"]) <= 1e-30
            and abs(med_A4 - gs["median_kappaA_ratio_cond1e4"]) < 1e-6 * med_A4
            and abs(med_B4 - gs["median_kappaB_ratio_cond1e4"]) < 1e-6 * med_B4)

    # recomputed from the csv
    names = list(dict.fromkeys(st["param"]))
    viol = int((st["smin_W"] > st["s_q_minus_r_W0"] * (1 + 1e-10)).sum())
    am = []
    rho = []
    for p, g in st.groupby("param", sort=False):
        am.append(int(g["kappa2_W"].values.argmin() == g["smin_W"].values.argmax()))
        rho.append(spearmanr(g["kappa2_W"], g["smin_W"]).statistic)
    kp_max = float(st["kplus_D_0.1"].max())
    trunc = float((st["numrank_D_0.1"] < st["rank_config"]).mean())
    numrank_ok = int((st["numrank_D_eps"] == st["rank_config"]).sum())
    test_names = re.findall(r"^def (test_\w+)", (AAREPO / "tests" / "test_aa_lora.py").read_text(), flags=re.M)
    n_tests = len(test_names)

    rows = [
        ["core_vs_dense_max_rel_err", sci(chk["core_vs_dense_max_rel_err"], 3),
         f"QR-core nonzero spectrum of gamma*B@A vs dense SVD, max over 384 pairs of max_j abs(s_dense_j - s_core_j)/s_core_0; criterion < {CORE_TOL:g} (chosen here)",
         "PASS" if chk["core_vs_dense_max_rel_err"] < CORE_TOL else "FAIL"],
        ["interlacing_violations", str(chk["interlacing_violations"]),
         "count of (map, expert) with sigma_q(W0+Delta) > sigma_{q-r}(W0); criterion == 0 (of 384)",
         "PASS" if chk["interlacing_violations"] == 0 else "FAIL"],
        ["interlacing, recomputed from lora_stats.csv", f"{viol} / {len(st)}",
         "rows with `smin_W` > `s_q_minus_r_W0` * (1 + 1e-10); criterion == 0", "PASS" if viol == 0 else "FAIL"],
        ["gauge_summary.max_prod_rel_err", sci(gs["max_prod_rel_err"], 3),
         f"max over 96 gauge tests of ||B'A' - BA||_F/||BA||_F (exact gauge change B->BR, A->R^-1 A); criterion < {GAUGE_TOL:g} (chosen here)",
         "PASS" if gs["max_prod_rel_err"] < GAUGE_TOL else "FAIL"],
        ["gauge_summary.max_core_spectrum_rel_err", sci(gs["max_core_spectrum_rel_err"], 3),
         f"max over 96 gauge tests of the core-spectrum change (relative to s_1); criterion < {GAUGE_TOL:g} (chosen here)",
         "PASS" if gs["max_core_spectrum_rel_err"] < GAUGE_TOL else "FAIL"],
        ["gauge_summary.median_kappaA_ratio_cond1e4", nm(gs["median_kappaA_ratio_cond1e4"], 1),
         "median of kappa(A) after / before a gauge change with cond(R) = 1e4 (factor kappa is NOT gauge invariant; demonstration, expect >> 1; criterion > 100, chosen here)",
         "PASS (demonstrates gauge dependence)" if gs["median_kappaA_ratio_cond1e4"] > 100 else "FAIL"],
        ["gauge_summary.median_kappaB_ratio_cond1e4", nm(gs["median_kappaB_ratio_cond1e4"], 1),
         "same for kappa(B)", "PASS (demonstrates gauge dependence)" if gs["median_kappaB_ratio_cond1e4"] > 100 else "FAIL"],
        ["gauge_summary reproduced from the 96 raw gauge entries", "identical" if same else "DIFFERENT",
         "max/median of the 96 entries vs the stored gauge_summary", "PASS" if same else "FAIL"],
        ["argmin-kappa == argmax-sigma_min, recomputed", f"{sum(am)} / {len(am)} maps",
         "per map, the expert with the smallest `kappa2_W` is the expert with the largest `smin_W` (LoRA: kappa is sigma_min only)",
         "PASS" if sum(am) == len(am) else "FAIL"],
        ["within-map Spearman(kappa2_W, smin_W), recomputed", f"mean {nm(np.mean(rho), 4)} (min {nm(np.min(rho), 4)}, max {nm(np.max(rho), 4)})",
         "mean over 48 maps of the Spearman rank correlation over the 8 experts (expect about -1)", "info"],
        ["kplus_D_0.1 maximum, recomputed", nm(kp_max, 4),
         "kappa_plus(Delta; tau = 0.1) is capped near 10 because sigma_1/(0.1 sigma_1) = 10 (threshold artifact)", "info (cap <= 10 holds)" if kp_max <= 10 else "FAIL"],
        ["updates truncated at tau = 0.1, recomputed", f"{100 * trunc:.1f} % ({int((st['numrank_D_0.1'] < st['rank_config']).sum())} / {len(st)})",
         "fraction of pairs with numrank_D_0.1 < 16", "info"],
        ["numerical rank at eps (fp32), recomputed", f"{numrank_ok} / {len(st)} equal 16",
         "`numrank_D_eps` == `rank_config` (all adapters have full rank 16)", "PASS" if numrank_ok == len(st) else "FAIL"],
        ["merge-correctness unit tests", f"{n_tests} test functions found",
         "aa:tests/test_aa_lora.py (" + ", ".join(test_names) + "). NOT executed here (read-only rule); the AA report states 5/5 pass",
         "not run"],
    ]
    check("C1-1", "AA_SIDE_REPORT s6", "QR-core spectrum exact (9e-10), in units of 1e-10", 9, chk["core_vs_dense_max_rel_err"] * 1e10, 0)
    check("C1-2", "AA_SIDE_REPORT s6", "rank-r interlacing holds (violations)", 0, chk["interlacing_violations"], 0)
    check("C1-3", "AA_SIDE_REPORT s6", "factor kappa changes by about 5000x under an exact gauge change (median kappa(A) ratio at cond(R) = 1e4)", 5000, med_A4, 0,
          f"kappa(A) ratio {med_A4:.0f}, kappa(B) ratio {med_B4:.0f}; 'about 5000' counted OK within 1000", tol=1000)
    check("C1-4", "AA_SIDE_REPORT s6", "5/5 merge-correctness tests (number of test functions in tests/test_aa_lora.py; not executed)", 5, n_tests, 0)
    check("C1-5", "AA_SIDE_REPORT s6", "kappa_plus(Delta; tau = 0.1) is capped at 10 (maximum over the 384 pairs)", 10, kp_max, 0)
    check("C1-6", "AA_SIDE_REPORT s6", "80% of updates truncated at tau = 0.1", 80, 100 * trunc, 0)
    check("C1-7", "AA_SIDE_REPORT s6", "argmin-kappa = argmax-sigma_min in 100% of maps", 100, 100 * sum(am) / len(am), 0)
    check("C1-8", "AA_SIDE_REPORT s1", "within-layer rho(kappa, sigma_min) = -1.00 (LoRA)", -1.00, float(np.mean(rho)), 2)
    h3("C1a. All checks")
    tab(["check", "value", "what it measures / criterion", "pass/fail"], rows, ["l", "r", "l", "l"])

    # per R_cond summary
    rows2 = []
    for c in (10.0, 1e4):
        sub = [x for x in gauge if x["R_cond"] == c]
        a = ratA[c]
        b = ratB[c]
        rows2.append([f"{c:g}", len(sub), sci(max(x["prod_rel_err"] for x in sub), 2), sci(max(x["core_spectrum_rel_err"] for x in sub), 2),
                      nm(np.median(a), 1), f"{nm(min(a), 1)} / {nm(max(a), 1)}", nm(np.median(b), 1), f"{nm(min(b), 1)} / {nm(max(b), 1)}",
                      sum(1 for x in sub if x["prod_rel_err"] < GAUGE_TOL and x["core_spectrum_rel_err"] < GAUGE_TOL)])
    h3("C1b. Gauge tests grouped by the condition number of the random gauge matrix R")
    tab(["cond(R)", "n tests", "max prod_rel_err", "max core_spectrum_rel_err", "median kappa(A) ratio", "min / max kappa(A) ratio",
         "median kappa(B) ratio", "min / max kappa(B) ratio", f"n pass (< {GAUGE_TOL:g})"], rows2)

    rows3 = []
    for x in gauge:
        ok = x["prod_rel_err"] < GAUGE_TOL and x["core_spectrum_rel_err"] < GAUGE_TOL
        rows3.append([x["param"], f"{x['R_cond']:g}", sci(x["prod_rel_err"], 2), sci(x["core_spectrum_rel_err"], 2),
                      nm(x["kappa_A_before"], 3), nm(x["kappa_A_after"], 1), nm(x["kappa_B_before"], 3), nm(x["kappa_B_after"], 1),
                      "PASS" if ok else "FAIL"])
    cover(src_aa("lora_checks.json") + " [gauge]", len(gauge), len(rows3), "C1c")
    h3(f"C1c. Every gauge test ({len(gauge)} rows = {len(names)} maps x {len({x['R_cond'] for x in gauge})} gauge strengths; expert = {TASKS8[0]} for the gauge tests)")
    tab(["param", "cond(R)", "prod_rel_err", "core_spectrum_rel_err", "kappa_A_before", "kappa_A_after", "kappa_B_before", "kappa_B_after", "pass"],
        rows3, ["l", "r", "r", "r", "r", "r", "r", "r", "l"])
    para("Legend: `prod_rel_err` = relative change of the product B@A after the gauge change; `core_spectrum_rel_err` = relative change "
         "of the nonzero singular values of B@A; `kappa_A`, `kappa_B` = condition numbers of the rank-16 factors A (16x768) and B (768x16) "
         "(gauge-dependent: changed by orders of magnitude although the function is identical). 'PASS' thresholds are set in this "
         "generator (1e-8 for the core check, 1e-9 for the gauge checks); the peer's files carry no thresholds.")


# ---- C2 -------------------------------------------------------------------------------------------------------------
def sec_C2():
    st = pd.read_csv(AA / "lora_stats.csv")
    header_block("C2", "LoRA spectra summary (KnOTS release bank, rank 16)",
                 "Descriptive statistics of per-(map, expert) spectra over the 4 attention map types (`module` = q_proj, k_proj, v_proj, out_proj; "
                 "12 blocks x 8 experts = 96 pairs each); no split and no accuracy; kappa2 and sigma values are unitless / weight-scale numbers, "
                 "ranks are counts. NOTE: lora_stats.csv exists for ONE bank only (the KnOTS release, loaded by aa_lora_spectra.py via load_knots_lora); "
                 "there is no `bank` column and no spectra file for lora_c5a, lora_c5a_seed1 or lora (vit20).",
                 [src_aa("lora_stats.csv") + f" ({len(st)} rows x {st.shape[1]} columns; full data there)", "aa:scripts/aa_lora_spectra.py"])
    mods = ["q_proj", "k_proj", "v_proj", "out_proj"]

    def gmean(x):
        return float(np.exp(np.mean(np.log(x))))

    def block(sub: pd.DataFrame, label: str):
        base = sub.drop_duplicates("param")
        return [label, len(sub), len(base),
                f"{sci(sub['kappa2_W'].median(), 3)}", f"{sci(sub['kappa2_W'].min(), 2)} / {sci(sub['kappa2_W'].max(), 2)}", sci(gmean(sub["kappa2_W"]), 3),
                sci(base["kappa2_W0"].median(), 3), f"{sci(base['kappa2_W0'].min(), 2)} / {sci(base['kappa2_W0'].max(), 2)}",
                nm((sub["kappa2_W"] / sub["kappa2_W0"]).median(), 3),
                sci(sub["smin_W"].median(), 3), sci(base["smin_W0"].median(), 3), nm((sub["smin_W"] < sub["smin_W0"]).mean(), 3),
                nm(sub["smax_W"].median(), 3)]
    rows = [block(st[st.module == m], m) for m in mods] + [block(st, "ALL")]
    h3("C2a. Full-weight conditioning kappa2(W) = sigma_1/sigma_q of W = W0 + Delta, by map type")
    tab(["module", "n pairs", "n maps", "kappa2_W median", "kappa2_W min / max", "kappa2_W geo-mean", "kappa2_W0 median (per map)",
         "kappa2_W0 min / max", "median kappa2_W / kappa2_W0", "smin_W median", "smin_W0 median (per map)", "frac smin_W < smin_W0", "smax_W median"], rows)

    def block2(sub: pd.DataFrame, label: str):
        return [label,
                sci(sub["fro_D"].median(), 3), sci(sub["spec_D"].median(), 3), nm(sub["stable_rank_D"].median(), 3), nm(sub["entropy_rank_D"].median(), 2),
                sci(sub["kappa2_D_dense"].median(), 2), sci(sub["sd_tail_over_head"].median(), 2),
                f"{int(sub['numrank_D_eps'].min())}-{int(sub['numrank_D_eps'].max())}",
                nm(sub["kplus_D_0.1"].median(), 3), nm(sub["numrank_D_0.1"].mean(), 2),
                nm(sub["kplus_D_0.01"].median(), 2), nm(sub["numrank_D_0.01"].mean(), 2),
                nm(sub["kplus_D_0.001"].median(), 2), nm(sub["numrank_D_0.001"].mean(), 2),
                nm(sub["kplus_D_0.0001"].median(), 2), nm(sub["numrank_D_0.0001"].mean(), 2)]
    rows = [block2(st[st.module == m], m) for m in mods] + [block2(st, "ALL")]
    h3("C2b. Spectrum of the update Delta = gamma * B @ A (rank 16 of 768), by map type")
    tab(["module", "fro_D median", "spec_D median", "stable_rank_D median", "entropy_rank_D median", "kappa2_D_dense median (numerical, Delta has rank 16)",
         "sd_tail_over_head median", "numrank_D_eps range", "kplus_D_0.1 median", "numrank_D_0.1 mean", "kplus_D_0.01 median", "numrank_D_0.01 mean",
         "kplus_D_0.001 median", "numrank_D_0.001 mean", "kplus_D_0.0001 median", "numrank_D_0.0001 mean"], rows)

    def block3(sub: pd.DataFrame, label: str):
        return [label, nm(sub["kappa_A"].median(), 3), f"{nm(sub['kappa_A'].min(), 2)} / {nm(sub['kappa_A'].max(), 2)}",
                nm(sub["kappa_B"].median(), 2), f"{nm(sub['kappa_B'].min(), 2)} / {nm(sub['kappa_B'].max(), 2)}"]
    rows = [block3(st[st.module == m], m) for m in mods] + [block3(st, "ALL")]
    h3("C2c. Factor condition numbers (gauge-dependent; see C1) by map type")
    tab(["module", "kappa_A median", "kappa_A min / max", "kappa_B median", "kappa_B min / max"], rows)

    # per-map table
    cover(src_aa("lora_stats.csv"), len(st), None, "C2a-C2g", "summarised (per map type, per map, per expert, per block); the 384 x 30 table is NOT reproduced - full data in the CSV")
    h3("C2d. Within-map behaviour across the 8 experts (48 maps)")
    per = []
    spread = []
    winners = []
    rho_list = []
    for p, g in st.groupby("param", sort=False):
        g = g.set_index("expert").loc[TASKS8]
        kmin_e = g["kappa2_W"].idxmin()
        kmax_e = g["kappa2_W"].idxmax()
        winners.append(kmin_e)
        sp = float(g["kappa2_W"].max() / g["kappa2_W"].min())
        rho = spearmanr(g["kappa2_W"], g["smin_W"]).statistic
        rho_list.append(rho)
        spread.append((g["module"].iloc[0], sp))
        per.append([p.replace("encoder.layers.", "L").replace(".self_attn.", ".").replace(".weight", ""),
                    sci(g["kappa2_W0"].iloc[0], 3), sci(g["kappa2_W"].min(), 3), sci(g["kappa2_W"].median(), 3), sci(g["kappa2_W"].max(), 3),
                    nm(sp, 3), kmin_e, sci(g["smin_W0"].iloc[0], 3), sci(g["smin_W"].max(), 3), nm(rho, 4)])
    tab(["map", "kappa2_W0", "kappa2_W min", "kappa2_W median", "kappa2_W max", "max/min over experts", "argmin-kappa expert", "smin_W0", "smin_W max", "Spearman(kappa, smin)"], per,
        ["l", "r", "r", "r", "r", "r", "l", "r", "r", "r"])
    sp_by = defaultdict(list)
    for m, sp in spread:
        sp_by[m].append(sp)
    rows = [[m, len(sp_by[m]), nm(np.median(sp_by[m]), 3), f"{nm(min(sp_by[m]), 3)} / {nm(max(sp_by[m]), 3)}"] for m in mods]
    rows.append(["ALL", len(spread), nm(np.median([s for _, s in spread]), 3), f"{nm(min(s for _, s in spread), 3)} / {nm(max(s for _, s in spread), 3)}"])
    h3("C2e. Relative spread of kappa2_W across the 8 experts of one map (max/min), by map type")
    tab(["module", "n maps", "median max/min", "min / max of max/min"], rows)

    # winners per expert
    cnt = {t: winners.count(t) for t in TASKS8}
    rows = []
    for t in TASKS8:
        g = st[st.expert == t]
        rows.append([t, cnt[t], sci(g["kappa2_W"].median(), 3), sci(g["smin_W"].median(), 3), sci(g["fro_D"].median(), 3), nm(g["stable_rank_D"].median(), 3)])
    h3("C2f. Per expert: how many of the 48 maps does it win under argmin-kappa, and its typical spectra")
    tab(["expert", "# maps where it is argmin-kappa2_W", "kappa2_W median", "smin_W median", "fro_D median", "stable_rank_D median"], rows)
    para(f"Argmin-kappa win counts in TASKS8 order: {[cnt[t] for t in TASKS8]} (this vector must equal the `counts` stored in "
         "`gentle.json` for `argmin_kappa`; checked in C4c).")

    # per layer x module median kappa
    rows = []
    for l in range(12):
        row = [str(l)]
        for m in mods:
            sub = st[(st.layer == l) & (st.module == m)]
            row.append(sci(sub["kappa2_W"].median(), 2))
        row.append(sci(st[st.layer == l]["kappa2_W0"].median(), 2))
        rows.append(row)
    h3("C2g. Median kappa2_W over the 8 experts, per block and map type (and median kappa2_W0 of the block's 4 base maps)")
    tab(["block"] + mods + ["kappa2_W0 median (4 maps)"], rows)

    para("Legend: `kappa2_W` = sigma_1/sigma_q of W0 + Delta_i in fp64; `kappa2_W0` the same for the base map; `smin_W` its smallest singular value; "
         "`fro_D`, `spec_D` = Frobenius and spectral norm of Delta; `stable_rank_D` = fro^2/spec^2; `entropy_rank_D` = exp(entropy of the normalized squared spectrum); "
         "`kplus_D_tau` = sigma_1/sigma_k with k = number of singular values above tau*sigma_1 (`numrank_D_tau` = that k); `kappa2_D_dense` = ordinary kappa2 of the "
         "dense rank-16 update (numerically about 1e11-1e14, i.e. meaningless); `kappa_A`, `kappa_B` = factor condition numbers (gauge-dependent).")


# ---- C3 -------------------------------------------------------------------------------------------------------------
def load_rule_sweep() -> pd.DataFrame:
    if ("sweep",) in _CACHE:
        return _CACHE[("sweep",)].copy()
    files = sorted(glob.glob(str(AA / "rule_sweep_dev_shard*.jsonl")))
    parts = []
    for fp in files:
        d = lora_df(Path(fp), src_aa(Path(fp).name))
        d["_file"] = Path(fp).name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    _CACHE[("sweep",)] = df
    return df.copy()


def sec_C3():
    raw = load_rule_sweep()
    n_raw = len(raw)
    assert raw["bank"].nunique() == 1 and raw["split"].nunique() == 1 and raw["budget"].nunique() == 1
    bank, split, budget = raw["bank"].iloc[0], raw["split"].iloc[0], raw["budget"].iloc[0]
    # duplicates across shards (same chash) must be identical
    g = raw.groupby("chash")["avg_acc"].agg(["min", "max", "count"])
    n_dup_groups = int((g["count"] > 1).sum())
    dup_inconsistent = int(((g["max"] - g["min"]) > 0).sum())
    df = raw.drop_duplicates("chash", keep="first").copy()
    df["pct"] = 100 * df["avg_acc"]
    shard_counts = raw.groupby("_file").size().to_dict()
    header_block("C3", "LoRA literal-rule sweep (KnOTS release bank), dev",
                 f"Varied: the score rule (`label`), the temperature multiplier `b` (beta = b / median-across-maps std of the rule's scores; b = 0 is uniform, b = inf is argmin selection) and the update scale `alpha` "
                 f"(merge W0 + alpha * sum_i lambda_i Delta_i on the 48 adapted maps); split = dev (20% of the official test set); budget A (weights only); "
                 f"metric = `avg_acc`, the ABSOLUTE accuracy averaged over the 8 tasks (verified: avg_acc == mean of the per-task `acc` in every row, max abs diff {max(v['max_diff_avg'] for v in VERIFY if v['file'].startswith('aa:results/aa/lora/rule_sweep')):.1e}); units = percent (x100).",
                 [src_aa(f"rule_sweep_dev_shard{i}.jsonl") for i in range(3)] + ["aa:scripts/aa_lora_rule_sweep.py"])
    para(f"Data notes: bank = `{bank}` in all {n_raw} raw rows (this sweep exists for the KnOTS release bank only; other banks appear in C4b/C4f); split `{split}`, budget `{budget}`; "
         f"rows per shard file {shard_counts}; {n_dup_groups} configurations appear in two shards (identical `chash`) - {dup_inconsistent} of them with different accuracy; "
         f"after removing the duplicates {len(df)} unique configurations remain, and all tables below use these. "
         "The grid is ragged: the first launch used the full-FT alpha grid (kept in the shard files). Labels with alpha above 1.6: "
         + ", ".join(f"`{l_}`" for l_ in sorted({l_ for l_ in df[df.alpha > 1.6].label})) + f" (alpha up to {df.alpha.max():g}); all other labels have alpha in {sset(sorted(set(df[~df.label.isin(set(df[df.alpha > 1.6].label))].alpha)))}.")

    expected = ["kappa", "kappa_rev", "logkappa", "logkappa_rev", "kappa_z", "kappa_z_rev", "dnorm", "dnorm_rev", "dlogkplus_0.01", "dlogkplus_0.01_rev",
                "dlogkplus_0.001", "dlogkplus_0.001_rev", "dstable_rank", "dentropy_rank", "factor_logkappaA", "factor_logkappaA_gauged", "shuf0:kappa",
                "shuf1:kappa", "shuf2:kappa", "global:kappa", "global:logkappa", "uniform", "metagpt", "uniform_rank16"] + [f"single:{t}" for t in TASKS8]
    present = set(df["label"])
    missing = [e for e in expected if e not in present]
    para("Rules defined in `aa_lora_rule_sweep.py` but with NO row in the dev shards: " + (", ".join(f"`{m}`" for m in missing) if missing else "none") +
         ". (`uniform_rank16` and the frozen shuffles at b = 128 are evaluated on test in C4g.)")

    label_order = ["uniform", "kappa", "kappa_rev", "logkappa", "logkappa_rev", "kappa_z", "kappa_z_rev", "global:kappa", "shuf0:kappa", "shuf1:kappa", "shuf2:kappa",
                   "dnorm", "dnorm_rev", "dlogkplus_0.01", "dlogkplus_0.01_rev", "metagpt"] + [f"single:{t}" for t in TASKS8]
    labels = [l for l in label_order if l in present] + sorted(present - set(label_order))
    alphas = sorted(df["alpha"].unique())

    # C3a best per rule
    uni = df[df.label == "uniform"].set_index("alpha")["pct"]
    ub_alpha = uni.idxmax()
    ub = uni.max()
    rows = []
    best_rows = {}
    for l in labels:
        s = df[df.label == l]
        i = s["pct"].idxmax()
        r = s.loc[i]
        best_rows[l] = r
        bstr = "-" if isnan(r["b"]) else gen(r["b"])
        rows.append([l, len(s), bstr, gen(r["alpha"]), nm(r["pct"], 2), sg(r["pct"] - ub, 2), nm(r["n_eff_mean"], 2)])
    rows_sorted = sorted(rows, key=lambda x: -float(x[4]))
    h3("C3a. Best (b, alpha) per rule, dev")
    tab(["rule (`label`)", "# configs", "best b", "best alpha", "best dev avg_acc (abs. %)", f"minus best uniform ({ub:.2f} at alpha {ub_alpha:g}) (pt)", "n_eff_mean at best"], rows_sorted,
        ["l", "r", "r", "r", "r", "r", "r"])
    single_best = max((best_rows[l]["pct"], l) for l in labels if l.startswith("single:"))
    para(f"Best single expert (max over the 8 `single:*` rows): **{single_best[1]}** at {nm(single_best[0], 2)}. Best uniform: {nm(ub, 2)} (alpha {ub_alpha:g}). "
         "`b` is blank for `single:*` and `metagpt` (not temperature rules). n_eff_mean = mean over maps of exp(entropy(lambda)) (8 = uniform, 1 = one-hot).")

    # C3b best at each alpha
    rows = []
    for l in labels:
        s = df[df.label == l]
        row = [l]
        for a in alphas:
            t = s[s.alpha == a]
            if len(t) == 0:
                row.append("-")
                continue
            i = t["pct"].idxmax()
            r = t.loc[i]
            if l in ("uniform",) or l.startswith("single:") or l == "metagpt":
                row.append(nm(r["pct"], 2))
            else:
                row.append(f"{nm(r['pct'], 2)} (b={gen(r['b'])})")
        rows.append(row)
    h3("C3b. Best dev accuracy (%) at each alpha, maximised over b (the b attaining it in brackets)")
    tab(["rule"] + [f"alpha={a:g}" for a in alphas], rows)

    # C3c full grid
    h3("C3c. Full grid: dev avg_acc (abs. %) for every (rule, b) x alpha that was run (blank = not run)")
    rows = []
    for l in labels:
        s = df[df.label == l]
        bs = sorted(s["b"].fillna(-1).unique(), key=lambda x: (x == INF, x))
        for b in bs:
            t = s[(s["b"].fillna(-1) == b)]
            row = [l, "-" if b == -1 else gen(b)]
            for a in alphas:
                u = t[t.alpha == a]
                row.append(nm(u["pct"].iloc[0], 2) if len(u) else "")
            rows.append(row)
    tab(["rule", "b"] + [f"alpha={a:g}" for a in alphas], rows)
    cover("aa:results/aa/lora/rule_sweep_dev_shard{0,1,2}.jsonl (unique configs)", len(df), sum(1 for r_ in rows for c_ in r_[2:] if c_ != ""), "C3c",
          f"every unique config shown ({n_raw} raw rows = {len(df)} unique + {n_raw - len(df)} cross-shard duplicates with identical results)")
    para("Legend: `kappa` = softmax(-beta * kappa2(W_i)) per map (the literal rule), `kappa_rev` = +beta (heaviest weight on the largest kappa), `logkappa` = on log kappa2, "
         "`kappa_z` = on the per-map z-scored kappa, `shufK:kappa` = the same scores permuted across experts within each map (seed K; matched-concentration null), "
         "`global:kappa` = one weight vector per expert from the mean score, `dnorm` = on the Frobenius norm of Delta (magnitude control), `dlogkplus_0.01` = log kappa_plus(Delta; tau = 0.01), "
         "`metagpt` = lambda proportional to ||Delta_i||^2 summed over all maps, `single:<task>` = one-hot on that expert for every map, `uniform` = lambda = 1/8 (Task Arithmetic with scale alpha). "
         "The `_rev` variants flip the sign of the score.")

    # cross-checks
    k = best_rows["kappa"]
    check("C3-1", "AA_SIDE_REPORT s6", "KnOTS dev: kappa best (literal rule)", 54.3, k["pct"], 1,
          f"b={gen(k['b'])}, alpha={k['alpha']:g}; note logkappa b={gen(best_rows['logkappa']['b'])} alpha={best_rows['logkappa']['alpha']:g} reaches {best_rows['logkappa']['pct']:.2f}")
    check("C3-2", "AA_SIDE_REPORT s6", "KnOTS dev: kappa best uses b = 128", 128, k["b"], 0)
    check("C3-3", "AA_SIDE_REPORT s6/s9", "KnOTS dev: best uniform (alpha 0.8)", 52.3, ub, 1, f"alpha={ub_alpha:g}")
    check("C3-4", "AA_SIDE_REPORT s6", "KnOTS dev: best single expert", 54.9, single_best[0], 1, f"{single_best[1]}")


# ---- C4 -------------------------------------------------------------------------------------------------------------
def pct_less(null, obs) -> float:
    null = np.asarray(null, float)
    return float(100 * np.mean(null < obs))


def sec_C4():
    st = pd.read_csv(AA / "lora_stats.csv")
    names = list(dict.fromkeys(st["param"]))
    header_block("C4", "LoRA selection analyses",
                 "Varied (per sub-table): C4a/C4b the selection scheme (argmin-kappa vs count-preserving, free random, global-frequency selections; 4 banks), "
                 "C4c gentlest/strongest hard selections, C4d the oracle allocation vs spectral scores, C4e single-map hand-overs, C4f the transfer of master's rule "
                 "`logkappa_q90_rev`, C4g frozen test configurations; split = dev unless a sub-table says test (test_freeze, oracle test column); metric = absolute `avg_acc` "
                 "(per-task `acc` averaged, verified per row below) in percent (x100) unless it is a correlation or a rank.",
                 [src_aa(f) for f in ["selection_controls.jsonl", "selection_controls_null20.jsonl", "replicate_banks.jsonl", "gentle.json", "oracle_vs_scores.json",
                                      "oracle_vs_scores.md", "single_map_swap_summary.json", "single_map_swap.jsonl", "q90_transfer.jsonl", "test_freeze.jsonl"]])

    # ---------------- C4a --------------------------------------------------------------------------------------
    sc5 = lora_df(AA / "selection_controls.jsonl", src_aa("selection_controls.jsonl"))
    sc20 = lora_df(AA / "selection_controls_null20.jsonl", src_aa("selection_controls_null20.jsonl"))
    for d in (sc5, sc20):
        d["pct"] = 100 * d["avg_acc"]
    h3("C4a. Selection controls on the KnOTS release bank (dev, budget A): argmin-kappa vs 5- and 20-seed nulls")
    para("`kappa_argmin` = per map the expert with the smallest kappa2(W_i) takes the whole map (b = inf); `global_freq` = lambda_i = (# maps won by expert i)/48 on every map; "
         "`count_preserving` = the same multiset of 48 per-map winners randomly reassigned across maps (seed s); `free_random_selection` = each map assigned to a uniformly random expert (seed s).")
    rows = [[r.label, gen(r.alpha), "-" if isnan(r.seed) else str(int(r.seed)), nm(r.pct, 2)] for r in sc5.itertuples()]
    para(f"**C4a-1. Every row of `selection_controls.jsonl` ({len(sc5)} rows; {int(sc5.seed.nunique())} seeds; alpha in {sset(sorted(set(sc5.alpha)))})**")
    tab(["label", "alpha", "seed", "dev avg_acc (abs. %)"], rows)
    cover(src_aa("selection_controls.jsonl"), len(sc5), len(rows), "C4a-1")

    para(f"**C4a-2. Every row of `selection_controls_null20.jsonl` ({len(sc20)} rows): the {int(sc20.seed.nunique())}-seed nulls (alpha in {sset(sorted(set(sc20.alpha)))} only)**")
    rows = []
    for s in range(20):
        row = [str(s)]
        for a in (0.8, 1.0):
            for lab in ("count_preserving", "free_random_selection"):
                v = sc20[(sc20.label == lab) & (sc20.alpha == a) & (sc20.seed == s)]
                assert len(v) == 1
                row.append(nm(v["pct"].iloc[0], 2))
        rows.append(row)
    for lab in ("kappa_argmin", "global_freq"):
        row = [lab + " (single run)"]
        for a in (0.8, 1.0):
            v = sc20[(sc20.label == lab) & (sc20.alpha == a)]
            assert len(v) == 1
            row += [nm(v["pct"].iloc[0], 2), ""]
        rows.append(row)
    tab(["seed", "count_preserving a=0.8", "free_random_selection a=0.8", "count_preserving a=1.0", "free_random_selection a=1.0"], rows)
    cover(src_aa("selection_controls_null20.jsonl"), len(sc20), sum(1 for r_ in rows for c_ in r_[1:] if c_ != ""), "C4a-2")

    para("**C4a-3. Percentile of argmin-kappa in the nulls** (percentile = 100 * fraction of null runs strictly below argmin-kappa; `p_ge` = fraction of null runs >= argmin-kappa; `p_(k+1)/(N+1)` = (1 + # null >= obs)/(N + 1))")
    rows = []
    perc = {}
    for src, d in (("null20 (20 seeds)", sc20), ("selection_controls (5 seeds)", sc5)):
        for a in (0.8, 1.0, 1.3):
            ka = d[(d.label == "kappa_argmin") & (d.alpha == a)]
            if len(ka) == 0:
                continue
            obs = ka["pct"].iloc[0]
            for lab in ("count_preserving", "free_random_selection"):
                nl = d[(d.label == lab) & (d.alpha == a)]["pct"].values
                if len(nl) == 0:
                    continue
                gf = d[(d.label == "global_freq") & (d.alpha == a)]["pct"].iloc[0]
                p = pct_less(nl, obs)
                perc[(src[:6], lab, a)] = p
                rows.append([src, lab, gen(a), len(nl), nm(obs, 2), nm(nl.mean(), 2), nm(nl.std(ddof=1), 2), nm(nl.min(), 2), nm(nl.max(), 2),
                             nm((obs - nl.mean()) / nl.std(ddof=1), 2), nm(p, 0), nm(float(np.mean(nl >= obs)), 3), nm((1 + np.sum(nl >= obs)) / (len(nl) + 1), 3), nm(gf, 2)])
    tab(["file", "null", "alpha", "N", "kappa_argmin avg_acc (abs. %)", "null mean", "null sd", "null min", "null max", "z of argmin", "percentile", "p_ge", "p_(k+1)/(N+1)", "global_freq (%)"], rows)
    check("C4a-1", "AA_SIDE_REPORT s6", "KnOTS release: argmin-kappa percentile of the 20-seed free-random null, alpha 0.8", 95, perc[("null20", "free_random_selection", 0.8)], 0)
    check("C4a-2", "AA_SIDE_REPORT s6", "KnOTS release: argmin-kappa percentile of the 20-seed free-random null, alpha 1.0", 95, perc[("null20", "free_random_selection", 1.0)], 0)
    check("C4a-3", "AA_SIDE_REPORT s6", "KnOTS release: argmin-kappa percentile of the 20-seed count-preserving null, alpha 0.8 (doc: '95th percentile ... count-preserving and free')", 95, perc[("null20", "count_preserving", 0.8)], 0)
    check("C4a-4", "AA_SIDE_REPORT s6", "KnOTS release: argmin-kappa percentile of the 20-seed count-preserving null, alpha 1.0", 95, perc[("null20", "count_preserving", 1.0)], 0)

    # singularity-avoidance reconstruction
    tab_st = st.set_index(["param", "expert"])
    S_ = np.array([[tab_st.loc[(n_, t_), "smin_W"] for t_ in TASKS8] for n_ in names])
    S0_ = np.array([tab_st.loc[(n_, TASKS8[0]), "smin_W0"] for n_ in names])
    K_ = np.array([[tab_st.loc[(n_, t_), "kappa2_W"] for t_ in TASKS8] for n_ in names])
    idx_ = np.arange(len(names))
    free = sc20[sc20.label == "free_random_selection"]
    defs_ = OrderedDict([("picks whose sigma_min is below the median sigma_min of the 8 experts of that map", lambda sel: int((S_[idx_, sel] < np.median(S_, 1)).sum())),
                         ("picks whose kappa2 is above the mean kappa2 of the 8 experts of that map", lambda sel: int((K_[idx_, sel] > K_.mean(1)).sum())),
                         ("picks whose sigma_min is below the BASE map's sigma_min (sigma_min-decreasing w.r.t. W0)", lambda sel: int((S_[idx_, sel] < S0_).sum()))])
    rows = []
    recon = {}
    for dn, fdef in defs_.items():
        row = [dn]
        for alphas_ in ((0.8, 1.0), (0.8,), (1.0,)):
            xs, ys = [], []
            for a_ in alphas_:
                for sd_ in range(20):
                    sel = np.random.default_rng(1000 + sd_).integers(0, 8, len(names))
                    xs.append(fdef(sel))
                    ys.append(float(free[(free.alpha == a_) & (free.seed == sd_)]["avg_acc"].iloc[0]))
            r_ = spearmanr(xs, ys)
            row += [str(len(xs)), sg(r_.statistic, 3), nm(r_.pvalue, 3)]
            if alphas_ == (0.8, 1.0):
                recon[dn] = (r_.statistic, r_.pvalue)
        rows.append(row)
    para("**C4a-4. 'Singularity avoidance' check, re-derived** (doc: 'across 40 random selections, accuracy does not track the number of sigma_min-decreasing picks (rho +0.14, p 0.37)'). "
         "The files do not store the random selections; they are regenerated here from the seeds in `aa_lora_selection_controls.py` (`np.random.default_rng(1000 + s).integers(0, 8, 48)`, s = 0..19), "
         "paired with the stored accuracy of `free_random_selection` seed s. The doc does not define 'sigma_min-decreasing', so three definitions are shown (Spearman over the pooled alpha 0.8 + 1.0 runs, n = 40, and per alpha, n = 20):")
    tab(["definition of the count", "n (alpha 0.8+1.0)", "Spearman rho", "p", "n (alpha 0.8)", "rho", "p", "n (alpha 1.0)", "rho", "p"], rows)
    k0 = list(recon)[0]
    note_row("C4a-5", "AA_SIDE_REPORT s6", "singularity avoidance: rho +0.14, p 0.37 over 40 random selections", "+0.14 / 0.37",
             f"{recon[k0][0]:+.3f} / {recon[k0][1]:.3f} (median-based definition); {recon[list(recon)[2]][0]:+.3f} / {recon[list(recon)[2]][1]:.3f} (base-relative definition)",
             "reconstruction; not a stored result. Qualitative claim (no significant tracking) holds under all three definitions; the doc's +0.14 is within 0.01 of the first two")

    # ---------------- C4b --------------------------------------------------------------------------------------
    rb = lora_df(AA / "replicate_banks.jsonl", src_aa("replicate_banks.jsonl"))
    rb["pct"] = 100 * rb["avg_acc"]
    banks_in_file = list(dict.fromkeys(rb["bank"]))
    h3("C4b. Cross-bank replication of 'argmin-kappa beats random per-map selection' (dev, budget A)")
    para(f"`replicate_banks.jsonl` has {len(rb)} rows and contains **{len(banks_in_file)} banks: {banks_in_file}**. The KnOTS release bank is NOT in this file; "
         "its row below is assembled from `rule_sweep_dev_shard*.jsonl` (uniform; `kappa` b=128) and `selection_controls_null20.jsonl` (argmin and the 20 free random selections). "
         "Per bank: uniform at alpha in {0.6, 0.8, 1.0}; `kappa_argmin` and `kappa_b128` at alpha in {0.8, 1.0}; 20 `free_random_selection` runs per alpha.")
    sweep = load_rule_sweep().drop_duplicates("chash")
    sweep["pct"] = 100 * sweep["avg_acc"]
    rows = []
    percs = {}
    summ = {}
    bank_order = ["knots_release", "lora_c5a", "lora_c5a_seed1", "lora"]
    pretty = {"knots_release": "KnOTS release", "lora_c5a": "KnOTS recipe, seed 0 (lora_c5a)", "lora_c5a_seed1": "KnOTS recipe, seed 1 (lora_c5a_seed1)", "lora": "vit20, 2000 steps (lora)"}
    for bk in bank_order:
        if bk == "knots_release":
            u = {a: float(sweep[(sweep.label == "uniform") & (sweep.alpha == a)]["pct"].iloc[0]) for a in (0.6, 0.8, 1.0)}
            ka = {a: float(sc20[(sc20.label == "kappa_argmin") & (sc20.alpha == a)]["pct"].iloc[0]) for a in (0.8, 1.0)}
            kb = {a: (float(sweep[(sweep.label == "kappa") & (sweep.b == 128) & (sweep.alpha == a)]["pct"].iloc[0])
                      if len(sweep[(sweep.label == "kappa") & (sweep.b == 128) & (sweep.alpha == a)]) else NAN) for a in (0.8, 1.0)}
            nulls = {a: sc20[(sc20.label == "free_random_selection") & (sc20.alpha == a)]["pct"].values for a in (0.8, 1.0)}
        else:
            s = rb[rb.bank == bk]
            u = {a: float(s[(s.label == "uniform") & (s.alpha == a)]["pct"].iloc[0]) for a in (0.6, 0.8, 1.0)}
            ka = {a: float(s[(s.label == "kappa_argmin") & (s.alpha == a)]["pct"].iloc[0]) for a in (0.8, 1.0)}
            kb = {a: float(s[(s.label == "kappa_b128") & (s.alpha == a)]["pct"].iloc[0]) for a in (0.8, 1.0)}
            nulls = {a: s[(s.label == "free_random_selection") & (s.alpha == a)]["pct"].values for a in (0.8, 1.0)}
            assert all(len(nulls[a]) == 20 for a in nulls), bk
        pr = {a: pct_less(nulls[a], ka[a]) for a in (0.8, 1.0)}
        percs[bk] = pr
        bu = max(u.values())
        kbbest = np.nanmax(list(kb.values()))
        summ[bk] = (kbbest, bu, max(u, key=u.get), dict(u))
        rows.append([pretty[bk] + ("" if bk != "knots_release" else " (assembled, see note)"),
                     f"{nm(u[0.6])} / {nm(u[0.8])} / {nm(u[1.0])}", nm(bu),
                     f"{nm(ka[0.8])} / {nm(ka[1.0])}", f"{nm(kb[0.8])} / {nm(kb[1.0])}",
                     f"{nm(nulls[0.8].mean())} +- {nm(nulls[0.8].std(ddof=1))} [{nm(nulls[0.8].min())}, {nm(nulls[0.8].max())}]",
                     f"{nm(nulls[1.0].mean())} +- {nm(nulls[1.0].std(ddof=1))} [{nm(nulls[1.0].min())}, {nm(nulls[1.0].max())}]",
                     f"{pr[0.8]:.0f} / {pr[1.0]:.0f}", f"{nm(kbbest)} vs {nm(bu)}"])
    para("**C4b-1. Per bank summary** (accuracies in %, dev; null = mean +- sd [min, max] of the 20 free random selections; percentile of `kappa_argmin` in that null = 100 * fraction of nulls strictly below it)")
    tab(["bank", "uniform a=0.6 / 0.8 / 1.0", "best uniform", "kappa_argmin a=0.8 / 1.0", "kappa_b128 a=0.8 / 1.0", "free-random null a=0.8", "free-random null a=1.0",
         "argmin-kappa percentile a=0.8 / 1.0", "best kappa_b128 vs best uniform"], rows, ["l"] + ["r"] * 8)
    para(f"Note for the KnOTS release row: `kappa_b128` at alpha 0.8 was not run in the rule sweep (b = 128 exists only for alpha in {sset(sorted(set(sweep[(sweep.label == 'kappa') & (sweep.b == 128)].alpha)))}), shown as '-'.")
    exp = {"knots_release": (95, 95, 54.3, 52.3), "lora_c5a": (95, 100, 55.3, 53.5), "lora_c5a_seed1": (55, 45, 51.8, 54.1), "lora": (20, 15, 58.0, 64.6)}
    for bk, (p8, p10, kb_doc, ub_doc) in exp.items():
        check(f"C4b-{bk}-pct0.8", "AA_SIDE_REPORT s6 cross-bank table", f"{bk}: argmin-kappa percentile at alpha 0.8", p8, percs[bk][0.8], 0)
        check(f"C4b-{bk}-pct1.0", "AA_SIDE_REPORT s6 cross-bank table", f"{bk}: argmin-kappa percentile at alpha 1.0", p10, percs[bk][1.0], 0)
        check(f"C4b-{bk}-kb", "AA_SIDE_REPORT s6 cross-bank table", f"{bk}: kappa b128 (best of alpha 0.8/1.0)", kb_doc, summ[bk][0], 1)
        altf = {"knots_release": "regmean_knots_release_dev.jsonl", "lora_c5a": "regmean_lora_c5a_dev.jsonl", "lora": "regmean_lora_dev.jsonl"}.get(bk)
        alt_note = ""
        if altf:
            du = lora_df(AA / altf, src_aa(altf))
            du = du[du.method == "uniform"]
            parts_ = []
            for a_ in (0.8, 1.0):
                v_ = du[du.alpha == a_]
                if len(v_):
                    parts_.append(f"alpha {a_:g}: {100 * v_['avg_acc'].iloc[0]:.3f} there vs {summ[bk][3][a_]:.3f} here")
            alt_note = (f"best uniform is at alpha {summ[bk][2]:g} here; the same uniform merge re-evaluated in {altf}: " + "; ".join(parts_) +
                        " (the doc's 64.6 matches the regmean_lora_dev.jsonl value; evaluation differences of this size between files are common)") if bk == "lora" else \
                       (f"re-evaluation of the same uniform merge in {altf}: " + "; ".join(parts_))
        check(f"C4b-{bk}-ub", "AA_SIDE_REPORT s6 cross-bank table / FULL_REPORT s15", f"{bk}: best uniform", ub_doc, summ[bk][1], 1, alt_note)
    check("C4b-nbanks", "task brief", "replicate_banks.jsonl contains all 4 banks", 4, len(banks_in_file), None, "the file has 3 banks; the KnOTS release reference is in selection_controls_null20.jsonl + rule_sweep", text=True)

    para(f"**C4b-2. Every row of `replicate_banks.jsonl` ({len(rb)} rows): the free random selections (dev acc %, seeds {int(rb.seed.min())}-{int(rb.seed.max())})**")
    rows = []
    for s in range(20):
        row = [str(s)]
        for bk in banks_in_file:
            for a in (0.8, 1.0):
                v = rb[(rb.bank == bk) & (rb.label == "free_random_selection") & (rb.alpha == a) & (rb.seed == s)]
                assert len(v) == 1
                row.append(nm(v["pct"].iloc[0], 2))
        rows.append(row)
    tab(["seed"] + [f"{bk} a={a}" for bk in banks_in_file for a in (0.8, 1.0)], rows)
    n_nulls_shown = sum(1 for r_ in rows for c_ in r_[1:] if c_ != "")
    para("**C4b-3. The remaining rows of `replicate_banks.jsonl` (uniform, kappa_argmin, kappa_b128), with per-task accuracies (%)**")
    rows = []
    for r in rb.itertuples():
        if r.label == "free_random_selection":
            continue
        rows.append([r.bank, r.label, gen(r.alpha), nm(r.pct, 2)] + [nm(100 * r.acc[t], 1) for t in TASKS8])
    tab(["bank", "label", "alpha", "avg_acc (abs. %)"] + TASKS8, rows)
    cover(src_aa("replicate_banks.jsonl"), len(rb), n_nulls_shown + len(rows), "C4b-2 (free random selections) + C4b-3 (other rows)")

    # ---------------- C4c --------------------------------------------------------------------------------------
    gj = read_json(AA / "gentle.json", src_aa("gentle.json"))
    h3("C4c. 'Functional gentleness': is argmin-kappa just the gentlest expert per map? (`gentle.json`, dev)")
    para(f"Within-map Spearman(sigma_min(W_i), functional energy tr(Delta C Delta^T) on pooled base-model inputs) = {gj['spearman_smin_energy']:+.4f} (mean over 48 maps); "
         f"agreement of argmin-kappa with argmin-energy (gentlest) = {gj['agree_argmin_kappa_argmin_energy']:.4f}, with argmax-energy (strongest) = {gj['agree_argmin_kappa_argmax_energy']:.4f}. "
         "Hard per-map selections evaluated on dev (budget A for kappa; budget B pooled inputs for the energy).")
    rows = []
    for e in gj["eval"]:
        rows.append([e["label"], gen(e["alpha"]), nm(100 * e["avg_acc"], 2)] + [nm(100 * e["acc"][t], 1) for t in TASKS8] + [str(e["counts"])])
    tab(["label", "alpha", "avg_acc (abs. %)"] + TASKS8 + ["# maps per expert (TASKS8 order)"], rows)
    cover(src_aa("gentle.json") + " [eval]", len(gj["eval"]), len(rows), "C4c")
    cnt_c2 = [int((np.array([st[(st.param == p)].set_index("expert").loc[TASKS8, "kappa2_W"].values.argmin() for p in names]) == i).sum()) for i in range(8)]
    cnt_g = [e["counts"] for e in gj["eval"] if e["label"] == "argmin_kappa"][0]
    para(f"Consistency: argmin-kappa win counts recomputed from `lora_stats.csv` = {cnt_c2}; stored in `gentle.json` = {cnt_g}; equal: {cnt_c2 == cnt_g}. "
         f"`gentle.json` `argmin_kappa` dev accuracy {nm(100 * [e['avg_acc'] for e in gj['eval'] if e['label'] == 'argmin_kappa' and e['alpha'] == 0.8][0])} (alpha 0.8) vs `selection_controls.jsonl` "
         f"`kappa_argmin` {nm(sc5[(sc5.label == 'kappa_argmin') & (sc5.alpha == 0.8)]['pct'].iloc[0])} (both are the same configuration evaluated in two runs).")

    # ---------------- C4d --------------------------------------------------------------------------------------
    ov = read_json(AA / "oracle_vs_scores.json", src_aa("oracle_vs_scores.json"))
    md_lines = (AA / "oracle_vs_scores.md").read_text().splitlines()
    register(AA / "oracle_vs_scores.md", src_aa("oracle_vs_scores.md"), len(md_lines))
    h3("C4d. Does the LoRA oracle allocation (budget D) follow any spectral score? (`oracle_vs_scores.json`, `.md`)")
    para("Two oracles: per-layer (per-map) lambda trained on labelled dev data, `layerwise_a1_seed0` at alpha fixed to 1 and `layerwise_alpha_seed0` with alpha also learned. "
         "For each score: within-map Spearman between the oracle's lambda and the score (permutation null), between-expert correlation (8 points), and within-expert correlation (shuffle maps per expert). "
         "`dev` and `test` columns are the oracle's own absolute avg_acc (the oracle is fitted on dev, so `dev` is optimistic; `test` is the untouched split).")
    rows = []
    for k, v in ov.items():
        rows.append([k, nm(v["alpha"], 4), nm(100 * v["dev"], 2), nm(100 * v["test"], 2), nm(v["n_eff"], 3)] + [nm(v["expert_mean_lambda"][t], 3) for t in TASKS8])
    para("**C4d-1. The two oracles** (absolute avg_acc, %)")
    tab(["oracle", "alpha", "dev avg_acc (abs. %)", "test avg_acc (abs. %)", "mean n_eff of lambda"] + [f"mean lambda {t}" for t in TASKS8], rows)
    score_names = ["kappa_W", "smin_W", "fro_D", "kplus_D_0.01", "stable_rank_D", "entropy_rank_D", "kappa_A(gauge-dep)"]
    rows = []
    regen = ["| oracle | score | within-map rho (p) | between (8 pts) | within-expert (p) |", "|---|---|---|---|---|"]
    for k, v in ov.items():
        for sn in score_names:
            x = v[sn]
            rows.append([k, sn, sg(x["within_map"], 3), nm(x["p"], 3), sg(x["between"], 2), sg(x["within_exp"], 3), nm(x["p_within"], 3)])
            regen.append(f"| {k} | {sn} | {x['within_map']:+.3f} ({x['p']:.3f}) | {x['between']:+.2f} | {x['within_exp']:+.3f} ({x['p_within']:.3f}) |")
    para("**C4d-2. Correlation of each oracle with each spectral score**")
    tab(["oracle", "score", "within-map rho", "p (perm)", "between-expert r (8 pts)", "within-expert rho", "p (perm)"], rows, ["l", "l", "r", "r", "r", "r", "r"])
    same_md = regen == md_lines
    para(f"The `.md` file ({len(md_lines) - 2} data rows) is regenerated from the `.json` with the script's format string: identical = {same_md} "
         f"({len(regen)} regenerated lines vs {len(md_lines)} in the file).")
    para("Verbatim copy of `oracle_vs_scores.md`:")
    codeblock("\n".join(md_lines))
    o2 = ov["layerwise_alpha_seed0"]
    check("C4d-1", "AA_SIDE_REPORT s9", "per-layer lambda oracle on KnOTS dev (alpha learned)", 64.8, 100 * o2["dev"], 1)
    check("C4d-2", "AA_SIDE_REPORT s1/s3", "LoRA oracle test accuracy (alpha learned)", 64.1, 100 * o2["test"], 1)
    o1 = ov["layerwise_a1_seed0"]
    check_array("C4d-FR", "FULL_REPORT s12 table", "LoRA r16 layerwise oracle alpha = 1 (dev, test) and oracle + alpha (dev, test)", [62.3, 61.7, 64.8, 64.1],
                [100 * o1["dev"], 100 * o1["test"], 100 * o2["dev"], 100 * o2["test"]], 1)
    check("C4d-FR-uni", "FULL_REPORT s12 table", "LoRA r16 uniform (best alpha): dev / test = 52.3 / 51.2 (dev shown)", 52.3, ub_uniform_dev(), 1)
    check("C4d-3", "AA_SIDE_REPORT s1", "LoRA oracle headroom over best uniform on dev (+12 pt)", 12, 100 * o2["dev"] - ub_uniform_dev(), 0,
          "%.2f - %.2f = %.2f; the doc's +12 is a truncation (it rounds to +13), counted OK within 1 pt" % (100 * o2["dev"], ub_uniform_dev(), 100 * o2["dev"] - ub_uniform_dev()), tol=1.0)

    # ---------------- C4e --------------------------------------------------------------------------------------
    sw = read_jsonl(AA / "single_map_swap.jsonl", src_aa("single_map_swap.jsonl"))
    sj = read_json(AA / "single_map_swap_summary.json", src_aa("single_map_swap_summary.json"))
    h3("C4e. Per-map marginal value of handing one map to one expert (dev, uniform base, alpha = 1; `single_map_swap*`)")
    para(f"Baseline = uniform lambda on all {len(names)} maps at alpha = {sj['alpha']:g}. For each map p and expert i, only map p is switched to one-hot(i) and the dev `avg_acc` (absolute, mean over tasks) is recorded: "
         f"{len(names)} x {len(TASKS8)} = {len(names) * len(TASKS8)} swaps + 1 baseline = {len(sw)} rows. `gain` = swap accuracy - baseline accuracy, in percentage points.")
    keyed = {}
    dup = 0
    for r in sw:
        k = (r["param"], r["expert"])
        if k in keyed:
            dup += 1
        keyed.setdefault(k, r)
    base = keyed[("__base__", "uniform")]
    V = np.array([[keyed[(n, t)]["avg_acc"] for t in TASKS8] for n in names])
    gain = V - base["avg_acc"]
    K = np.array([[st[(st.param == n) & (st.expert == t)]["kappa2_W"].iloc[0] for t in TASKS8] for n in names])
    rho_k = float(np.mean([spearmanr(-K[p], gain[p]).statistic for p in range(len(names))]))
    ranks = np.array([rankdata(-gain[p])[K[p].argmin()] for p in range(len(names))])
    rec = {"alpha": base["alpha"], "base_avg_acc": base["avg_acc"], "mean_within_map_spearman(-kappa, swap_gain)": rho_k,
           "argmin_kappa_mean_rank (1=best, 8=worst, random 4.5)": float(ranks.mean()), "frac_argmin_kappa_is_worst": float(np.mean(ranks == 8)),
           "frac_argmin_kappa_is_best": float(np.mean(ranks == 1)), "mean_gain_of_argmin_kappa": float(np.mean(gain[np.arange(len(names)), K.argmin(1)])),
           "mean_gain_random_choice": float(gain.mean()), "mean_gain_best_choice": float(gain.max(1).mean()), "mean_gain_worst_choice": float(gain.min(1).mean())}
    rows = []
    mx = 0.0
    for k_, v in sj.items():
        d = abs(rec[k_] - v)
        mx = max(mx, d)
        if k_.startswith("mean_gain"):
            shown, shown2 = nm(100 * v, 4) + " pt", nm(100 * rec[k_], 4) + " pt"
        elif k_ == "base_avg_acc":
            shown, shown2 = nm(100 * v, 4) + " %", nm(100 * rec[k_], 4) + " %"
        else:
            shown, shown2 = nm(v, 6), nm(rec[k_], 6)
        rows.append([k_, shown, shown2, f"{d:.1e}"])
    para(f"**C4e-1. Stored summary vs recomputation from the 385 raw rows** (duplicates in the raw file: {dup}; max abs difference over the 10 numbers: {mx:.1e})")
    tab(["quantity", "stored in single_map_swap_summary.json", "recomputed here", "abs diff"], rows, ["l", "r", "r", "r"])
    check("C4e-1", "AA_SIDE_REPORT s6", "argmin-kappa mean rank among 8 swaps", 4.27, rec["argmin_kappa_mean_rank (1=best, 8=worst, random 4.5)"], 2)
    check("C4e-2", "AA_SIDE_REPORT s6", "argmin-kappa is the best swap in 10% of maps", 0.10, rec["frac_argmin_kappa_is_best"], 2)
    check("C4e-3", "AA_SIDE_REPORT s6", "argmin-kappa is the worst swap in 10% of maps", 0.10, rec["frac_argmin_kappa_is_worst"], 2)
    check("C4e-4", "AA_SIDE_REPORT s6", "within-map rho(-kappa, gain)", -0.08, rho_k, 2)

    # distribution of ranks
    rows = [[str(r_), int((ranks == r_).sum()), nm(100 * np.mean(ranks == r_), 1)] for r_ in range(1, 9)]
    ties = int((ranks != np.round(ranks)).sum())
    para(f"**C4e-2. Where argmin-kappa ranks among the 8 swaps of its map** (1 = best; ties give fractional ranks: {ties} maps)")
    tab(["rank of the argmin-kappa swap", "# maps", "% of maps"], rows)

    # gains by expert
    best_idx = gain.argmax(1)
    worst_idx = gain.argmin(1)
    kidx = K.argmin(1)
    rows = []
    for i, t in enumerate(TASKS8):
        rows.append([t, nm(100 * gain[:, i].mean(), 3), nm(100 * np.median(gain[:, i]), 3), nm(100 * gain[:, i].min(), 2), nm(100 * gain[:, i].max(), 2),
                     nm(100 * np.mean(gain[:, i] > 0), 1), int((best_idx == i).sum()), int((worst_idx == i).sum()), int((kidx == i).sum())])
    para("**C4e-3. Swap gain by receiving expert** (pt of mean dev accuracy relative to the uniform baseline)")
    tab(["expert", "mean gain", "median gain", "min gain", "max gain", "% of maps with gain > 0", "# maps where it is the best swap", "# maps where it is the worst swap", "# maps where it is argmin-kappa"], rows)
    mods = ["q_proj", "k_proj", "v_proj", "out_proj"]
    modof = {p: [m for m in mods if m in p][0] for p in names}
    rows = []
    for m in mods:
        idx = [i for i, p in enumerate(names) if modof[p] == m]
        g = gain[idx]
        rows.append([m, len(idx), nm(100 * g.mean(), 3), nm(100 * g.max(1).mean(), 3), nm(100 * g.min(1).mean(), 3), nm(100 * g[np.arange(len(idx)), K[idx].argmin(1)].mean(), 3),
                     nm(ranks[idx].mean(), 2)])
    rows.append(["ALL", len(names), nm(100 * gain.mean(), 3), nm(100 * gain.max(1).mean(), 3), nm(100 * gain.min(1).mean(), 3), nm(100 * rec["mean_gain_of_argmin_kappa"], 3), nm(ranks.mean(), 2)])
    para("**C4e-4. Swap gain by map type** (pt)")
    tab(["module", "# maps", "mean gain (all 8 experts)", "mean best-swap gain", "mean worst-swap gain", "mean gain of argmin-kappa swap", "mean rank of argmin-kappa"], rows)
    flat = [(gain[p, i], names[p], TASKS8[i]) for p in range(len(names)) for i in range(8)]
    flat.sort(key=lambda x: -x[0])
    rows = [[f"{100 * g_:+.3f}", p_.replace("encoder.layers.", "L").replace(".self_attn.", ".").replace(".weight", ""), e_] for g_, p_, e_ in flat[:10]]
    rows += [["...", "...", "..."]]
    rows += [[f"{100 * g_:+.3f}", p_.replace("encoder.layers.", "L").replace(".self_attn.", ".").replace(".weight", ""), e_] for g_, p_, e_ in flat[-10:]]
    para("**C4e-5. The 10 best and 10 worst single-map swaps** (gain in pt). Full 385-row data: `single_map_swap.jsonl`.")
    tab(["gain (pt)", "map", "expert"], rows, ["r", "l", "l"])
    cover(src_aa("single_map_swap.jsonl"), len(sw), None, "C4e-1..C4e-5", "summarised (recomputed summary, rank histogram, gains by expert / map type, 10 best + 10 worst swaps); the 385 raw rows are NOT reproduced")

    # ---------------- C4f --------------------------------------------------------------------------------------
    q9 = lora_df(AA / "q90_transfer.jsonl", src_aa("q90_transfer.jsonl"))
    q9["pct"] = 100 * q9["avg_acc"]
    h3("C4f. Transfer of master's dev-best full-FT rule `logkappa_q90_rev` to the four LoRA banks (dev, budget A; `q90_transfer.jsonl`)")
    para(f"Rule: per adapted map lambda = softmax(+beta * log(s_1 / s_ceil(0.9 q)) of W0 + B_i A_i) (higher bulk anisotropy -> more weight), b in {sset(sorted(set(q9.b.dropna())))}, alpha in {sset(sorted(set(q9.alpha)))}; "
         f"`shuffled` = the same scores permuted across experts within each map (seeds {int(q9.seed.min())}-{int(q9.seed.max())}) at the same (b, alpha). The file has no per-task accuracies (only `avg_acc`, absolute, percent here).")
    rows = []
    for r in q9.itertuples():
        rows.append([r.bank, gen(r.alpha), r.label, "-" if isnan(r.b) else gen(r.b), "-" if isnan(r.seed) else str(int(r.seed)), nm(r.pct, 2)])
    para(f"**C4f-1. Every row ({len(q9)})**")
    tab(["bank", "alpha", "label", "b", "seed", "dev avg_acc (abs. %)"], rows)
    cover(src_aa("q90_transfer.jsonl"), len(q9), len(rows), "C4f-1")
    rows = []
    diffs_knots, diffs_all = [], []
    diffs_by_bank = defaultdict(list)
    diffs_abkey = {}
    for bk in bank_order:
        for a in (0.8, 1.0):
            uu = float(q9[(q9.bank == bk) & (q9.alpha == a) & (q9.label == "uniform")]["pct"].iloc[0])
            for b in (1.0, 4.0):
                rr = float(q9[(q9.bank == bk) & (q9.alpha == a) & (q9.label == "logkappa_q90_rev") & (q9.b == b)]["pct"].iloc[0])
                sh = q9[(q9.bank == bk) & (q9.alpha == a) & (q9.label == "shuffled") & (q9.b == b)]["pct"].values
                rows.append([bk, gen(a), gen(b), nm(uu), nm(rr), sg(rr - uu), nm(sh.mean()), f"{nm(sh.min())} / {nm(sh.max())}", sg(rr - sh.mean()), int((sh < rr).sum()), len(sh)])
                diffs_all.append(rr - uu)
                diffs_by_bank[bk].append(rr - uu)
                diffs_abkey[(bk, a, b)] = rr - uu
                if bk != "lora":
                    diffs_knots.append(rr - uu)
    para("**C4f-2. Summary: rule minus uniform, and rule against the 5 matched shuffles**")
    tab(["bank", "alpha", "b", "uniform", "rule", "rule - uniform (pt)", "shuffled mean", "shuffled min / max", "rule - shuffled mean (pt)", "# shuffles below the rule", "# shuffles"], rows)
    para(f"Range of (rule - uniform) over the 3 KnOTS-type banks: {min(diffs_knots):+.2f} to {max(diffs_knots):+.2f} pt; over all 4 banks including vit20 (`lora`): {min(diffs_all):+.2f} to {max(diffs_all):+.2f} pt.")
    check("C4f-1", "AA_SIDE_REPORT s8", "LoRA transfer of logkappa_q90_rev: smallest loss (doc: -4 to -18 pt), 3 KnOTS-type banks", -4, max(diffs_knots), 0)
    check("C4f-2", "AA_SIDE_REPORT s8", "LoRA transfer of logkappa_q90_rev: largest loss (doc: -4 to -18 pt), 3 KnOTS-type banks", -18, min(diffs_knots), 0)
    check("C4f-3", "AA_SIDE_REPORT s8", "LoRA transfer of logkappa_q90_rev, vit20 (`lora`) bank only: best case (doc range: -4 to -18 pt)", -4, max(diffs_by_bank["lora"]), 0,
          "vit20 rule - uniform ranges over %+.2f ... %+.2f pt, outside the quoted -4 to -18" % (min(diffs_by_bank["lora"]), max(diffs_by_bank["lora"])))

    FR = "FULL_REPORT s15"
    check_range("C4f-FR-knots", FR, "transfer of logkappa_q90_rev, KnOTS release: '-4.1 to -17.6'", "-17.6 to -4.1", -17.6, -4.1, diffs_by_bank["knots_release"], 1)
    check_range("C4f-FR-c5a", FR, "transfer, lora_c5a: '-9.3 to -16.7'", "-16.7 to -9.3", -16.7, -9.3, diffs_by_bank["lora_c5a"], 1)
    check_range("C4f-FR-seed1", FR, "transfer, lora_c5a_seed1: '-8.7 to -12.5'", "-12.5 to -8.7", -12.5, -8.7, diffs_by_bank["lora_c5a_seed1"], 1)
    check_array("C4f-FR-vit20-b1", FR, "transfer, vit20 at b = 1: '+0.3 / -0.3' (alpha 0.8 / 1.0)", [0.3, -0.3], [diffs_abkey[("lora", 0.8, 1.0)], diffs_abkey[("lora", 1.0, 1.0)]], 1)
    check_array("C4f-FR-vit20-b4", FR, "transfer, vit20 at b = 4: '-1.7 / -2.9' (alpha 0.8 / 1.0)", [-1.7, -2.9], [diffs_abkey[("lora", 0.8, 4.0)], diffs_abkey[("lora", 1.0, 4.0)]], 1)

    # ---------------- C4g --------------------------------------------------------------------------------------
    tf = lora_df(AA / "test_freeze.jsonl", src_aa("test_freeze.jsonl"))
    tf["pct"] = 100 * tf["avg_acc"]
    h3(f"C4g. Frozen configurations evaluated once on the untouched TEST split (KnOTS release; `test_freeze.jsonl`, {len(tf)} rows)")
    para("Configurations were frozen from the dev sweep (C3): uniform at its dev-best alpha 0.8; `kappa_b128` and `kappa_argmin` at alpha 1.0; `shufK:kappa_b128` = the shuffled-score null at the literal rule's exact configuration; "
         "`single:eurosat` = best single expert on dev (alpha 0.6); `metagpt` alpha 1.0; `*_rank16` = merged update truncated to rank 16 per map (fixed final-rank budget). Split = test (80% of the official test set); metric = absolute `avg_acc` in percent.")
    rows = []
    for r in tf.itertuples():
        rows.append([r.label, gen(r.alpha), "-" if r.compress_rank is None or isnan(r.compress_rank) else str(int(r.compress_rank)), r.split, nm(r.pct, 2)] + [nm(100 * r.acc[t], 2) for t in TASKS8])
    para("**C4g-1. Every row (test avg_acc and per-task accuracy, absolute %)**")
    tab(["label", "alpha", "compress_rank", "split", "avg_acc (abs. %)"] + TASKS8, rows)
    tfd = tf.set_index("label")["pct"]
    rows = [[l, nm(tfd[l], 2), sg(tfd[l] - tfd["uniform"], 2)] for l in tfd.index if "rank16" not in l] + \
           [[l, nm(tfd[l], 2), sg(tfd[l] - tfd["uniform_rank16"], 2) + " (vs uniform_rank16)"] for l in tfd.index if "rank16" in l]
    para("**C4g-2. Each frozen config minus uniform (test)**")
    tab(["label", "test avg_acc (abs. %)", "minus uniform (pt)"], rows)
    cover(src_aa("test_freeze.jsonl"), len(tf), len(tf), "C4g-1")
    check("C4g-1", "AA_SIDE_REPORT s6", "frozen TEST kappa (b128, alpha 1.0)", 53.3, tfd["kappa_b128"], 1)
    check("C4g-2", "AA_SIDE_REPORT s6", "frozen TEST uniform", 51.2, tfd["uniform"], 1)
    check("C4g-3", "AA_SIDE_REPORT s6", "frozen TEST kappa rank-16 compressed", 53.2, tfd["kappa_b128_rank16"], 1)
    check("C4g-4", "AA_SIDE_REPORT s6", "frozen TEST uniform rank-16 compressed", 50.2, tfd["uniform_rank16"], 1)
    check("C4g-5", "AA_SIDE_REPORT s6", "frozen TEST shuffle 0", 46.8, tfd["shuf0:kappa_b128"], 1)
    check("C4g-6", "AA_SIDE_REPORT s6", "frozen TEST shuffle 1", 51.3, tfd["shuf1:kappa_b128"], 1)
    check("C4g-7", "AA_SIDE_REPORT s6", "frozen TEST shuffle 2", 50.2, tfd["shuf2:kappa_b128"], 1)
    check("C4g-8", "AA_SIDE_REPORT s6", "frozen TEST best single expert (eurosat)", 54.4, tfd["single:eurosat"], 1)


def ub_uniform_dev() -> float:
    df = load_rule_sweep().drop_duplicates("chash")
    return float(100 * df[df.label == "uniform"]["avg_acc"].max())


# ---- C5 -------------------------------------------------------------------------------------------------------------
MORDER = {"shrink": 0, "kcap_mean": 1, "kcap_zero": 2}
KEYCOLS = ["method", "ridge_to", "shrink", "kappa_target"]


def knone(x):
    return None if isnan(x) else float(x)


def cfg_groups(df: pd.DataFrame) -> "OrderedDict":
    """{(method, ridge_to, shrink, kappa_target): {alpha: row}} in a sensible order."""
    d = df.copy()
    d["_m"] = d["method"].map(MORDER)
    d["_s"] = -d["shrink"]
    d["_k"] = d["kappa_target"].fillna(-1.0)
    d = d.sort_values(["_m", "_s", "_k", "alpha"], kind="mergesort")
    out: OrderedDict = OrderedDict()
    for r in d.to_dict("records"):
        key = (r["method"], r["ridge_to"], float(r["shrink"]), knone(r["kappa_target"]))
        out.setdefault(key, {})[float(r["alpha"])] = r
    return out


def kcell(x) -> str:
    return "-" if x is None else gen(x)


def master_dev_tables(df: pd.DataFrame, tag: str, label: str) -> dict:
    """C_dev / B_dev: pivots + long form. Returns a few summary numbers for cross-checks."""
    cfgs = cfg_groups(df)
    alphas = sorted(df["alpha"].unique())
    nimg = sorted(set(df["n_images"].map(lambda s: json.loads(s)["sun397"] if isinstance(s, str) else s)))
    para(f"**{tag}-1. Dev mean_norm (%) for every config x alpha** ({len(cfgs)} configs x {len(alphas)} alphas = {len(df)} rows; images per task used for the Gram: {nimg}). "
         "`method`: `shrink` = RegMean with off-diagonal shrinkage a (`shrink`; a = 1 plain RegMean, a = 0 diagonal Gram); `kcap_mean` / `kcap_zero` = kappa cap on G (a = 1) with ridge target R = mean of the updates / zero.")
    rows = []
    for key, byA in cfgs.items():
        best_a = max(byA, key=lambda a: byA[a]["norm"])
        rows.append([key[0], key[1], gen(key[2]), kcell(key[3])] + [nm(byA[a]["norm"]) if a in byA else "-" for a in alphas] + [gen(best_a), nm(byA[best_a]["norm"])])
    tab(["method", "ridge_to", "shrink", "kappa_target"] + [f"alpha={a:g}" for a in alphas] + ["best alpha", "best norm"], rows)
    para(f"**{tag}-2. Dev mean_acc (%, absolute) for every config x alpha**")
    rows = []
    for key, byA in cfgs.items():
        best_a = max(byA, key=lambda a: byA[a]["acc"])
        rows.append([key[0], key[1], gen(key[2]), kcell(key[3])] + [nm(byA[a]["acc"]) if a in byA else "-" for a in alphas] + [gen(best_a), nm(byA[best_a]["acc"])])
    tab(["method", "ridge_to", "shrink", "kappa_target"] + [f"alpha={a:g}" for a in alphas] + ["best alpha", "best acc"], rows)
    n_null = int(df["kappa_before_median"].isna().sum())
    consts = {c: sorted(set(df[c].astype(str))) for c in ("budget", "gram_tag", "split") if c in df.columns}
    commits = df["commit"].value_counts().to_dict()
    para(f"**{tag}-3. Every row (long form, {len(df)} rows)** with the Gram-conditioning statistics recorded by the run "
         "(`kappa_before_median` = median over maps of kappa2(G) of the (shrunk) summed Gram; `kappa_after_median` / `kappa_after_max` = median / max over maps of kappa2(G + rho I) actually solved; "
         f"`rho_median` = median ridge; '-' = null in the file, {n_null} of {len(df)} rows, because those runs did not log kappa). Other cfg keys are constant in this file: {consts}; "
         f"rows by git commit stamp of the run: {commits}.")
    rows = []
    for key, byA in cfgs.items():
        for a in alphas:
            r = byA.get(a)
            if r is None:
                continue
            rows.append([key[0], key[1], gen(key[2]), kcell(key[3]), gen(a), nm(r["norm"], 3), nm(r["acc"], 3), nm(r["min_norm"], 2),
                         sci(r.get("kappa_before_median", NAN), 3), sci(r.get("kappa_after_median", NAN), 3), sci(r.get("kappa_after_max", NAN), 3), sci(r.get("rho_median", NAN), 3)])
    tab(["method", "ridge_to", "shrink", "kappa_target", "alpha", "mean_norm", "mean_acc", "min_norm", "kappa_before_median", "kappa_after_median", "kappa_after_max", "rho_median"], rows)
    cover(src_ma(label + ".jsonl"), len(df), len(rows), f"{tag}-3 (long form); also pivoted in {tag}-1/-2")
    top = df.sort_values("norm", ascending=False).head(10)
    para(f"**{tag}-4. Top 10 dev configs by mean_norm**")
    rows = [[r.method, r.ridge_to, gen(r.shrink), kcell(knone(r.kappa_target)), gen(r.alpha), nm(r.norm, 3), nm(r.acc, 3)] for r in top.itertuples()]
    tab(["method", "ridge_to", "shrink", "kappa_target", "alpha", "mean_norm", "mean_acc"], rows)
    return {"cfgs": cfgs}


def master_test_table(df: pd.DataFrame, extra_cols: list[str], extra_hdr: list[str], tag: str) -> None:
    rows = []
    for r in df.to_dict("records"):
        base = []
        for c in ("method", "ridge_to", "shrink", "kappa_target", "alpha", "n", "seed"):
            if c in r:
                v = r[c]
                if c in ("method", "ridge_to"):
                    base.append("-" if isnan(v) else str(v))
                elif c == "kappa_target":
                    base.append(kcell(knone(v)))
                elif c in ("n", "seed"):
                    base.append(str(int(v)))
                else:
                    base.append(gen(v))
        rows.append(base + [nm(r["norm"], 2), nm(r["acc"], 2), nm(r["min_norm"], 2)] +
                    [sci(r.get(c, NAN), 3) for c in extra_cols] + [nm(r[f"acc_{t}"], 1) for t in TASKS8])
    hdr = [c for c in ("method", "ridge_to", "shrink", "kappa_target", "alpha", "n", "seed") if c in df.columns]
    tab(hdr + ["mean_norm", "mean_acc", "min_norm"] + extra_hdr + [f"acc {t}" for t in TASKS8], rows)


def arr_for(df, cell, param, n, split_col="seed"):
    """Array of norm values (percent) for (cell, param, n) sorted by seed."""
    pcol = "shrink" if cell == "fixed_shrink" else "kappa_t"
    sub = df[(df["cell"] == cell) & (np.isclose(df[pcol].astype(float), param)) & (df["n"] == n)].sort_values("seed")
    return sub["norm"].values


def sec_C5():
    header_block("C5", "Full fine-tune RegMean (8 FusionBench CLIP ViT-B/32 full-FT experts), master",
                 "Varied: the RegMean regulariser (off-diagonal shrinkage a, kappa cap kappa_t on G with ridge target mean/zero, fixed relative ridge), the update scale alpha, the Gram budget "
                 "(C = task-separated: expert i on its own task's unlabeled train images; B = every expert on the same pooled images, no task ids), the number n of calibration images per task "
                 "(16/64/512) and the calibration subset seed; split = dev (20% of the official test set) for model selection and test (the other 80%) for frozen configs; "
                 "metric = `mean_norm` = mean over the 8 tasks of accuracy / accuracy of that task's own expert (NORMALIZED accuracy; expert accuracies from "
                 "master:results/experts/expert_accuracy.json), `mean_acc` = unweighted mean ABSOLUTE accuracy, `min_norm` = worst task's normalized accuracy; units = percent (x100). "
                 "Verification that the stored metrics are what they say: see C5i.",
                 [src_ma(f) for f in ["regmean_C_dev.jsonl", "regmean_B_dev.jsonl", "instability_C_dev.jsonl", "regmean_C_test.jsonl", "regmean_B_test.jsonl", "instability_C_test.jsonl",
                                      "twobytwo_C_dev.jsonl", "twobytwo_C_test.jsonl", "twobytwo_summary.json", "strength_sweep_C_dev_n16s0.jsonl", "strength_sweep_C_dev_n16s1.jsonl",
                                      "strength_sweep_C_dev_n16s2.jsonl", "strength_sweep_C_dev_n512s0.jsonl", "strength_sweep_C_test_n16s0.jsonl", "strength_sweep_C_test_n16s1.jsonl",
                                      "strength_sweep_C_test_n16s2.jsonl", "strength_sweep_C_test_n512s0.jsonl"]] +
                 ["master:src/condmerge/regmean.py, master:experiments/regmean_sweep.py, regmean_instability.py, regmean_2x2.py, analyze_regmean_2x2.py"])
    para("Legend: **RegMean** per linear map solves D (G + rho I) = sum_i D_i G~_i + rho R with G = sum_i G~_i, G_i = E_task-i[x x^T] (input second moment of expert i), "
         "off-diagonal shrinkage G~_i = a G_i + (1 - a) diag(G_i). `shrink` rows: rho = 0. **kappa cap** (`kcap_mean`, `kcap_zero`; a = 1): rho = the smallest ridge with kappa2(G + rho I) <= kappa_t "
         "and R = mean of the updates (`mean`) or 0 (`zero`). Merged weights W0 + alpha * D (alpha = 1 is standard RegMean); non-linear parameters get the uniform average. "
         "2x2 / sweep cells: `kcap_ridge` (kappa-cap ridge toward the mean), `kcap_shrink` (the largest a with kappa2(a G + (1 - a) diag G) <= kappa_t), `fixed_shrink` (a fixed), "
         "`rel_ridge` (fixed relative ridge rho = c tr(G)/d, c = r_med / kappa_t with r_med the median over maps of lambda_max/(tr/d), `c_rel` = that c). "
         "`rel_rho_median` = median over maps of rho / (tr(G)/d) (the effective relative ridge); `shrink_median` / `shrink_min` = median / min over maps of the shrinkage a used.")

    cdev = master_df(MA / "regmean_C_dev.jsonl", src_ma("regmean_C_dev.jsonl"))
    bdev = master_df(MA / "regmean_B_dev.jsonl", src_ma("regmean_B_dev.jsonl"))
    idev = master_df(MA / "instability_C_dev.jsonl", src_ma("instability_C_dev.jsonl"), "dev")
    ctest = master_df(MA / "regmean_C_test.jsonl", src_ma("regmean_C_test.jsonl"))
    btest = master_df(MA / "regmean_B_test.jsonl", src_ma("regmean_B_test.jsonl"))
    itest = master_df(MA / "instability_C_test.jsonl", src_ma("instability_C_test.jsonl"), "test")
    tdev = master_df(MA / "twobytwo_C_dev.jsonl", src_ma("twobytwo_C_dev.jsonl"), "dev")
    ttest = master_df(MA / "twobytwo_C_test.jsonl", src_ma("twobytwo_C_test.jsonl"), "test")
    sd_files = sorted(glob.glob(str(MA / "strength_sweep_C_dev_*.jsonl")))
    st_files = sorted(glob.glob(str(MA / "strength_sweep_C_test_*.jsonl")))
    sdev = pd.concat([master_df(Path(f), src_ma(Path(f).name), "dev").assign(_file=Path(f).name) for f in sd_files], ignore_index=True)
    stest = pd.concat([master_df(Path(f), src_ma(Path(f).name), "test").assign(_file=Path(f).name) for f in st_files], ignore_index=True)
    exp_sets = ["n16s0", "n16s1", "n16s2", "n512s0"]
    for kind, files, df_ in (("dev", sd_files, sdev), ("test", st_files, stest)):
        have = [re.search(r"_(n\d+s\d+)\.jsonl", Path(f).name).group(1) for f in files]
        rows_per = df_.groupby("_file").size().to_dict()
        by_cell = df_.groupby("_file")["cell"].apply(lambda x: dict(x.value_counts())).to_dict() if False else {f_: dict(g["cell"].value_counts()) for f_, g in df_.groupby("_file")}
        para(f"Completeness of `strength_sweep_C_{kind}_*`: files present {have} (expected {exp_sets}); rows per file {rows_per}; "
             f"complete: {sorted(have) == sorted(exp_sets) and len(set(rows_per.values())) == 1 and (list(rows_per.values())[0] == (19 if kind == 'dev' else 3))}. "
             f"Rows per arm in the first file: { {k_: int(v_) for k_, v_ in list(by_cell.values())[0].items()} }.")

    # ------------------------------------------------------------------ C5a
    h3("C5a. regmean_C_dev.jsonl: every config, budget C (n = 512 images per task, all 8 tasks), dev")
    resC = master_dev_tables(cdev, "C5a", "regmean_C_dev")
    h3("C5b. regmean_B_dev.jsonl: every config, budget B (pooled images, no task ids), dev")
    resB = master_dev_tables(bdev, "C5b", "regmean_B_dev")

    def best_of(df, **flt):
        d = df
        for k, v in flt.items():
            d = d[d[k] == v] if not isinstance(v, tuple) else d[d[k].isin(v)]
        return d.sort_values("norm", ascending=False).iloc[0]

    # families (budget C dev)
    fam_rows = []
    fams = [("plain RegMean (shrink a = 1, alpha = 1)", cdev[(cdev.method == "shrink") & (cdev.shrink == 1.0) & (cdev.alpha == 1.0)]),
            ("best shrink config (any a, alpha)", cdev[cdev.method == "shrink"]),
            ("best kcap_mean config (any kappa_t, alpha)", cdev[cdev.method == "kcap_mean"]),
            ("best kcap_zero config (any kappa_t, alpha)", cdev[cdev.method == "kcap_zero"]),
            ("best of all configs", cdev)]
    for name, d in fams:
        r = d.sort_values("norm", ascending=False).iloc[0]
        fam_rows.append([name, r["method"], r["ridge_to"], gen(r["shrink"]), kcell(knone(r["kappa_target"])), gen(r["alpha"]), nm(r["norm"], 3), nm(r["acc"], 3)])
    h3("C5c. Dev-best of each family (budget C), and the dev value of every config that was evaluated on test")
    para(f"**C5c-1. Dev-best of each family (budget C, {len(cdev)} dev rows)**")
    tab(["family", "method", "ridge_to", "shrink", "kappa_target", "alpha", "dev mean_norm", "dev mean_acc"], fam_rows)

    def dev_vs_test(dev_df, test_df, title):
        rows = []
        for r in test_df.to_dict("records"):
            m = dev_df[(dev_df.method == r["method"]) & np.isclose(dev_df.shrink.astype(float), float(r["shrink"])) & (dev_df.alpha == r["alpha"]) & (dev_df.ridge_to == r["ridge_to"])]
            if isnan(r["kappa_target"]):
                m = m[m.kappa_target.isna()]
            else:
                m = m[np.isclose(m.kappa_target.astype(float), float(r["kappa_target"]))]
            assert len(m) == 1
            m = m.iloc[0]
            rank = int((dev_df["norm"] > m["norm"]).sum()) + 1
            rows.append([r["method"], r["ridge_to"], gen(r["shrink"]), kcell(knone(r["kappa_target"])), gen(r["alpha"]), nm(m["norm"], 2), str(rank), nm(r["norm"], 2), sg(r["norm"] - m["norm"], 2),
                         nm(m["acc"], 2), nm(r["acc"], 2)])
        para(title)
        tab(["method", "ridge_to", "shrink", "kappa_target", "alpha", "dev mean_norm", "dev rank of 95", "TEST mean_norm", "test - dev (pt)", "dev mean_acc", "TEST mean_acc"], rows)
    dev_vs_test(cdev, ctest, f"**C5c-2. Dev vs test for the {len(ctest)} frozen budget-C configs of `regmean_C_test.jsonl`**")
    dev_vs_test(bdev, btest, f"**C5c-3. Dev vs test for the {len(btest)} frozen budget-B configs of `regmean_B_test.jsonl`**")

    # ------------------------------------------------------------------ C5d instability
    h3("C5d. instability_C_dev.jsonl: noisy Grams (n images per task, subset seeds), every (n, seed, method, alpha), dev")
    para("Budget C, n images per task in {16, 64, 512} (n = 512 is one set, seed 0 = all calibration images; n = 16 and 64 use 3 random subsets, seeds 0/1/2), alpha in {1.0, 1.4}. "
         "`method` = `shrink` (a = `shrink`; 1.0 = plain RegMean, 0.0 = diagonal Gram) or `kcap_mean` (kappa cap kappa_t = `kappa_target`, ridge toward the mean). "
         "In this dev file the kappa statistics are logged only on the `kcap_mean` rows (kappa(G) of the plain summed Gram, before the cap; after the cap = kappa_t unless kappa(G) < kappa_t); the `shrink` rows have null kappa fields "
         "(the test file `instability_C_test.jsonl` logs them on every row, see C5e-3).")
    para(f"**C5d-1. Every row ({len(idev)} rows)**")
    d = idev.sort_values(["n", "seed", "method", "shrink", "kappa_target", "alpha"], ascending=[True, True, True, False, True, True], kind="mergesort")
    rows = [[str(int(r.n)), str(int(r.seed)), r.method, gen(r.shrink), kcell(knone(r.kappa_target)), gen(r.alpha), nm(r.norm, 3), nm(r.acc, 3), nm(r.min_norm, 2),
             sci(r.kappa_before_median, 3), sci(r.kappa_before_max, 3), sci(r.kappa_after_median, 3)] for r in d.itertuples()]
    tab(["n", "seed", "method", "shrink", "kappa_target", "alpha", "mean_norm", "mean_acc", "min_norm", "kappa_before_median", "kappa_before_max", "kappa_after_median"], rows)
    cover(src_ma("instability_C_dev.jsonl"), len(idev), len(rows), "C5d-1")

    para("**C5d-2. Per-method dev mean_norm (%) by n at alpha = 1.0 and alpha = 1.4: three subset seeds, their mean, and min-max** (n = 512 is a single set)")
    rows = []
    inst_mean = {}
    meths = [("shrink", 1.0, None), ("shrink", 0.95, None), ("shrink", 0.9, None), ("shrink", 0.5, None), ("shrink", 0.0, None)] + [("kcap_mean", 1.0, k) for k in (1e2, 1e3, 1e4, 1e5, 1e6)]
    for al in (1.0, 1.4):
        for (m, sh, kt) in meths:
            row = [gen(al), m, gen(sh), kcell(kt)]
            for n in (16, 64, 512):
                sub = idev[(idev.method == m) & (np.isclose(idev.shrink, sh)) & (idev.alpha == al) & (idev.n == n)]
                if kt is None:
                    sub = sub[sub.kappa_target.isna()]
                else:
                    sub = sub[np.isclose(sub.kappa_target.fillna(-1), kt)]
                sub = sub.sort_values("seed")
                v = sub["norm"].values
                inst_mean[(al, m, sh, kt, n)] = float(v.mean())
                inst_mean[(al, m, sh, kt, n, "vals")] = v
                if n == 512:
                    row += [nm(v[0], 2)]
                else:
                    row += ["/".join(nm(x, 2) for x in v), nm(v.mean(), 2), f"{nm(v.min(), 1)}-{nm(v.max(), 1)}"]
            rows.append(row)
    tab(["alpha", "method", "shrink", "kappa_target", "n=16 seeds 0/1/2", "n=16 mean", "n=16 min-max", "n=64 seeds 0/1/2", "n=64 mean", "n=64 min-max", "n=512"], rows)

    para("**C5d-3. kappa2(G) of the plain summed Gram (logged on the `kcap_mean` rows) for each calibration set; kappa after the cap = kappa_t**")
    kk = idev[(idev.method == "kcap_mean") & idev.kappa_before_median.notna()].groupby(["n", "seed"]).agg(med=("kappa_before_median", "first"), mx=("kappa_before_max", "first"),
                                                                                                            nmed=("kappa_before_median", "nunique")).reset_index()
    rows = [[str(int(r.n)), str(int(r.seed)), sci(r.med, 3), sci(r.mx, 3)] for r in kk.itertuples()]
    tab(["n per task", "subset seed", "median over maps of kappa2(G)", "max over maps of kappa2(G)"], rows)
    k16 = kk[kk.n == 16]
    para(f"n = 16: median kappa2(G) over the 3 subsets = {sci(k16.med.mean(), 3)} (mean of the three medians; individual {', '.join(sci(x, 3) for x in k16.med)}); "
         f"max kappa2(G) = {sci(k16.mx.max(), 3)} (largest of the three maxima; individual {', '.join(sci(x, 3) for x in k16.mx)}). "
         "The very large maxima come from near-singular 3072-dimensional fc2 / patch inputs per the report.")

    para("**C5d-4. kappa cap (kappa_t = 1e5, ridge toward the mean) minus fixed shrinkage a = 0.95, dev, alpha = 1.0: per seed and mean (pt of mean_norm)**")
    rows = []
    edge_dev = {}
    for n in (16, 64, 512):
        kc = inst_mean[(1.0, "kcap_mean", 1.0, 1e5, n, "vals")]
        sh = inst_mean[(1.0, "shrink", 0.95, None, n, "vals")]
        pl = inst_mean[(1.0, "shrink", 1.0, None, n, "vals")]
        dif = kc - sh
        edge_dev[n] = float(dif.mean())
        rows.append([str(n), nm(kc.mean(), 2), nm(sh.mean(), 2), nm(pl.mean(), 2), "/".join(sg(x, 2) for x in dif), sg(dif.mean(), 2), f"{int((dif > 0).sum())}/{len(dif)}"])
    tab(["n per task", "kappa cap 1e5 (mean over seeds)", "shrink 0.95", "plain (shrink 1.0)", "cap - shrink per seed", "cap - shrink mean", "# seeds positive"], rows)

    # ------------------------------------------------------------------ C5e test rows
    h3("C5e. Test files: every row (frozen configs; test split)")
    para(f"**C5e-1. regmean_C_test.jsonl ({len(ctest)} rows)**: budget C, n = 512; configs frozen from dev (RESEARCH_LOG 16:15 entry). `mean_norm`/`mean_acc` in %; kappa columns as in C5a-3; then absolute per-task accuracy (%).")
    master_test_table(ctest, ["kappa_before_median", "kappa_after_median", "kappa_after_max", "rho_median"], ["kappa_before_median", "kappa_after_median", "kappa_after_max", "rho_median"], "C5e-1")
    cover(src_ma("regmean_C_test.jsonl"), len(ctest), len(ctest), "C5e-1")
    para(f"**C5e-2. regmean_B_test.jsonl ({len(btest)} rows)**: budget B (pooled, no task ids).")
    master_test_table(btest, ["kappa_before_median", "kappa_after_median", "kappa_after_max", "rho_median"], ["kappa_before_median", "kappa_after_median", "kappa_after_max", "rho_median"], "C5e-2")
    cover(src_ma("regmean_B_test.jsonl"), len(btest), len(btest), "C5e-2")
    para(f"**C5e-3. instability_C_test.jsonl ({len(itest)} rows)**: n = 16 per task, subset seeds 0/1/2, alpha = 1.")
    master_test_table(itest, ["kappa_before_median", "kappa_before_max", "kappa_after_median"], ["kappa_before_median", "kappa_before_max", "kappa_after_median"], "C5e-3")
    cover(src_ma("instability_C_test.jsonl"), len(itest), len(itest), "C5e-3")

    def tget(df, **flt):
        d = df
        for k, v in flt.items():
            if isinstance(v, float) and math.isnan(v):
                d = d[d[k].isna()]
            else:
                d = d[np.isclose(d[k].astype(float), v)] if isinstance(v, (int, float)) and k not in ("n", "seed") else d[d[k] == v]
        assert len(d) == 1, (flt, len(d))
        return d.iloc[0]

    para("**C5e-4. Test summary of the primary comparisons (mean_norm %, alpha as stated)**")
    rows = []
    t_plain = tget(ctest, method="shrink", shrink=1.0, alpha=1.0)
    t_kc = tget(ctest, method="kcap_mean", kappa_target=1e5, alpha=1.2)
    t_sh = tget(ctest, method="shrink", shrink=0.99, alpha=1.2)
    for nm_, r in (("RegMean plain (a = 1, alpha = 1)", t_plain), ("kappa cap 1e5, ridge->mean (alpha 1.2)", t_kc), ("shrink a = 0.99 (alpha 1.2)", t_sh)):
        rows.append(["C, n = 512", nm_, nm(r["norm"], 2), nm(r["acc"], 2)])
    b_plain = tget(btest, method="shrink", shrink=1.0, alpha=1.0)
    b_diag = tget(btest, method="shrink", shrink=0.0, alpha=1.6)
    b_kc12 = tget(btest, method="kcap_mean", kappa_target=1e3, alpha=1.2)
    for nm_, r in (("RegMean-B plain (alpha 1)", b_plain), ("RegMean-B kappa cap 1e3 (alpha 1.2)", b_kc12), ("RegMean-B diagonal solve a = 0 (alpha 1.6)", b_diag)):
        rows.append(["B, pooled", nm_, nm(r["norm"], 2), nm(r["acc"], 2)])
    tab(["budget", "config", "test mean_norm (%)", "test mean_acc (%)"], rows)
    inst_t = {}
    for n_meth, (m, sh, kt) in (("plain", ("shrink", 1.0, None)), ("shrink 0.95", ("shrink", 0.95, None)), ("kappa cap 1e5", ("kcap_mean", 1.0, 1e5))):
        sub = itest[(itest.method == m) & np.isclose(itest.shrink, sh)]
        if kt is not None:
            sub = sub[np.isclose(sub.kappa_target.fillna(-1), kt)]
        inst_t[n_meth] = sub.sort_values("seed")["norm"].values

    check("C5-1", "REPORT_02 s2 / task brief", "full-FT RegMean C plain, test mean_norm (a = 1, alpha = 1)", 91.39, t_plain["norm"], 2)
    check("C5-2", "REPORT_02 s2 / task brief", "full-FT RegMean C plain, test mean_acc", 82.77, t_plain["acc"], 2)
    check("C5-3", "REPORT_02 s2", "kappa cap 1e5 (alpha 1.2), test mean_norm", 91.34, t_kc["norm"], 2)
    check("C5-4", "REPORT_02 s2", "shrink a = 0.99 (alpha 1.2), test mean_norm", 91.24, t_sh["norm"], 2)
    d_plain = tget(cdev, method="shrink", shrink=1.0, alpha=1.0)
    d_kc_mean = tget(cdev, method="kcap_mean", kappa_target=1e5, alpha=1.2)
    d_kc_zero = tget(cdev, method="kcap_zero", kappa_target=1e5, alpha=1.2)
    d_sh = tget(cdev, method="shrink", shrink=0.99, alpha=1.2)
    check("C5-5", "REPORT_02 s2", "plain RegMean, dev mean_norm", 91.68, d_plain["norm"], 2)
    check("C5-6", "REPORT_02 s2 / FULL_REPORT s16.1", "kappa cap 1e5 (dev-best, alpha 1.2) dev mean_norm - the config evaluated on test is ridge->mean", 92.08, d_kc_mean["norm"], 2,
          f"ridge->mean = {d_kc_mean['norm']:.4f}; the ridge->zero variant gives {d_kc_zero['norm']:.4f} (that is the dev-best overall and rounds to 92.08)")
    check("C5-7", "REPORT_02 s2", "shrink a = 0.99 (dev-best, alpha 1.2) dev mean_norm", 91.93, d_sh["norm"], 2)
    check("C5-8", "REPORT_02 s2", "RegMean-B plain test at alpha = 1", 75.65, b_plain["norm"], 2)
    check("C5-9", "REPORT_02 s2", "RegMean-B kappa-capped version test", 75.9, b_kc12["norm"], 1, "kappa_t = 1e3, alpha = 1.2 (alpha = 1 gives 75.06)")
    check("C5-10", "REPORT_02 s2", "RegMean-B best (diagonal solve, alpha 1.6) test", 77.35, b_diag["norm"], 2)
    # instability numbers
    for n, (pl_doc, sh_doc, kc_doc) in {16: (64.95, 88.14, 88.37), 64: (89.99, 90.42, 90.73), 512: (91.68, 91.31, 91.58)}.items():
        check(f"C5-inst-plain-{n}", "REPORT_02 s2 table", f"alpha = 1, C, n = {n}, plain, dev (mean over seeds)", pl_doc, inst_mean[(1.0, "shrink", 1.0, None, n)], 2)
        check(f"C5-inst-shr-{n}", "REPORT_02 s2 table", f"alpha = 1, C, n = {n}, shrink a = 0.95, dev", sh_doc, inst_mean[(1.0, "shrink", 0.95, None, n)], 2)
        check(f"C5-inst-kc-{n}", "REPORT_02 s2 table", f"alpha = 1, C, n = {n}, kappa cap 1e5, dev", kc_doc, inst_mean[(1.0, "kcap_mean", 1.0, 1e5, n)], 2)
    check("C5-inst-plain-16-lo", "REPORT_02 s2 table", "n = 16 plain dev range lower (63.7-65.9)", 63.7, inst_mean[(1.0, "shrink", 1.0, None, 16, "vals")].min(), 1)
    check("C5-inst-plain-16-hi", "REPORT_02 s2 table", "n = 16 plain dev range upper (63.7-65.9)", 65.9, inst_mean[(1.0, "shrink", 1.0, None, 16, "vals")].max(), 1)
    check("C5-inst-test-plain", "REPORT_02 s2 table", "n = 16 plain, test (mean over 3 seeds)", 64.20, inst_t["plain"].mean(), 2)
    check("C5-inst-test-shr", "REPORT_02 s2 table", "n = 16 shrink 0.95, test", 87.39, inst_t["shrink 0.95"].mean(), 2)
    check("C5-inst-test-kc", "REPORT_02 s2 table", "n = 16 kappa cap 1e5, test", 87.69, inst_t["kappa cap 1e5"].mean(), 2)
    for i, (sd, v) in enumerate(zip((0.19, 0.43, 0.27), inst_t["kappa cap 1e5"] - inst_t["shrink 0.95"])):
        check(f"C5-inst-test-edge{i}", "REPORT_02 s2 / RESEARCH_LOG", f"n = 16 test cap - shrink 0.95, seed {i}", sd, v, 2)
    check_range("C5-edge-dev", "REPORT_02 s2 / AA_SIDE_REPORT s9", "dev kappa cap 1e5 minus shrink 0.95 at alpha 1 for n = 16 / 64 / 512 (doc: '+0.2 to +0.3 at every n')", "+0.2 to +0.3", 0.2, 0.3,
                [edge_dev[n] for n in (16, 64, 512)], 1)
    check("C5-kG-med16", "REPORT_02 s2", "kappa(G) median at n = 16 (2.9e5)", 2.9, k16.med.mean() / 1e5, 1, "mean of the three subset medians in units of 1e5")
    check("C5-kG-max16", "REPORT_02 s2 / s5.1 / FULL_REPORT s16.2", "kappa(G) maximum at n = 16 (1.6e13), in units of 1e13", 1.6, k16.mx.max() / 1e13, 1,
          f"individual maxima per subset: {', '.join(sci(x, 3) for x in k16.mx)}; the doc's 1.6e13 matches the smallest (subset 2), the largest is {sci(k16.mx.max(), 3)} (same order of magnitude)", sev="minor")

    # ------------------------------------------------------------------ C5f twobytwo
    h3("C5f. twobytwo_C_dev.jsonl and twobytwo_C_test.jsonl: the pre-declared 2x2 cells, every row")
    kts = sorted(set(tdev["kappa_t"].dropna()))
    shs = sorted(set(tdev["shrink"].dropna()))
    sets_ = sorted({(int(a), int(b)) for a, b in zip(tdev.n, tdev.seed)})
    para(f"Budget C, alpha = {sset(sorted(set(tdev.alpha)))}. Cells: `kcap_ridge`, `kcap_shrink`, `rel_ridge` at kappa_t in {sset(kts)} and `fixed_shrink` at a in {sset(shs)}; "
         f"Gram sets (n, subset seed): {sets_}. {len(tdev)} rows per split = {len(tdev) // len(sets_)} configs x {len(sets_)} Gram sets. "
         "`c_rel` is the relative ridge of the `rel_ridge` cell; `lambda_ratio_median` = r_med.")
    for lab, df_ in (("dev", tdev), ("test", ttest)):
        d = df_.sort_values(["n", "seed", "cell", "kappa_t", "shrink"], kind="mergesort")
        rows = [[str(int(r.n)), str(int(r.seed)), r.cell, kcell(knone(r.kappa_t)), kcell(knone(r.shrink)), gen(r.alpha), nm(r.norm, 3), nm(r.acc, 3), nm(r.min_norm, 2),
                 nm(r.shrink_median, 4), nm(r.shrink_min, 3), sci(r.rel_rho_median, 3), sci(r.c_rel, 3), nm(r.lambda_ratio_median, 2)] for r in d.itertuples()]
        para(f"**C5f-{'1' if lab == 'dev' else '2'}. twobytwo_C_{lab}.jsonl, every row ({len(df_)} rows)**")
        tab(["n", "seed", "cell", "kappa_t", "shrink", "alpha", "mean_norm", "mean_acc", "min_norm", "shrink_median", "shrink_min", "rel_rho_median", "c_rel", "lambda_ratio_median"], rows)
        cover(src_ma(f"twobytwo_C_{lab}.jsonl"), len(df_), len(rows), f"C5f-{'1' if lab == 'dev' else '2'}")

    # 2x2 summary from raw
    def two_summary(df_, kt):
        out = {}
        for n in (16, 512):
            v = {"kcap_ridge": arr_for(df_, "kcap_ridge", kt, n), "kcap_shrink": arr_for(df_, "kcap_shrink", kt, n),
                 "rel_ridge": arr_for(df_, "rel_ridge", kt, n), "fixed_shrink": arr_for(df_, "fixed_shrink", 0.95, n)}
            m = {k: float(x.mean()) for k, x in v.items()}
            eff = {"iso_kadaptive (kcap_ridge - kcap_shrink)": m["kcap_ridge"] - m["kcap_shrink"], "iso_fixed (rel_ridge - fixed_shrink)": m["rel_ridge"] - m["fixed_shrink"],
                   "adapt_ridge (kcap_ridge - rel_ridge)": m["kcap_ridge"] - m["rel_ridge"], "adapt_shrink (kcap_shrink - fixed_shrink)": m["kcap_shrink"] - m["fixed_shrink"]}
            diffs = {k: v[k] - v["fixed_shrink"] for k in ("kcap_ridge", "kcap_shrink", "rel_ridge")}
            out[n] = {"vals": v, "mean": m, "eff": eff, "diff": diffs}
        return out

    two = {}
    for lab, df_ in (("dev", tdev), ("test", ttest)):
        for kt in (1e5, 1e4):
            two[(lab, kt)] = two_summary(df_, kt)
    para("**C5f-3. 2x2 summary recomputed from the raw rows** (mean_norm %, mean over the 3 subsets for n = 16; 'vs shrink' = cell minus `fixed_shrink` a = 0.95, per subset seed 0/1/2). "
         "kappa_t = 1e5 is the pre-declared primary comparison; kappa_t = 1e4 is the second grid point of the same run.")
    rows = []
    for (lab, kt), res in two.items():
        for n in (16, 512):
            r = res[n]
            rows.append([lab, gen(kt), str(n), nm(r["mean"]["kcap_ridge"]), nm(r["mean"]["kcap_shrink"]), nm(r["mean"]["rel_ridge"]), nm(r["mean"]["fixed_shrink"]),
                         "/".join(sg(x) for x in r["diff"]["kcap_ridge"]), "/".join(sg(x) for x in r["diff"]["kcap_shrink"]), "/".join(sg(x) for x in r["diff"]["rel_ridge"])])
    tab(["split", "kappa_t", "n", "kcap_ridge", "kcap_shrink", "rel_ridge (c = r_med/kappa_t)", "fixed_shrink 0.95", "kcap_ridge vs shrink", "kcap_shrink vs shrink", "rel_ridge vs shrink"], rows)
    para("**C5f-4. The four 2x2 effects** (pt of mean_norm)")
    rows = []
    for (lab, kt), res in two.items():
        for n in (16, 512):
            r = res[n]
            rows.append([lab, gen(kt), str(n)] + [sg(x) for x in r["eff"].values()])
    tab(["split", "kappa_t", "n", "iso_kadaptive = kcap_ridge - kcap_shrink", "iso_fixed = rel_ridge - fixed_shrink", "adapt_ridge = kcap_ridge - rel_ridge", "adapt_shrink = kcap_shrink - fixed_shrink"], rows)

    # compare with twobytwo_summary.json
    tj = read_json(MA / "twobytwo_summary.json", src_ma("twobytwo_summary.json"))
    mxd = 0.0
    nchk = 0
    for lab in ("dev", "test"):
        for n in (16, 512):
            jj = tj["predeclared"][lab][str(n)]
            mine = two[(lab, 1e5)][n]
            for cell in ("kcap_ridge", "kcap_shrink", "fixed_shrink", "rel_ridge"):
                mxd = max(mxd, abs(jj["mean"][cell] - mine["mean"][cell]))
                nchk += 1
            for (jk, mk) in zip(jj["effects"].keys(), mine["eff"].keys()):
                mxd = max(mxd, abs(jj["effects"][jk] - mine["eff"][mk]))
                nchk += 1
    para(f"Cross-check against the peer's derived file `twobytwo_summary.json` (pre-declared block, {nchk} numbers): max abs difference {mxd:.1e} pt.")

    # ------------------------------------------------------------------ C5g strength sweep
    h3("C5g. strength_sweep_C_dev_* and strength_sweep_C_test_*: equal-grid post-hoc sweep, every row and per-arm summary")
    gk = {arm: sorted({float(r["kappa_t"]) for r in sdev.to_dict("records") if r["cell"] == arm}) for arm in ("kcap_ridge", "rel_ridge")}
    gs_ = sorted({float(x) for x in sdev[sdev.cell == "fixed_shrink"]["shrink"]})
    para(f"Budget C, alpha = {sset(sorted(set(sdev.alpha)))}. Arms (equal grids): `kcap_ridge` kappa_t in {sset(gk['kcap_ridge'])}; `rel_ridge` with c = r_med / kappa_t over kappa_t in {sset(gk['rel_ridge'])}; "
         f"`fixed_shrink` a in {sset(gs_)}. Gram sets n16s0, n16s1, n16s2 (n = 16 images per task, random subsets with seeds 0-2) and n512s0. "
         "Declared selection rule (RESEARCH_LOG 22:40/22:45): per arm and per n, the grid value with the highest DEV mean_norm (for n = 16, the mean over the 3 subsets); only those are evaluated on test.")
    for lab, df_ in (("dev", sdev), ("test", stest)):
        d = df_.sort_values(["n", "seed", "cell", "kappa_t", "shrink"], kind="mergesort")
        rows = [[str(int(r.n)), str(int(r.seed)), r.cell, kcell(knone(r.kappa_t)), kcell(knone(r.shrink)), sci(r.c_rel, 3), gen(r.alpha), nm(r.norm, 3), nm(r.acc, 3), nm(r.min_norm, 2),
                 nm(r.shrink_median, 4), nm(r.shrink_min, 3), sci(r.rel_rho_median, 3), nm(r.lambda_ratio_median, 2)] for r in d.itertuples()]
        para(f"**C5g-{'1' if lab == 'dev' else '2'}. Every {lab} row ({len(df_)} rows)**")
        tab(["n", "seed", "cell", "kappa_t", "shrink", "c_rel", "alpha", "mean_norm", "mean_acc", "min_norm", "shrink_median", "shrink_min", "rel_rho_median", "lambda_ratio_median"], rows)
        cover(f"master:results/regmean/strength_sweep_C_{lab}_*.jsonl (4 files)", len(df_), len(rows), f"C5g-{'1' if lab == 'dev' else '2'}")

    def sweep_index(df_):
        out = {}
        for r in df_.to_dict("records"):
            p = r["shrink"] if r["cell"] == "fixed_shrink" else r["kappa_t"]
            out.setdefault((r["cell"], float(p)), {}).setdefault(int(r["n"]), {})[int(r["seed"])] = r
        return out

    sd_idx = sweep_index(sdev)
    st_idx = sweep_index(stest)
    arms = ("kcap_ridge", "rel_ridge", "fixed_shrink")
    para("**C5g-3. Per-arm dev summary: mean_norm (%) by grid value** (n = 16: the three subsets and their mean; `rho` = median over the sets of rel_rho_median, the effective relative ridge rho/(tr(G)/d))")
    rows = []
    sel = {}
    mean_dev = {}
    for arm in arms:
        ps = sorted({p for (a, p) in sd_idx if a == arm})
        for p in ps:
            d16 = sd_idx[(arm, p)].get(16, {})
            d512 = sd_idx[(arm, p)].get(512, {})
            v16 = [d16[s]["norm"] for s in sorted(d16)]
            rho16 = float(np.median([d16[s]["rel_rho_median"] for s in d16]))
            crel16 = float(np.median([d16[s]["c_rel"] for s in d16])) if arm == "rel_ridge" else NAN
            v512 = d512[0]["norm"] if d512 else NAN
            mean_dev[(arm, p, 16)] = float(np.mean(v16))
            mean_dev[(arm, p, 512)] = v512
            rows.append([arm, gen(p), sci(crel16, 3) if arm == "rel_ridge" else "-", "/".join(nm(x, 2) for x in v16), nm(np.mean(v16), 2), sci(rho16, 2),
                         nm(v512, 2), sci(d512[0]["rel_rho_median"], 2) if d512 else "-"])
    tab(["arm", "grid value (kappa_t, or a)", "c_rel (median over n=16 sets)", "n=16 subsets 0/1/2", "n=16 mean", "n=16 rho", "n=512", "n=512 rho"], rows)
    for n in (16, 512):
        for arm in arms:
            cand = {p: v for (a, p, nn), v in mean_dev.items() if a == arm and nn == n}
            sel[(arm, n)] = max(cand, key=cand.get)
    para("**C5g-4. Dev-selected grid value per arm and n (declared rule) and the corresponding TEST evaluation (mean_norm %)**")
    rows = []
    test_mean = {}
    for n in (16, 512):
        for arm in arms:
            p = sel[(arm, n)]
            tdict = st_idx.get((arm, p), {}).get(n, {})
            tv = [tdict[s]["norm"] for s in sorted(tdict)]
            test_mean[(arm, n)] = float(np.mean(tv)) if tv else NAN
            tcount = len(tv)
            rows.append([str(n), arm, gen(p), nm(mean_dev[(arm, p, n)], 2), "/".join(nm(x, 2) for x in tv) if tv else "-", nm(test_mean[(arm, n)], 2) if tv else "-", str(tcount)])
    tab(["n", "arm", "dev-selected value", "dev mean_norm", "test mean_norm per subset seed", "test mean", "# test rows found"], rows)
    # which test configs exist
    present = sorted({(a, p, n) for (a, p), d in st_idx.items() for n in d})
    para("Test configurations present in `strength_sweep_C_test_*`: " + "; ".join(f"{a} {gen(p)} n={n}" for a, p, n in present) + ". "
         "They coincide with the dev-selected configs above: " + str(all((a, p) == (a, sel[(a, n)]) for a, p, n in present)) + ".")
    para("**C5g-5. Test differences (pt of mean_norm) between arms, per subset seed and mean**")
    rows = []
    sweep_t = {}
    for n in (16, 512):
        vals = {}
        for arm in arms:
            p = sel[(arm, n)]
            tdict = st_idx.get((arm, p), {}).get(n, {})
            vals[arm] = np.array([tdict[s]["norm"] for s in sorted(tdict)])
        if not all(len(v) for v in vals.values()):
            rows.append([str(n), "incomplete", "", "", ""])
            continue
        for lab_, a1, a2 in (("kcap_ridge - rel_ridge (primary: isolates kappa-adaptive strength)", "kcap_ridge", "rel_ridge"), ("kcap_ridge - fixed_shrink", "kcap_ridge", "fixed_shrink"),
                             ("rel_ridge - fixed_shrink", "rel_ridge", "fixed_shrink")):
            dd = vals[a1] - vals[a2]
            sweep_t[(n, a1, a2)] = dd
            rows.append([str(n), lab_, "/".join(sg(x) for x in dd), sg(dd.mean()), f"{int((dd > 0).sum())}/{len(dd)}"])
    tab(["n", "difference", "per subset seed 0/1/2", "mean", "# positive"], rows)
    para(f"Context: plain (unregularised) RegMean on test is {inst_t['plain'].mean():.2f} at n = 16 (mean of the 3 subsets, C5e-3) and {t_plain['norm']:.2f} at n = 512 (regmean_C_test, C5e-1).")

    # reproducibility
    pairs = []
    for lab, a, b in (("dev", tdev, sdev), ("test", ttest, stest)):
        ka = {(r.cell, None if isnan(r.kappa_t) else float(r.kappa_t), None if isnan(r.shrink) else float(r.shrink), int(r.n), int(r.seed), float(r.alpha)): r.norm for r in a.itertuples()}
        for r in b.itertuples():
            k = (r.cell, None if isnan(r.kappa_t) else float(r.kappa_t), None if isnan(r.shrink) else float(r.shrink), int(r.n), int(r.seed), float(r.alpha))
            if k in ka:
                pairs.append((lab, abs(r.norm - ka[k])))
    para(f"Reproducibility: {sum(1 for l, _ in pairs if l == 'dev')} dev configs and {sum(1 for l, _ in pairs if l == 'test')} test configs appear both in the pre-declared 2x2 run and in the sweep; "
         f"max abs mean_norm difference = {max(d for _, d in pairs) if pairs else NAN:.4f} pt.")
    check("C5g-repro", "RESEARCH_LOG 22:45", "24 repeated configs reproduce exactly (max |diff| 0.000)", 24, sum(1 for l, _ in pairs if l == "dev"), 0, f"max |diff| = {max(d for l, d in pairs if l == 'dev'):.4f} (dev)")

    # compare with summary json: sweep dev means, selected, test
    mx2 = 0.0
    n2 = 0
    for k, v in tj["sweep"]["dev"].items():
        arm, ps = k.split("|")
        p = float(ps)
        for nstr, vv in v.items():
            mx2 = max(mx2, abs(vv["norm"] - mean_dev[(arm, p, int(nstr))]))
            n2 += 1
    for nstr, selj in tj["sweep"]["selected"].items():
        for arm, key in selj.items():
            n2 += 1
            if float(key.split("|")[1]) != sel[(arm, int(nstr))]:
                mx2 = max(mx2, 1.0)
    for nstr, rr in tj["sweep"]["test"].items():
        for arm in arms:
            mx2 = max(mx2, abs(rr[arm]["mean"] - test_mean[(arm, int(nstr))]))
            n2 += 1
    para(f"Cross-check against `twobytwo_summary.json` sweep block ({n2} numbers: dev means, selected values, test means): max abs difference {mx2:.1e} pt.")

    # log claims
    k3 = mean_dev
    check("C5g-1", "RESEARCH_LOG 22:45", "kcap ridge peak n = 16, kappa_t 3e4 (88.32)", 88.32, k3[("kcap_ridge", 30000.0, 16)], 2)
    check("C5g-2", "RESEARCH_LOG 22:45", "kcap ridge peak n = 16, kappa_t 1e5 (88.37)", 88.37, k3[("kcap_ridge", 100000.0, 16)], 2)
    check("C5g-3", "RESEARCH_LOG 22:45", "kcap ridge n = 512 peak (91.58)", 91.58, k3[("kcap_ridge", 100000.0, 512)], 2)
    check("C5g-4", "RESEARCH_LOG 22:45", "fixed relative ridge peak n = 16 (88.02)", 88.02, max(v for (a, p, nn), v in k3.items() if a == "rel_ridge" and nn == 16), 2)
    check("C5g-5", "RESEARCH_LOG 22:45", "fixed shrinkage peak n = 16 (88.14)", 88.14, max(v for (a, p, nn), v in k3.items() if a == "fixed_shrink" and nn == 16), 2)
    check("C5g-6", "RESEARCH_LOG 22:45", "fixed shrinkage peak n = 512 (91.55)", 91.55, max(v for (a, p, nn), v in k3.items() if a == "fixed_shrink" and nn == 512), 2)
    plain512 = float(d_plain["norm"])
    vals512 = [v for (a, p, nn), v in k3.items() if nn == 512]
    n_within = sum(1 for v in vals512 if plain512 - v <= 0.2 + 1e-9)
    peaks = {arm: max(v for (a, p, nn), v in k3.items() if a == arm and nn == 512) for arm in arms}
    check_bool("C5g-7", "RESEARCH_LOG 22:45", "n = 512: 'every regularised solve is within 0.2 of plain RegMean (91.68)'", "all regularised grid values within 0.2",
               f"{n_within}/{len(vals512)} grid values within 0.2", n_within == len(vals512),
               note=f"worst grid value is {plain512 - min(vals512):.2f} below plain; the statement holds for each arm's peak ({', '.join(f'{arm} {plain512 - pk:.2f} below' for arm, pk in peaks.items())}) but not for the whole grid",
               sev="wording")
    two_t = two[("test", 1e5)]
    check("C5f-1", "RESEARCH_LOG 22:40", "cell 2 (kcap_shrink) vs a = 0.95, test, n = 16 (-0.11)", -0.11, float(two_t[16]["diff"]["kcap_shrink"].mean()), 2)
    check("C5f-2", "RESEARCH_LOG 22:40", "cell 2 (kcap_shrink) vs a = 0.95, test, n = 512 (-0.48)", -0.48, float(two_t[512]["diff"]["kcap_shrink"].mean()), 2)
    check("C5f-3", "RESEARCH_LOG 22:40", "cell 4 (rel_ridge) vs a = 0.95, test, n = 512 (+0.31)", 0.31, float(two_t[512]["diff"]["rel_ridge"].mean()), 2,
          "recomputed 91.3750 - 91.0721 = +0.30")
    check("C5f-4", "RESEARCH_LOG 22:40", "cell 4 (rel_ridge) vs a = 0.95, test, n = 16 (-0.49)", -0.49, float(two_t[16]["diff"]["rel_ridge"].mean()), 2)
    # selection claim
    check("C5g-sel16", "RESEARCH_LOG 22:45 table", "n = 16 dev-selected kcap_ridge kappa_t = 1e5", 1e5, sel[("kcap_ridge", 16)], 0, text=True)
    check("C5g-sel16r", "RESEARCH_LOG 22:45 table", "n = 16 dev-selected rel_ridge value (c = r_med/1e4)", 1e4, sel[("rel_ridge", 16)], 0, text=True)
    check("C5g-sel16s", "RESEARCH_LOG 22:45 table", "n = 16 dev-selected a", 0.95, sel[("fixed_shrink", 16)], 2, text=True)
    check("C5g-sel512", "RESEARCH_LOG 22:45 table", "n = 512 dev-selected kcap_ridge kappa_t", 1e5, sel[("kcap_ridge", 512)], 0, text=True)
    check("C5g-sel512r", "RESEARCH_LOG 22:45 table", "n = 512 dev-selected rel_ridge value (c = r_med/1e6)", 1e6, sel[("rel_ridge", 512)], 0, text=True)
    check("C5g-sel512s", "RESEARCH_LOG 22:45 table", "n = 512 dev-selected a", 0.99, sel[("fixed_shrink", 512)], 2, text=True)

    STASH.update(two=two, mean_dev=mean_dev, sel=sel, test_mean=test_mean, sweep_t=sweep_t, sd_idx=sd_idx, st_idx=st_idx, inst_t=inst_t, inst_mean=inst_mean,
                 t_plain=t_plain, stest=stest, sdev=sdev, d_plain=d_plain, edge_dev=edge_dev, ctest=ctest, cdev=cdev, idev=idev, itest=itest)

    # ------------------------------------------------------------------ C5h kappa(G) statistics
    h3("C5h. kappa2(G) statistics wherever the runs recorded them (full-FT)")
    para("kappa2(G) = condition number of the summed per-map input Gram; reported as the median over the linear maps that have a Gram (`kappa_before_*` = of the (shrunk) Gram before any ridge). "
         "'-' = not recorded. Sources: dev/test files above; each value is constant across alpha within a Gram set.")
    rows = []

    def first_val(df_, col, **flt):
        d = df_
        for k, v in flt.items():
            if v is None:
                d = d[d[k].isna()]
            else:
                d = d[np.isclose(d[k].astype(float), v)] if k not in ("method",) else d[d[k] == v]
        if col not in d.columns:
            return NAN
        x = d[col].dropna()
        return float(x.iloc[0]) if len(x) else NAN
    for lab, df_, kind in (("budget C, n = 512 (dev+test)", pd.concat([cdev, ctest]), "reg"), ("budget B pooled 4096/task (dev+test)", pd.concat([bdev, btest]), "reg")):
        row = [lab]
        row.append(sci(first_val(df_, "kappa_before_median", method="shrink", shrink=1.0), 3))
        row.append(sci(first_val(df_, "kappa_after_max", method="shrink", shrink=1.0), 3))
        for a in (0.99, 0.95, 0.9, 0.7, 0.5, 0.0):
            row.append(sci(first_val(df_, "kappa_before_median", method="shrink", shrink=a), 3))
        rows.append(row)
    for n in (16, 64, 512):
        for seed in ((0, 1, 2) if n < 512 else (0,)):
            d_ = idev[(idev.n == n) & (idev.seed == seed)]
            t_ = itest[(itest.n == n) & (itest.seed == seed)]
            row = [f"budget C, n = {n}, subset seed {seed} (instability, dev+test)"]
            row.append(sci(first_val(d_, "kappa_before_median", method="kcap_mean"), 3))
            row.append(sci(first_val(d_, "kappa_before_max", method="kcap_mean"), 3))
            row += ["-", sci(first_val(t_, "kappa_before_median", method="shrink", shrink=0.95), 3) if len(t_) else "-", "-", "-", "-", "-"]
            rows.append(row)
    tab(["Gram set", "kappa2(G) median, plain (a = 1)", "kappa2(G) max, plain", "median at a = 0.99", "a = 0.95", "a = 0.9", "a = 0.7", "a = 0.5", "a = 0.0 (diagonal)"], rows)
    para("For the n = 16/64/512 instability sets the plain-Gram maximum is the `kappa_before_max` logged on the kappa-cap rows; shrunk-Gram medians exist only for a = 0.95 on the n = 16 test rows. "
         "The 2x2/sweep runs additionally log `lambda_ratio_median` (r_med = median over maps of lambda_max/(tr G / d)): see C5f-1/2 and C5g-1/2.")

    # ------------------------------------------------------------------ C5i verification
    h3("C5i. Verification of the metrics in every master file")
    rows = []
    allm = {"regmean_C_dev.jsonl": cdev, "regmean_B_dev.jsonl": bdev, "instability_C_dev.jsonl": idev, "regmean_C_test.jsonl": ctest, "regmean_B_test.jsonl": btest,
            "instability_C_test.jsonl": itest, "twobytwo_C_dev.jsonl": tdev, "twobytwo_C_test.jsonl": ttest}
    for _, g_ in sdev.groupby("_file"):
        allm[g_["_file"].iloc[0]] = g_
    for _, g_ in stest.groupby("_file"):
        allm[g_["_file"].iloc[0]] = g_
    for v in VERIFY:
        if v["kind"] != "master":
            continue
        fname = v["file"].split("/")[-1]
        cm = allm[fname]["commit"].value_counts().to_dict()
        rows.append([v["file"], v["n"], v["split"], f"{v['max_diff_norm']:.1e}", f"{v['max_diff_avg']:.1e}", ", ".join(f"{k}: {n_}" for k, n_ in cm.items())])
    tab(["file", "rows", "split used for expert_accuracy reference", "max abs diff: stored mean_norm / min_norm vs recomputed from `acc` and expert_accuracy.json", "max abs diff: stored mean_acc vs mean of per-task acc",
         "rows per git commit stamp of the producing run"], rows)
    para("`mean_norm` = mean_t acc_t / expert_acc_t (expert accuracy of the task's own full-FT expert on the same split) is confirmed in every row; `mean_acc` is the plain mean of the 8 absolute task accuracies. "
         "Percent columns in this appendix are these stored fractions x100.")


# ---- C6 -------------------------------------------------------------------------------------------------------------
# (file, bank, N images per task, calibration subset, final-rank truncation)
LORA_RM = [
    ("regmean_knots_release_dev.jsonl", "knots_release", "512", "all 512 calibration images", "none (exact merge, rank <= 128)"),
    ("regmean_knots_release_n64_dev.jsonl", "knots_release", "64", "first 64 images (gram suffix _n64, seed -1)", "none"),
    ("regmean_knots_release_n16_dev.jsonl", "knots_release", "16", "s0 = first 16 images (gram suffix _n16, seed -1)", "none"),
    ("regmean_knots_release_n16_s1_dev.jsonl", "knots_release", "16", "s1 = random subset, seed 1 (suffix _n16_s1)", "none"),
    ("regmean_knots_release_n16_s2_dev.jsonl", "knots_release", "16", "s2 = random subset, seed 2 (suffix _n16_s2)", "none"),
    ("regmean_knots_release_r16_dev.jsonl", "knots_release", "512", "all 512 calibration images", "every merged update truncated to rank 16 (one adapter)"),
    ("regmean_lora_c5a_dev.jsonl", "lora_c5a", "512", "all 512 calibration images", "none"),
    ("regmean_lora_dev.jsonl", "lora (vit20)", "512", "all 512 calibration images", "none"),
]
BUDGET_ORDER = {"C": 0, "Bp": 1, "B0": 2}

# per sub-table "what is varied / split / metric / units" and sources (emitted right after each ### heading)
_RS = [f"aa:results/aa/lora/rule_sweep_dev_shard{i}.jsonl" for i in range(3)]
_RM = lambda names: [f"aa:results/aa/lora/{n}" for n in names]  # noqa: E731
_MR = lambda names: [f"master:results/regmean/{n}" for n in names]  # noqa: E731
SUBINFO = {
    "C1a": ("Varied: nothing (a list of checks); split: none; metric: relative errors, ratios, counts; units: unitless.", _RM(["lora_checks.json", "lora_stats.csv"]) + ["aa:tests/test_aa_lora.py"]),
    "C1b": ("Varied: strength of the random gauge change, cond(R) in {10, 1e4}; split: none; metric: relative errors and ratios of the factor condition numbers; units: unitless.", _RM(["lora_checks.json"])),
    "C1c": ("Varied: map (48) x gauge strength (2); split: none; metric: relative errors, factor condition numbers before/after; units: unitless.", _RM(["lora_checks.json"])),
    "C2a": ("Varied: map type (4); split: none (spectra of the 8 experts' W0 + Delta); metric: kappa2 and sigma values of the full weight; units: unitless / weight scale.", _RM(["lora_stats.csv"])),
    "C2b": ("Varied: map type (4); split: none; metric: norms, stable/entropy rank, kappa_plus and numerical rank of the update Delta; units: unitless / counts.", _RM(["lora_stats.csv"])),
    "C2c": ("Varied: map type (4); split: none; metric: condition numbers of the LoRA factors A and B; units: unitless.", _RM(["lora_stats.csv"])),
    "C2d": ("Varied: map (48), each over the 8 experts; split: none; metric: kappa2(W_i) spread, argmin expert, Spearman(kappa2, sigma_min); units: unitless.", _RM(["lora_stats.csv"])),
    "C2e": ("Varied: map type (4); split: none; metric: max/min over experts of kappa2(W_i) within a map; units: ratio.", _RM(["lora_stats.csv"])),
    "C2f": ("Varied: expert (8); split: none; metric: number of maps won under argmin-kappa, medians of kappa2, sigma_min, ||Delta||_F, stable rank; units: counts / unitless.", _RM(["lora_stats.csv"])),
    "C2g": ("Varied: block (12) x map type (4); split: none; metric: median over experts of kappa2(W_i); units: unitless.", _RM(["lora_stats.csv"])),
    "C3a": ("Varied: rule x b, maximised over (b, alpha) per rule; split: dev; metric: avg_acc (absolute); units: percent, differences in pt.", _RS),
    "C3b": ("Varied: rule x alpha, maximised over b per cell; split: dev; metric: avg_acc (absolute); units: percent.", _RS),
    "C3c": ("Varied: rule x b x alpha (the full ragged grid); split: dev; metric: avg_acc (absolute); units: percent.", _RS),
    "C4a": ("Varied: selection scheme (kappa_argmin / global_freq / count_preserving / free_random_selection) x alpha x seed; split: dev; metric: avg_acc (absolute); units: percent, percentile of a null.", _RM(["selection_controls.jsonl", "selection_controls_null20.jsonl"])),
    "C4b": ("Varied: bank (4) x scheme x alpha x seed; split: dev; metric: avg_acc (absolute); units: percent, percentile of a 20-seed null.", _RM(["replicate_banks.jsonl", "selection_controls_null20.jsonl"]) + [_RS[0].replace("shard0", "shard*")]),
    "C4c": ("Varied: hard selection rule (argmin kappa / gentlest / strongest functional energy) x alpha; split: dev; metric: avg_acc (absolute), agreement fractions, Spearman; units: percent / unitless.", _RM(["gentle.json", "lora_stats.csv"])),
    "C4d": ("Varied: oracle (2) x spectral score (7); split: dev for fitting, test for the oracle's own accuracy; metric: Spearman correlations with permutation p-values, avg_acc (absolute); units: unitless / percent.", _RM(["oracle_vs_scores.json", "oracle_vs_scores.md"])),
    "C4e": ("Varied: map (48) x receiving expert (8), one map handed over at a time to a uniform base; split: dev; metric: avg_acc (absolute) and gain over the uniform base; units: percent, pt.", _RM(["single_map_swap.jsonl", "single_map_swap_summary.json", "lora_stats.csv"])),
    "C4f": ("Varied: bank (4) x alpha x b x shuffle seed for master's rule logkappa_q90_rev; split: dev; metric: avg_acc (absolute); units: percent, pt.", _RM(["q90_transfer.jsonl"])),
    "C4g": ("Varied: the 10 frozen configurations; split: TEST; metric: avg_acc and per-task accuracy (absolute); units: percent, pt.", _RM(["test_freeze.jsonl"])),
    "C5a": ("Varied: 19 RegMean configs (shrink a, kappa cap kappa_t with ridge->mean/zero) x alpha (5); budget C, n = 512; split: dev; metric: mean_norm (NORMALIZED) and mean_acc (absolute); units: percent.", _MR(["regmean_C_dev.jsonl"])),
    "C5b": ("Varied: 19 RegMean configs x alpha (5); budget B (pooled, no task ids); split: dev; metric: mean_norm (NORMALIZED) and mean_acc (absolute); units: percent.", _MR(["regmean_B_dev.jsonl"])),
    "C5c": ("Varied: family of config; split: dev, and test for the frozen configs; metric: mean_norm (NORMALIZED) / mean_acc (absolute); units: percent.", _MR(["regmean_C_dev.jsonl", "regmean_C_test.jsonl", "regmean_B_dev.jsonl", "regmean_B_test.jsonl"])),
    "C5d": ("Varied: calibration images per task n in {16, 64, 512} x subset seed x method x alpha; budget C; split: dev; metric: mean_norm (NORMALIZED), mean_acc, kappa2(G); units: percent / unitless.", _MR(["instability_C_dev.jsonl"])),
    "C5e": ("Varied: the frozen configs (budget C, budget B, noisy n = 16 Grams); split: TEST; metric: mean_norm (NORMALIZED), mean_acc and per-task absolute accuracy; units: percent.", _MR(["regmean_C_test.jsonl", "regmean_B_test.jsonl", "instability_C_test.jsonl"])),
    "C5f": ("Varied: 2x2 cell (kappa-adaptive/fixed x isotropic ridge/diagonal shrinkage) x kappa_t x Gram set (n = 16 seeds 0-2, n = 512); alpha = 1; split: dev and test; metric: mean_norm (NORMALIZED), mean_acc; units: percent, pt.", _MR(["twobytwo_C_dev.jsonl", "twobytwo_C_test.jsonl", "twobytwo_summary.json"])),
    "C5g": ("Varied: arm (kappa-cap ridge / fixed relative ridge / fixed shrinkage) x grid value x Gram set; alpha = 1; split: dev (all grid) and test (dev-selected values only); metric: mean_norm (NORMALIZED), mean_acc; units: percent, pt.", _MR(["strength_sweep_C_dev_n16s0.jsonl", "strength_sweep_C_dev_n16s1.jsonl", "strength_sweep_C_dev_n16s2.jsonl", "strength_sweep_C_dev_n512s0.jsonl", "strength_sweep_C_test_n16s0.jsonl", "strength_sweep_C_test_n16s1.jsonl", "strength_sweep_C_test_n16s2.jsonl", "strength_sweep_C_test_n512s0.jsonl"])),
    "C5h": ("Varied: Gram set (budget, n, subset seed) and shrinkage a; split: dev+test rows; metric: kappa2 of the summed Gram per map (median, max); units: unitless.", _MR(["regmean_C_dev.jsonl", "regmean_B_dev.jsonl", "regmean_C_test.jsonl", "regmean_B_test.jsonl", "instability_C_dev.jsonl", "instability_C_test.jsonl"])),
    "C5i": ("Varied: file; split: as stated; metric: max abs difference between the stored mean_norm / mean_acc and their recomputation from the per-task accuracies and master:results/experts/expert_accuracy.json; units: fraction.", _MR(["*.jsonl (all files above)"]) + ["master:results/experts/expert_accuracy.json"]),
    "C6a": ("Varied: dev file (bank x N x calibration subset x truncation); split: dev; metric: counts and kappa2(G) of the C/full Gram; units: counts / unitless.", _RM([f for f, *_ in LORA_RM])),
    "C6b": ("Varied: budget x solve x regulariser x alpha in {0.8, 1.0, 1.3}, per dev file; split: dev; metric: avg_acc (ABSOLUTE) and Gram conditioning; units: percent / unitless.", _RM([f for f, *_ in LORA_RM])),
    "C6c": ("Varied: regulariser x budget/solve (C6c-1), calibration size (C6c-2), bank/N setting (C6c-3), truncation (C6c-4); split: dev; metric: best avg_acc over alpha (ABSOLUTE); units: percent, pt.", _RM([f for f, *_ in LORA_RM])),
    "C6d": ("Varied: dev file x budget/solve x map type; split: none (Gram statistics at the plain solve); metric: kappa2 of the summed Gram per map; units: unitless.", _RM([f.replace("_dev.jsonl", "_maps.json") for f, *_ in LORA_RM])),
    "C6e": ("Varied: cell (kappa-cap ridge / kappa-cap shrinkage / fixed shrinkage / fixed relative ridge) x strength x Gram set (n512, n16_s0/s1/s2); alpha = 0.8; KnOTS, C, full; split: dev; metric: avg_acc (ABSOLUTE); units: percent, pt.", _RM(["regmean_2x2_dev.jsonl"])),
    "C6f": ("Varied: bank (3) x config (uniform, shrink0.95, kcap1e4_zero) at the dev-best alpha; split: TEST; metric: avg_acc and per-task accuracy (ABSOLUTE) and the paired-bootstrap CI of kcap minus shrink; units: percent, pt.", _RM(["regmean_test.json"] + ["regmean_knots_release_dev.jsonl", "regmean_lora_c5a_dev.jsonl", "regmean_lora_dev.jsonl"])),
    "C6g": ("Varied: dev file; split: dev; metric: best avg_acc (ABSOLUTE) over alpha per regulariser and Gram conditioning; units: percent.", _RM(["regmean_summary.md"] + [f for f, *_ in LORA_RM])),
    "C6h": ("Varied: file; split: as stated; metric: max abs difference between avg_acc and the mean of per-task accuracies, and cross-file agreement of the same uniform merge; units: fraction.", _RM(["(all LoRA jsonl files above)"])),
}




def rm_load(fname: str) -> pd.DataFrame:
    df = lora_df(AA / fname, src_aa(fname))
    df["pct"] = 100 * df["avg_acc"]
    return df


def rm_pick(df, budget="C", solve="full", shrink=1.0, kt=None, rt=None) -> pd.DataFrame:
    d = df[(df.method == "regmean") & (df.budget == budget) & (df.solve == solve) & np.isclose(df.shrink.astype(float), shrink)]
    d = d[d.kappa_target.isna()] if kt is None else d[np.isclose(d.kappa_target.astype(float), kt)]
    if kt is not None and rt is not None:
        d = d[d.ridge_to == rt]
    if kt is None:
        d = d[d.ridge_to.isna() | (d.ridge_to == "None")] if "ridge_to" in d else d
    return d.sort_values("alpha")


def rm_best(df, **kw):
    d = rm_pick(df, **kw)
    if len(d) == 0:
        return None
    i = d["pct"].idxmax()
    return float(d.loc[i, "alpha"]), float(d.loc[i, "pct"]), {float(a): float(v) for a, v in zip(d["alpha"], d["pct"])}


def sec_C6():
    header_block("C6", "LoRA RegMean (rank-16 banks): dev sweeps, calibration size, 2x2, frozen test",
                 "Varied: bank (knots_release, lora_c5a, lora = vit20), Gram budget (C task-separated / Bp pooled label-free through each expert / B0 pooled through the base), solve (full 768-dim input space vs exact adapter subspace `sub`), "
                 "regulariser (RegMean off-diagonal shrinkage a = `shrink`; kappa cap `kappa_target` with ridge toward `zero` or the `mean` update), alpha in {0.8, 1.0, 1.3}, calibration images per task N in {512, 64, 16} "
                 "and the N = 16 subset (s0/s1/s2), and rank-16 truncation (`r16`); split = dev for the sweeps and the 2x2, test (N = 512, C, full solve) for `regmean_test.json`; "
                 "metric = `avg_acc`, the ABSOLUTE accuracy averaged over the 8 tasks (NOT normalized; verified in C6h), units = percent (x100).",
                 [src_aa(f) for f, *_ in LORA_RM] + [src_aa(f.replace("_dev.jsonl", "_maps.json")) for f, *_ in LORA_RM] +
                 [src_aa("regmean_2x2_dev.jsonl"), src_aa("regmean_test.json"), src_aa("regmean_summary.md"),
                  "aa:scripts/aa_lora_regmean.py, aa_lora_regmean_2x2.py, aa_lora_regmean_test.py, aa_lora_grams.py, aa_summarize_regmean.py; aa:src/condmerge/regmean_lora.py"])
    para("Legend: per adapted map (48 attention maps) D = (sum_i D_i G~_i)(sum_i G~_i + rho I)^-1 with G~ = a G + (1 - a) diag G (a = `shrink`; a = 1 plain RegMean). `kappa_target` = kappa cap: rho = smallest ridge with "
         "kappa2(sum G~ + rho I) <= kappa_t (`median_rho` = median over maps); `ridge_to` = `zero` (plain Tikhonov) or `mean` (ridge toward the uniform average of the updates). `solve = full`: dense 768x768 solve, output rank <= 128; "
         "`solve = sub`: solve restricted to the span P of the experts' A-factors (m <= 128 dims), exact for each expert. Budgets: C = expert i on its own task's calibration split (task-separated, unlabeled); "
         "Bp = every expert on the pooled 8 x N images (label-free, task-ID-free); B0 = pooled images through the BASE model (all experts share one Gram, so RegMean must equal averaging; sanity check). "
         "`median_kappa_G` / `max_kappa_G` = median / max over maps of kappa2(sum G~) (inf = numerically singular), `median_kappa_solve` = of the matrix actually solved. "
         "`uniform` rows = plain average of the dense updates. A collapsed solve returns exactly 4.09% (chance-level constant prediction), shown as 4.09.")

    rb_banks = sorted({b_ for b_ in lora_df(AA / "replicate_banks.jsonl", "x")["bank"]} | {"knots_release"})
    rm_banks = sorted({b.split(" ")[0] for _, b, *_ in LORA_RM})
    gram_files = sorted(p_.name for p_ in (AAREPO / "artifacts" / "cache" / "aa").glob("grams_lora_*"))
    missing_banks = sorted(set(rb_banks) - set(rm_banks))
    para(f"Banks with RegMean runs: {rm_banks}. Banks present elsewhere in the LoRA results but WITHOUT any RegMean file: {missing_banks} "
         f"(directory listing of aa:artifacts/cache/aa: {len(gram_files)} Gram files `grams_lora_*`, none for {missing_banks}).")
    data = {}
    for f, bank, n, sub, trunc in LORA_RM:
        data[f] = rm_load(f)
        gr = sorted({str(x) for x in data[f]["grams"].dropna()}) if "grams" in data[f].columns else []
        exp_g = "_n512" if n == "512" else ("_n64" if n == "64" else None)
        if n == "16":
            exp_g = "_n16" + ("_s1" if "s1" in f else "_s2" if "s2" in f else "")
        if gr and exp_g and gr != [exp_g]:
            UNPARSED.append(f"{f}: `grams` field {gr} does not match the declared calibration set {exp_g}")

    # ---------------------------------------------------------------- C6a overview
    h3("C6a. Overview of the eight dev files")
    rows = []
    for f, bank, n, sub, trunc in LORA_RM:
        d = data[f]
        reg = d[d.method == "regmean"]
        cfgs = reg.groupby(["budget", "solve", "shrink", "kappa_target", "ridge_to"], dropna=False).ngroups
        mpath = AA / f.replace("_dev.jsonl", "_maps.json")
        mj = read_json(mpath, src_aa(mpath.name))
        kc = np.array([v["kappa_G"] for v in mj["C/full"].values()])
        rows.append([f, bank, n, sub, trunc, len(d), cfgs, "/".join(sorted(reg.budget.unique(), key=lambda b: BUDGET_ORDER[b])), "/".join(sorted(reg.solve.unique())),
                     sci(np.median(kc), 3), sci(np.max(kc), 3)])
    tab(["file", "bank", "N per task", "calibration set", "final-rank truncation", "rows", "regmean configs", "budgets", "solves", "median kappa2(G), C/full", "max kappa2(G), C/full"], rows,
        ["l", "l", "r", "l", "l", "r", "r", "l", "l", "r", "r"])
    para("The N = 16 'seeds': per `aa_lora_grams.py`, `--seed -1` (the default, file suffix `_n16`, labelled `n16_s0` in the 2x2 script) takes the FIRST N calibration images, `--seed k >= 0` a random subset of N (suffix `_n16_s<k>`). "
         "So LoRA s0 is a deterministic prefix, while master's full-FT n = 16 subsets (C5d) are random subsets with seeds 0/1/2: the calibration images differ between the two sides.")

    # ---------------------------------------------------------------- C6b per-file tables
    h3("C6b. Every config of every dev file (accuracy at alpha 0.8 / 1.0 / 1.3, best alpha, Gram conditioning)")
    best_tables = {}
    for f, bank, n, sub, trunc in LORA_RM:
        d = data[f]
        para(f"**C6b: `{f}`** - bank `{bank}`, N = {n} per task, calibration set: {sub}; truncation: {trunc}. {len(d)} rows.")
        u = d[d.method == "uniform"].sort_values("alpha")
        para("uniform (mean of dense updates): " + "; ".join(f"alpha {a:g}: {nm(v)}" for a, v in zip(u["alpha"], u["pct"])) + f"; best {nm(u['pct'].max())} at alpha {u.loc[u['pct'].idxmax(), 'alpha']:g}.")
        reg = d[d.method == "regmean"].copy()
        rows = []
        seen = OrderedDict()
        for r in reg.to_dict("records"):
            key = (r["budget"], r["solve"], float(r["shrink"]), knone(r["kappa_target"]), None if r.get("ridge_to") in (None, "None") or isnan(r.get("ridge_to")) else r["ridge_to"])
            seen.setdefault(key, []).append(r)
        for key, rs in seen.items():
            byA = {float(r["alpha"]): r for r in rs}
            ba = max(byA, key=lambda a: byA[a]["pct"])
            r0 = rs[0]
            rows.append([key[0], key[1], gen(key[2]), kcell(key[3]), "-" if key[4] is None else key[4]] + [nm(byA[a]["pct"]) if a in byA else "-" for a in (0.8, 1.0, 1.3)] +
                         [gen(ba), nm(byA[ba]["pct"]), sci(r0["median_kappa_G"], 3), sci(r0["max_kappa_G"], 3), sci(r0["median_kappa_solve"], 3), sci(r0["median_rho"], 3)])
        tab(["budget", "solve", "shrink", "kappa_target", "ridge_to", "avg_acc a=0.8 (abs. %)", "avg_acc a=1.0", "avg_acc a=1.3", "best alpha", "best avg_acc", "median_kappa_G", "max_kappa_G", "median_kappa_solve", "median_rho"], rows)
        cover(src_aa(f), len(d), sum(1 for r_ in rows for c_ in r_[5:8] if c_ != "-") + len(u), "C6b", "every (config, alpha) accuracy shown + the uniform rows")
        best_tables[f] = seen

    # ---------------------------------------------------------------- C6c doc-style tables recomputed
    h3("C6c. The AA_SIDE_REPORT section 9 tables recomputed from the raw dev files (C budget unless stated; best dev accuracy over alpha, in %)")
    kn = data["regmean_knots_release_dev.jsonl"]
    rows = []
    doc9 = {("C", "full"): (65.3, 4.1, 64.8, 64.3, 65.5, 64.6, 61.6, 50.9), ("C", "sub"): (57.9, None, 57.8, 57.4, 57.9, 57.9, 57.7, None),
            ("Bp", "full"): (51.8, None, 51.7, 51.7, 51.8, 51.6, 51.6, None), ("B0", "full"): (52.3, None, None, None, None, None, None, None)}
    for (bud, sol) in (("C", "full"), ("C", "sub"), ("Bp", "full"), ("B0", "full")):
        pl = rm_best(kn, budget=bud, solve=sol)
        s95 = rm_best(kn, budget=bud, solve=sol, shrink=0.95)
        s90 = rm_best(kn, budget=bud, solve=sol, shrink=0.9)
        k4 = rm_best(kn, budget=bud, solve=sol, kt=1e4, rt="zero")
        k3 = rm_best(kn, budget=bud, solve=sol, kt=1e3, rt="zero")
        k2 = rm_best(kn, budget=bud, solve=sol, kt=1e2, rt="zero")
        rows.append([f"{bud}, {sol}", f"{nm(pl[1])} (a={pl[0]:g}); a=1.3: {nm(pl[2][1.3])}", f"{nm(s95[1])} / {nm(s90[1])}", f"{nm(k4[1])} / {nm(k3[1])} / {nm(k2[1])}; a=1.3 (1e4): {nm(k4[2][1.3])}"])
        d9 = doc9[(bud, sol)]
        tag = f"C6c-{bud}-{sol}"
        check(tag + "-plain", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} plain best", d9[0], pl[1], 1)
        if d9[1] is not None:
            check(tag + "-plain13", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} plain at alpha 1.3", d9[1], pl[2][1.3], 1)
        if d9[2] is not None:
            check(tag + "-s95", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} shrink 0.95", d9[2], s95[1], 1)
            check(tag + "-s90", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} shrink 0.9", d9[3], s90[1], 1)
            check(tag + "-k4", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} kcap 1e4 ->zero", d9[4], k4[1], 1)
            check(tag + "-k3", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} kcap 1e3 ->zero", d9[5], k3[1], 1)
            check(tag + "-k2", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} kcap 1e2 ->zero", d9[6], k2[1], 1)
        if d9[7] is not None:
            check(tag + "-k4-13", "AA_SIDE_REPORT s9", f"KnOTS dev {bud}/{sol} kcap 1e4 at alpha 1.3", d9[7], k4[2][1.3], 1)
    ub = float(kn[kn.method == "uniform"]["pct"].max())
    rows.append(["B0 = averaging (sanity)", f"B0/full plain: {nm(rm_best(kn, budget='B0', solve='full')[1])}; uniform best {nm(ub)}", "", ""])
    check("C6c-uniform", "AA_SIDE_REPORT s9", "KnOTS dev uniform best (alpha 0.8)", 52.3, ub, 1)
    para("**C6c-1. KnOTS bank, dev (section 9 main table)**")
    tab(["budget, solve", "plain best (alpha); acc at alpha 1.3", "shrink a = 0.95 / 0.9", "kappa-cap 1e4 / 1e3 / 1e2 (ridge->zero); alpha 1.3 value for 1e4"], rows)

    # calibration-size table
    para("**C6c-2. Calibration-size test (C budget, full solve)**")
    sets = [("512", "regmean_knots_release_dev.jsonl"), ("64 (first 64)", "regmean_knots_release_n64_dev.jsonl"), ("16 s0 (first 16)", "regmean_knots_release_n16_dev.jsonl"),
            ("16 s1", "regmean_knots_release_n16_s1_dev.jsonl"), ("16 s2", "regmean_knots_release_n16_s2_dev.jsonl")]
    rows = []
    cal = {}
    for lab, f in sets:
        d = data[f]
        pl = rm_best(d)
        s95 = rm_best(d, shrink=0.95)
        k4 = rm_best(d, kt=1e4, rt="zero")
        k4m = rm_best(d, kt=1e4, rt="mean")
        mk = d[(d.method == "regmean") & (d.budget == "C") & (d.solve == "full")]["median_kappa_G"].iloc[0]
        cal[lab] = (pl, s95, k4, mk)
        rows.append([lab, nm(pl[1]), "/".join(nm(pl[2][a]) for a in (0.8, 1.0, 1.3)), nm(s95[1]), nm(k4[1]), nm(k4m[1]), sg(k4[1] - s95[1]), sci(mk, 3)])
    tab(["N (set)", "plain best", "plain at alpha 0.8/1.0/1.3", "shrink 0.95 best", "kappa-cap 1e4 ->zero best", "kappa-cap 1e4 ->mean best", "cap(->zero) - shrink (pt)", "median kappa2(G)"], rows)
    check("C6c-cal512-plain", "AA_SIDE_REPORT s9", "N = 512 plain", 65.3, cal["512"][0][1], 1)
    check("C6c-cal512-s95", "AA_SIDE_REPORT s9", "N = 512 shrink 0.95", 64.8, cal["512"][1][1], 1)
    check("C6c-cal512-k4", "AA_SIDE_REPORT s9", "N = 512 kcap 1e4", 65.5, cal["512"][2][1], 1)
    check("C6c-cal512-kG", "AA_SIDE_REPORT s9", "N = 512 median kappa(G) 1.05e5", 1.05, cal["512"][3] / 1e5, 2, "in units of 1e5")
    check("C6c-cal16-plain", "AA_SIDE_REPORT s9", "N = 16 (s0) plain = 4.1 at every alpha (best over alpha)", 4.1, cal["16 s0 (first 16)"][0][1], 1)
    check("C6c-cal16-plain-all", "AA_SIDE_REPORT s9 / REPORT_02", "N = 16 plain collapses at EVERY alpha in all 3 subsets: max over alpha and subsets", 4.1,
          max(max(cal[l][0][2].values()) for l in ("16 s0 (first 16)", "16 s1", "16 s2")), 1)
    check("C6c-cal16-s95", "AA_SIDE_REPORT s9", "N = 16 (s0) shrink 0.95", 61.7, cal["16 s0 (first 16)"][1][1], 1)
    check("C6c-cal16-k4", "AA_SIDE_REPORT s9", "N = 16 (s0) kcap 1e4", 62.2, cal["16 s0 (first 16)"][2][1], 1)
    check("C6c-cal16-kG", "AA_SIDE_REPORT s9", "N = 16 (s0) median kappa(G) 2.2e5", 2.2, cal["16 s0 (first 16)"][3] / 1e5, 1, "in units of 1e5")

    # all settings
    para("**C6c-3. All LoRA settings, matched single configs (C budget, full solve, kappa-cap 1e4 -> zero vs shrink 0.95, best alpha each; plain shown for reference)**")
    settings = [("KnOTS, N = 512", "regmean_knots_release_dev.jsonl", 0.7, 65.3, 64.8, 65.5), ("KnOTS, N = 64", "regmean_knots_release_n64_dev.jsonl", 0.8, 4.1, 63.9, 64.7),
                ("KnOTS, N = 16 s0", "regmean_knots_release_n16_dev.jsonl", 0.5, 4.1, 61.7, 62.2), ("KnOTS, N = 16 s1", "regmean_knots_release_n16_s1_dev.jsonl", 0.9, 4.1, 63.2, 64.1),
                ("KnOTS, N = 16 s2", "regmean_knots_release_n16_s2_dev.jsonl", 0.7, 4.1, 63.8, 64.5), ("lora_c5a, N = 512", "regmean_lora_c5a_dev.jsonl", 0.5, 4.1, 65.1, 65.6),
                ("vit20 (lora), N = 512", "regmean_lora_dev.jsonl", 0.2, 4.1, 82.4, 82.6)]
    rows = []
    diffs = []
    diffs_r = []
    for lab, f, dd, pl_doc, s95_doc, k4_doc in settings:
        d = data[f]
        pl = rm_best(d)
        s95 = rm_best(d, shrink=0.95)
        k4 = rm_best(d, kt=1e4, rt="zero")
        ub_ = float(d[d.method == "uniform"]["pct"].max())
        diff = k4[1] - s95[1]
        diffs.append(diff)
        diffs_r.append(round(k4[1], 1) - round(s95[1], 1))
        rows.append([lab, nm(ub_), nm(pl[1]) + f" (a={pl[0]:g}); a=1.3: {nm(pl[2][1.3])}", f"{nm(s95[1])} (a={s95[0]:g})", f"{nm(k4[1])} (a={k4[0]:g})", sg(diff)])
        check(f"C6c-set-{lab}-diff", "AA_SIDE_REPORT s9 (matched single configs table)", f"{lab}: kappa-cap minus shrink, difference of the 1-decimal values as the doc prints it", dd, diffs_r[-1], 1,
              f"unrounded difference {diff:+.3f}")
        check(f"C6c-set-{lab}-s95", "AA_SIDE_REPORT s9", f"{lab}: shrink 0.95", s95_doc, s95[1], 1)
        check(f"C6c-set-{lab}-k4", "AA_SIDE_REPORT s9", f"{lab}: kappa-cap 1e4", k4_doc, k4[1], 1)
    tab(["setting", "best uniform", "plain best (alpha); at alpha 1.3", "shrink 0.95 best", "kappa-cap 1e4 ->zero best", "cap - shrink (pt)"], rows)
    para(f"Positive in {sum(1 for x in diffs if x > 0)}/{len(diffs)} settings; mean difference {np.mean(diffs):+.2f} pt (range {min(diffs):+.2f} to {max(diffs):+.2f}). The settings share banks and gram sets, so they are not independent draws.")
    check("C6c-7of7", "AA_SIDE_REPORT s9 / REPORT_02", "kappa-cap beats shrink 0.95 in 7/7 settings", 7, sum(1 for x in diffs if x > 0), 0)
    check("C6c-mean", "AA_SIDE_REPORT s9 / REPORT_02", "mean difference over the 7 settings (+0.6)", 0.6, float(np.mean(diffs)), 1,
          f"the doc's +0.6 is the mean of the 1-decimal differences ({np.mean(diffs_r):.3f}); the unrounded mean is {np.mean(diffs):+.3f} (rounds to +0.7)")
    check("C6c-unrounded-range", "AA_SIDE_REPORT s9", "largest single-setting difference in the table (+0.9, N = 16 s1)", 0.9, max(diffs), 1, f"unrounded range {min(diffs):+.3f} ... {max(diffs):+.3f}")

    # rank-16
    r16 = data["regmean_knots_release_r16_dev.jsonl"]
    u16 = float(r16[r16.method == "uniform"]["pct"].max())
    p16 = rm_best(r16)
    s16 = rm_best(r16, shrink=0.95)
    k16 = rm_best(r16, kt=1e4, rt="zero")
    para(f"**C6c-4. Matched final-rank budget (KnOTS dev, every merged map truncated to rank 16 = one adapter)**: uniform {nm(u16)} (best alpha); RegMean plain {nm(p16[1])} (a={p16[0]:g}); shrink 0.95 {nm(s16[1])} (a={s16[0]:g}); "
         f"kappa-cap 1e4 ->zero {nm(k16[1])} (a={k16[0]:g}). Exact (rank <= 128) versions: plain {nm(rm_best(kn)[1])}, kappa-cap {nm(rm_best(kn, kt=1e4, rt='zero')[1])}: extra rank accounts for "
         f"{nm(rm_best(kn, kt=1e4, rt='zero')[1] - k16[1])} pt (kappa-cap) / {nm(rm_best(kn)[1] - p16[1])} pt (plain); at matched rank the kappa-cap is {sg(k16[1] - u16)} pt over uniform and {sg(k16[1] - s16[1])} over shrink 0.95.")
    check("C6c-r16-uni", "AA_SIDE_REPORT s9", "matched rank: uniform", 51.3, u16, 1)
    check("C6c-r16-plain", "AA_SIDE_REPORT s9", "matched rank: RegMean plain", 61.4, p16[1], 1)
    check("C6c-r16-s95", "AA_SIDE_REPORT s9", "matched rank: shrink 0.95", 61.2, s16[1], 1)
    check("C6c-r16-k4", "AA_SIDE_REPORT s9", "matched rank: kappa-cap 1e4", 61.7, k16[1], 1)
    check("C6c-extra-rank", "AA_SIDE_REPORT s9", "extra rank accounts for about 3.8 pt of RegMean's gain (kappa-cap exact minus kappa-cap rank-16)", 3.8, rm_best(kn, kt=1e4, rt="zero")[1] - k16[1], 1,
          "plain exact minus plain rank-16 would be %.2f" % (rm_best(kn)[1] - p16[1]))
    check("C6c-over-uniform", "AA_SIDE_REPORT s9", "at matched rank still about +10 pt over uniform", 10, k16[1] - u16, 0)
    ks = [(rm_best(data[f], kt=1e4, rt="zero")[1] - float(data[f][data[f].method == "uniform"]["pct"].max())) for f in ("regmean_knots_release_dev.jsonl", "regmean_lora_c5a_dev.jsonl", "regmean_lora_dev.jsonl")]
    check("C6c-12-15-lo", "AA_SIDE_REPORT s9", "RegMean-C kappa-cap over uniform on LoRA: smallest of the 3 banks (+12 ... +15 pt)", 12, min(ks), 0)
    check("C6c-12-15-hi", "AA_SIDE_REPORT s9", "RegMean-C kappa-cap over uniform on LoRA: largest of the 3 banks", 15, max(ks), 0)

    # Bp (label-free pooled Gram) vs uniform per bank
    for bank_, f_ in (("knots_release", "regmean_knots_release_dev.jsonl"), ("lora_c5a", "regmean_lora_c5a_dev.jsonl"), ("lora (vit20)", "regmean_lora_dev.jsonl")):
        d_ = data[f_]
        uni_ = d_[d_.method == "uniform"]
        bp_ = d_[(d_.method == "regmean") & (d_.budget == "Bp") & (d_.solve == "full")]
        ib = bp_["pct"].idxmax()
        iu = uni_["pct"].idxmax()
        gain_ = bp_.loc[ib, "pct"] - uni_.loc[iu, "pct"]
        txt_ = f"{gain_:+.2f} pt (best Bp/full {bp_.loc[ib, 'pct']:.2f} at alpha {bp_.loc[ib, 'alpha']:g} vs best uniform {uni_.loc[iu, 'pct']:.2f} at alpha {uni_.loc[iu, 'alpha']:g})"
        if gain_ <= 0.05:
            check_bool(f"C6-Bp-{bank_}", "AA_SIDE_REPORT s9", "'The label-free pooled (Bp) Grams carry no task information for rank-16 adapters' (stated for KnOTS: Bp full 51.8 vs uniform 52.3)",
                       "Bp - uniform about 0 or below", txt_, True)
        else:
            note_row(f"C6-Bp-{bank_}", "AA_SIDE_REPORT s9", "'The label-free pooled (Bp) Grams carry no task information for rank-16 adapters' (stated for KnOTS: Bp full 51.8 vs uniform 52.3)",
                     "Bp - uniform about 0 or below", txt_,
                     "on this bank the best Bp/full configs ARE above uniform; the statement does not generalise (alpha grid ends at 1.3, where the vit20 optimum also sits)" if "vit20" in bank_ else
                     "on this bank the best Bp/full configs are above uniform; the statement does not generalise")

    # ---------------------------------------------------------------- C6d maps
    h3("C6d. regmean_*_maps.json: per-map Gram conditioning (kappa2 of the summed Gram, at the plain a = 1 solve)")
    para("Each `regmean_<bank>[_nXX]_maps.json` stores, per (budget/solve) key, the per-map `info` of the plain (a = 1, no cap) solve: `kappa_G`, `kappa_solve` (= kappa_G, no ridge), `rho` (0), and `trace_mean` (full) or `dim` (sub). "
         "48 maps per key. Summary over the 48 maps per file and key:")
    rows = []
    for f, bank, n, sub, trunc in LORA_RM:
        mj = read_json(AA / f.replace("_dev.jsonl", "_maps.json"), src_aa(f.replace("_dev.jsonl", "_maps.json")))
        for key, maps in mj.items():
            ks = np.array([v["kappa_G"] for v in maps.values()], float)
            fin = ks[np.isfinite(ks)]
            extra = ""
            if key.endswith("/sub"):
                extra = f"dim {int(np.min([v['dim'] for v in maps.values()]))}-{int(np.max([v['dim'] for v in maps.values()]))}"
            else:
                tm = np.array([v["trace_mean"] for v in maps.values()], float)
                extra = f"trace_mean median {sci(np.median(tm), 3)}"
            rows.append([f.replace("_dev.jsonl", "_maps.json"), key, len(ks), sci(np.median(ks), 3), sci(ks.min(), 3), sci(ks.max(), 3), int(np.isinf(ks).sum()),
                         int((ks > 1e12).sum()), sci(fin.max() if len(fin) else NAN, 3), extra])
    tab(["file", "budget/solve", "# maps", "median kappa_G", "min", "max", "# inf", "# > 1e12", "max over finite", "other"], rows)
    para("**C6d-2. kappa_G of the C/full Gram by map type** (median / max over the 12 blocks; inf = numerically singular)")
    rows = []
    mods = ["q_proj", "k_proj", "v_proj", "out_proj"]
    for f, bank, n, sub, trunc in LORA_RM:
        mj = read_json(AA / f.replace("_dev.jsonl", "_maps.json"), src_aa(f.replace("_dev.jsonl", "_maps.json")))
        maps = mj["C/full"]
        row = [f.replace("_dev.jsonl", "_maps.json")]
        for m in mods:
            ks = np.array([v["kappa_G"] for p, v in maps.items() if m in p], float)
            row.append(f"{sci(np.median(ks), 2)} / {sci(ks.max(), 2)} (inf: {int(np.isinf(ks).sum())}/{len(ks)})")
        rows.append(row)
    tab(["file (C/full)"] + [f"{m}: median / max" for m in mods], rows)
    mj0 = read_json(AA / "regmean_knots_release_maps.json", "x")["C/full"]
    gname = lambda p_: f"L{p_.split('.')[2]}." + ("out_in" if "out_proj" in p_ else "attn_in")  # noqa: E731
    inf_maps = [p_.replace("encoder.layers.", "L").replace(".self_attn.", ".").replace(".weight", "") for p_, v in mj0.items() if not np.isfinite(v["kappa_G"])]
    para(f"q_proj, k_proj and v_proj of a block share one input Gram (the LN1 output, `L<l>.attn_in`), so their kappa_G are identical by construction; out_proj has its own (`L<l>.out_in`): "
         f"the {len(mj0)} maps contain only {len({gname(p_) for p_ in mj0})} distinct Grams. Maps with a numerically singular Gram (kappa_G = inf) in `regmean_knots_release_maps.json` C/full: {inf_maps}.")
    k512 = np.array([v["kappa_G"] for v in read_json(AA / "regmean_knots_release_maps.json", "x")["C/full"].values()], float)
    check("C6d-1", "AA_SIDE_REPORT s9", "KnOTS N = 512 C/full median kappa(G) ~ 1e5 and some maps singular (max inf)", "inf", sci(k512.max(), 3), None, f"median {sci(np.median(k512), 3)}", text=True)

    # ---------------------------------------------------------------- C6e 2x2
    t2 = lora_df(AA / "regmean_2x2_dev.jsonl", src_aa("regmean_2x2_dev.jsonl"))
    t2["pct"] = 100 * t2["avg_acc"]
    h3("C6e. regmean_2x2_dev.jsonl: isotropic ridge vs diagonal shrinkage x kappa-adaptive vs fixed strength (KnOTS, C, full, alpha = 0.8, dev)")
    para("Cells (`cell`): `kcap_ridge` (isotropic ridge, strength set by the kappa cap `param` = kappa_t), `kcap_shrink` (diagonal shrinkage a, the largest a meeting the cap `param` = kappa_t; "
         "`frac_cap_unreachable` = fraction of maps where even a = 0 violates the cap), `fixed_shrink` (`param` = a), `rel_ridge` (fixed relative ridge rho = `param` * tr(G)/d; `param` = c = r / kappa_t with r = the median over maps of lambda_max/(tr/d) of the N = 512 Gram). "
         "`grams` = n512 or the three N = 16 sets (n16_s0 = first 16 images, n16_s1, n16_s2 = random subsets). All rows (cells x gram sets):")
    rows = []
    for r in t2.itertuples():
        rows.append([r.grams, r.cell, gen(r.param) if r.cell != "rel_ridge" else nm(r.param, 6), "-" if r.ridge_to is None or isnan(r.ridge_to) else r.ridge_to, gen(r.alpha), nm(r.pct, 2),
                     sci(r.median_rho, 3), sci(r.median_kappa_solve, 3), nm(r.median_shrink_a, 4), nm(r.frac_cap_unreachable, 4)] + [nm(100 * r.acc[t], 1) for t in TASKS8])
    tab(["grams", "cell", "param", "ridge_to", "alpha", "avg_acc (abs. %)", "median_rho", "median_kappa_solve", "median_shrink_a", "frac_cap_unreachable"] + TASKS8, rows)
    cover(src_aa("regmean_2x2_dev.jsonl"), len(t2), len(rows), "C6e")

    def cellv(g, cell, param, rt=None):
        d = t2[(t2.grams == g) & (t2.cell == cell) & np.isclose(t2.param.astype(float), param)]
        if rt is not None:
            d = d[d.ridge_to == rt]
        assert len(d) == 1, (g, cell, param, rt, len(d))
        return float(d["pct"].iloc[0])

    c_small = float(t2[(t2.cell == "rel_ridge")]["param"].min())
    gsets16 = ["n16_s0", "n16_s1", "n16_s2"]

    def agg(cell, param, rt=None):
        return float(np.mean([cellv(g, cell, param, rt) for g in gsets16])), cellv("n512", cell, param, rt)
    res = {"ridge_ad": agg("kcap_ridge", 1e4, "zero"), "ridge_fix": agg("rel_ridge", c_small, "zero"), "diag_ad": agg("kcap_shrink", 1e4), "diag_fix": agg("fixed_shrink", 0.95)}
    para("**C6e-2. The 2x2 table** (kappa_t = 1e4 / c = %.4f ('strength matched'), fixed shrink a = 0.95; mean of the three N = 16 sets, N = 512 in brackets; all dev %% at alpha 0.8)" % c_small)
    rows = [["isotropic ridge", f"{nm(res['ridge_ad'][0])} ({nm(res['ridge_ad'][1])})", f"{nm(res['ridge_fix'][0])} ({nm(res['ridge_fix'][1])})", f"{sg(res['ridge_ad'][0] - res['ridge_fix'][0])} ({sg(res['ridge_ad'][1] - res['ridge_fix'][1])})"],
            ["diagonal shrinkage", f"{nm(res['diag_ad'][0])} ({nm(res['diag_ad'][1])})", f"{nm(res['diag_fix'][0])} ({nm(res['diag_fix'][1])})", f"{sg(res['diag_ad'][0] - res['diag_fix'][0])} ({sg(res['diag_ad'][1] - res['diag_fix'][1])})"],
            ["isotropy effect (ridge - diagonal)", f"{sg(res['ridge_ad'][0] - res['diag_ad'][0])} ({sg(res['ridge_ad'][1] - res['diag_ad'][1])})", f"{sg(res['ridge_fix'][0] - res['diag_fix'][0])} ({sg(res['ridge_fix'][1] - res['diag_fix'][1])})", ""]]
    tab(["", "kappa-adaptive strength", "fixed strength", "adaptivity effect"], rows)
    unr = t2[(t2.cell == "kcap_shrink")]
    def rng_txt(v):
        v = 100 * np.asarray(v, float)
        return f"{v.min():.1f} %" if abs(v.max() - v.min()) < 1e-9 else f"{v.min():.1f}-{v.max():.1f} %"
    para(f"`frac_cap_unreachable` for `kcap_shrink` (fraction of the 48 maps where even a = 0 violates the cap; over the 4 gram sets): kappa_t = 1e4: {rng_txt(unr[np.isclose(unr.param, 1e4)]['frac_cap_unreachable'])}; "
         f"kappa_t = 1e3: {rng_txt(unr[np.isclose(unr.param, 1e3)]['frac_cap_unreachable'])}.")
    for key, doc_n16, doc_512 in (("ridge_ad", 63.60, 65.52), ("ridge_fix", 63.47, 65.29), ("diag_ad", 62.58, 64.53), ("diag_fix", 62.86, 64.76)):
        check(f"C6e-{key}-16", "AA_SIDE_REPORT s9 2x2 table", f"2x2 {key} mean of 3 N=16 sets", doc_n16, res[key][0], 2)
        check(f"C6e-{key}-512", "AA_SIDE_REPORT s9 2x2 table", f"2x2 {key} N=512", doc_512, res[key][1], 2)
    check("C6e-iso-ad", "AA_SIDE_REPORT s9", "isotropy effect (kappa-adaptive) N=16", 1.02, res["ridge_ad"][0] - res["diag_ad"][0], 2)
    check("C6e-iso-ad-512", "AA_SIDE_REPORT s9", "isotropy effect (kappa-adaptive) N=512", 0.99, res["ridge_ad"][1] - res["diag_ad"][1], 2)
    check("C6e-iso-fx", "AA_SIDE_REPORT s9", "isotropy effect (fixed) N=16", 0.61, res["ridge_fix"][0] - res["diag_fix"][0], 2)
    check("C6e-iso-fx-512", "AA_SIDE_REPORT s9", "isotropy effect (fixed) N=512", 0.53, res["ridge_fix"][1] - res["diag_fix"][1], 2)
    check("C6e-adapt-r", "AA_SIDE_REPORT s9", "adaptivity effect, isotropic ridge, N=16", 0.13, res["ridge_ad"][0] - res["ridge_fix"][0], 2)
    check("C6e-adapt-r-512", "AA_SIDE_REPORT s9", "adaptivity effect, isotropic ridge, N=512", 0.23, res["ridge_ad"][1] - res["ridge_fix"][1], 2)
    check("C6e-adapt-d", "AA_SIDE_REPORT s9", "adaptivity effect, diagonal shrinkage, N=16", -0.28, res["diag_ad"][0] - res["diag_fix"][0], 2)
    check("C6e-adapt-d-512", "AA_SIDE_REPORT s9", "adaptivity effect, diagonal shrinkage, N=512", -0.23, res["diag_ad"][1] - res["diag_fix"][1], 2)
    ur = unr["frac_cap_unreachable"].values
    check("C6e-unreach-lo", "AA_SIDE_REPORT s9", "kappa-capped shrinkage cannot reach its target in 12-17% of maps (lower end)", 12, 100 * ur.min(), 0)
    check("C6e-unreach-hi", "AA_SIDE_REPORT s9", "kappa-capped shrinkage cannot reach its target in 12-17% of maps (upper end)", 17, 100 * ur.max(), 0,
          "over all four gram sets and both kappa_t the range is %.1f-%.1f %% (kappa_t = 1e4 alone: 12.5 %%)" % (100 * ur.min(), 100 * ur.max()), sev="minor")

    # ---------------------------------------------------------------- C6f test
    tj = read_json(AA / "regmean_test.json", src_aa("regmean_test.json"))
    h3("C6f. regmean_test.json: frozen TEST evaluation (C budget, N = 512, full solve; dev-best alpha per config frozen)")
    para("Configs: `uniform`; `shrink0.95` (RegMean with a = 0.95); `kcap1e4_zero` (kappa cap 1e4, ridge -> zero). Test = the 80% split; metric = absolute `avg_acc` and per-task `acc` (%). "
         "`kcap_minus_shrink` = `kcap1e4_zero` minus `shrink0.95` (point estimate and 95% paired bootstrap CI over test images within task, 2000 resamples; example sampling only, not calibration resampling). "
         "Alpha is verified below against the dev-best alpha of the same config in the dev file.")
    rows = []
    rows2 = []
    dev_files = {"knots_release": "regmean_knots_release_dev.jsonl", "lora_c5a": "regmean_lora_c5a_dev.jsonl", "lora": "regmean_lora_dev.jsonl"}
    for bank, bj in tj.items():
        d = data[dev_files[bank]]
        for lab, kw in (("uniform", None), ("shrink0.95", dict(shrink=0.95)), ("kcap1e4_zero", dict(kt=1e4, rt="zero"))):
            e = bj[lab]
            if kw is None:
                u = d[d.method == "uniform"]
                dev_alpha = float(u.loc[u["pct"].idxmax(), "alpha"])
                dev_acc = float(u["pct"].max())
            else:
                bb = rm_best(d, **kw)
                dev_alpha, dev_acc = bb[0], bb[1]
            rows.append([bank, lab, gen(e["alpha"]), nm(100 * e["avg_acc"], 2), nm(dev_acc, 2), gen(dev_alpha), "yes" if abs(dev_alpha - e["alpha"]) < 1e-9 else "NO"] + [nm(100 * e["acc"][t], 2) for t in TASKS8])
            assert abs(e["avg_acc"] - tasks_mean(e["acc"])) < 1e-12
        k = bj["kcap_minus_shrink"]
        rows2.append([bank, sg(100 * k["point"], 3), f"[{sg(100 * k['ci95'][0], 3)}, {sg(100 * k['ci95'][1], 3)}]",
                      sg(100 * (bj["kcap1e4_zero"]["avg_acc"] - bj["shrink0.95"]["avg_acc"]), 3), sg(100 * (bj["kcap1e4_zero"]["avg_acc"] - bj["uniform"]["avg_acc"]), 2),
                      sg(100 * (bj["shrink0.95"]["avg_acc"] - bj["uniform"]["avg_acc"]), 2)])
    para("**C6f-1. Every bank x config: frozen alpha, TEST accuracy (absolute %) and per-task accuracy**")
    tab(["bank", "config", "frozen alpha", "TEST avg_acc (abs. %)", "dev avg_acc of the same config (abs. %)", "dev-best alpha", "frozen alpha == dev-best?"] + TASKS8, rows)
    para("**C6f-2. kappa-cap minus shrink with the 95% bootstrap CI**")
    tab(["bank", "kcap_minus_shrink point (pt)", "95% CI (pt)", "recomputed from the two avg_acc (pt)", "kcap1e4_zero - uniform (pt)", "shrink0.95 - uniform (pt)"], rows2)
    cover(src_aa("regmean_test.json"), 3 * 4, len(rows) + len(rows2), "C6f-1/2", "3 banks x (uniform, shrink0.95, kcap1e4_zero, kcap_minus_shrink) = 12 entries")
    for bank, doc_pt, doc_lo, doc_hi in (("knots_release", 0.83, 0.71, 0.95), ("lora_c5a", 0.51, 0.40, 0.62), ("lora", 0.20, 0.10, 0.30)):
        k = tj[bank]["kcap_minus_shrink"]
        check(f"C6f-{bank}-pt", "AA_SIDE_REPORT s9 / task brief", f"{bank}: test kappa-cap minus shrink point", doc_pt, 100 * k["point"], 2)
        check(f"C6f-{bank}-lo", "AA_SIDE_REPORT s9", f"{bank}: CI lower", doc_lo, 100 * k["ci95"][0], 2)
        check(f"C6f-{bank}-hi", "AA_SIDE_REPORT s9", f"{bank}: CI upper", doc_hi, 100 * k["ci95"][1], 2)
    for bank, (u_, s_, k_) in {"knots_release": (51.2, 63.5, 64.3), "lora_c5a": (53.0, 64.5, 65.0), "lora": (66.7, 81.9, 82.1)}.items():
        check(f"C6f-{bank}-uni", "AA_SIDE_REPORT s9 test table", f"{bank}: test uniform", u_, 100 * tj[bank]["uniform"]["avg_acc"], 1)
        check(f"C6f-{bank}-shr", "AA_SIDE_REPORT s9 test table", f"{bank}: test RegMean + shrink 0.95", s_, 100 * tj[bank]["shrink0.95"]["avg_acc"], 1)
        check(f"C6f-{bank}-kc", "AA_SIDE_REPORT s9 test table", f"{bank}: test RegMean + kappa-cap 1e4", k_, 100 * tj[bank]["kcap1e4_zero"]["avg_acc"], 1)

    STASH.update(lora_res=res, lora_tj=tj, lora_p16=p16, lora_s16=s16, lora_k16=k16, lora_u16=u16, lora_kn=kn)

    # ---------------------------------------------------------------- C6g peer summary
    summ_path = AA / "regmean_summary.md"
    peer = summ_path.read_text()
    register(summ_path, src_aa("regmean_summary.md"), len(peer.splitlines()))
    h3("C6g. regmean_summary.md: the peer's own summary (copied verbatim), and a check that it is current")
    para(f"This is the PEER'S OWN summary file (written by `aa_summarize_regmean.py`; file mtime {time.strftime('%m-%d %H:%M', time.localtime(summ_path.stat().st_mtime))}). "
         "It is copied here unchanged. Format: per bank/gram set, one row per (budget, solve, regulariser) with the best dev accuracy over alpha (alpha in brackets), the accuracy at alpha 1.3, and median kappa(G) / kappa(solve); "
         "then per-map kappa(G) lines. Each sub-section title is the file tag.")
    para("(Copied verbatim; every line is indented by 4 spaces only so that its `##` lines are not mistaken for headings of this document.)")
    emit("\n".join("    " + l for l in peer.rstrip("\n").splitlines()))
    emit()
    regen = regen_peer_summary()
    import difflib
    diff = list(difflib.unified_diff(peer.splitlines(), regen.splitlines(), "peer regmean_summary.md", "regenerated from raw dev files", lineterm="", n=0))
    nplus = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    nminus = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    para(f"**Staleness check.** Regenerating the summary with the peer's own logic from the CURRENT raw files (the glob `regmean_*_dev.jsonl` also matches `regmean_2x2_dev.jsonl`, whose rows have no `method` key, which would crash "
         f"`aa_summarize_regmean.py` if run today; that file is skipped here) differs from the stored file by {nplus} added and {nminus} removed lines. "
         + ("Differences (unified diff, zero context):" if diff else "The stored file is identical to the regeneration."))
    if diff:
        codeblock("\n".join(diff[:400]))
    check_bool("C6g-stale", "regmean_summary.md vs raw dev files", "regmean_summary.md reflects the final dev files", "current", f"{nplus} lines would be added, {nminus} removed", nplus + nminus == 0,
               note="the stored file (16:15) predates the end of regmean_lora_dev.jsonl (16:18; the vit20 Bp rows and per-map kappa lines are missing) and regmean_knots_release_r16_dev.jsonl (16:23; no section)", sev="staleness")

    # ---------------------------------------------------------------- C6h verification
    h3("C6h. Verification that every LoRA accuracy is ABSOLUTE accuracy (avg_acc), per file")
    rows = []
    for v in VERIFY:
        if v["kind"] in ("lora", "lora-noacc"):
            rows.append([v["file"], v["n"], "-" if isnan(v["max_diff_avg"]) else f"{v['max_diff_avg']:.1e}", "-" if isnan(v["acc_min"]) else f"{100 * v['acc_min']:.2f} / {100 * v['acc_max']:.2f}",
                         "avg_acc == mean of the 8 per-task accuracies" if v["kind"] == "lora" else "no per-task acc in this file; cross-checked against files with the same config (below)"])
    para("**C6h-1. Per-file check**")
    tab(["file", "rows", "max abs diff avg_acc vs mean(acc)", "min / max per-task acc in file (%)", "result"], rows)
    para("Per-task `acc` is correct/total on that task's split (aa:src/condmerge/aa_common.py `TaskData.evaluate`: `acc[t] = correct / n`; `avg_acc = mean(acc)`); no division by an expert's accuracy anywhere in the LoRA pipeline "
         "(unlike master's `mean_norm`). Cross-file consistency of the same nominal configuration (uniform on the KnOTS release bank, dev):")
    sw = load_rule_sweep().drop_duplicates("chash")
    q9 = lora_df(AA / "q90_transfer.jsonl", "q")
    sm = read_jsonl(AA / "single_map_swap.jsonl", src_aa("single_map_swap.jsonl"))
    base_swap = [r for r in sm if r["param"] == "__base__"][0]
    rb = lora_df(AA / "replicate_banks.jsonl", "r")
    def sv(label, alpha):
        return float(sw[(sw.label == label) & (sw.alpha == alpha)]["avg_acc"].iloc[0])

    def qv(bank, alpha):
        return float(q9[(q9.bank == bank) & (q9.label == "uniform") & (q9.alpha == alpha)]["avg_acc"].iloc[0])

    def rv(bank, alpha):
        return float(rb[(rb.bank == bank) & (rb.label == "uniform") & (rb.alpha == alpha)]["avg_acc"].iloc[0])
    ru = lora_df(AA / "regmean_knots_release_dev.jsonl", "x")
    ru = ru[ru.method == "uniform"]
    pairs = [(("rule_sweep uniform a=0.8", sv("uniform", 0.8)), ("q90_transfer knots_release uniform a=0.8", qv("knots_release", 0.8))),
             (("rule_sweep uniform a=1.0", sv("uniform", 1.0)), ("q90_transfer knots_release uniform a=1.0", qv("knots_release", 1.0))),
             (("rule_sweep uniform a=1.0", sv("uniform", 1.0)), ("single_map_swap __base__ (uniform a=1.0)", float(base_swap["avg_acc"]))),
             (("replicate_banks lora_c5a uniform a=0.8", rv("lora_c5a", 0.8)), ("q90_transfer lora_c5a uniform a=0.8", qv("lora_c5a", 0.8))),
             (("replicate_banks lora uniform a=1.0", rv("lora", 1.0)), ("q90_transfer lora uniform a=1.0", qv("lora", 1.0))),
             (("rule_sweep uniform a=0.8", sv("uniform", 0.8)), ("regmean_knots_release_dev uniform a=0.8 (mean of dense updates)", float(ru[ru.alpha == 0.8]["avg_acc"].iloc[0])))]
    for bk_, f_ in (("lora_c5a", "regmean_lora_c5a_dev.jsonl"), ("lora", "regmean_lora_dev.jsonl")):
        du_ = lora_df(AA / f_, "x")
        du_ = du_[du_.method == "uniform"]
        for a_ in (0.8, 1.0):
            pairs.append(((f"replicate_banks {bk_} uniform a={a_:g}", rv(bk_, a_)), (f"{f_} uniform a={a_:g}", float(du_[du_.alpha == a_]["avg_acc"].iloc[0]))))
    rows = [[a[0], nm(100 * a[1], 4), b_[0], nm(100 * b_[1], 4), f"{abs(a[1] - b_[1]):.1e}"] for a, b_ in pairs]
    mxpair = max(abs(a[1] - b_[1]) for a, b_ in pairs)
    para("**C6h-2. Cross-file consistency**")
    tab(["config in file A", "A (abs. %)", "same config in file B", "B (abs. %)", "abs diff (fraction)"], rows)
    para(f"Differences of order 1e-5 to 1e-3 between files for the same nominal uniform merge are most likely numerical (the uniform update is built by different code paths: a float32 weighted sum in the sweep scripts, a float64 mean cast to float32 in the RegMean scripts, evaluated in fp16, so a few images flip); this is an inference from the scripts, not something the files record. It is not a different metric: the largest difference above is {mxpair:.1e} (fraction), i.e. {100 * mxpair:.3f} pt.")


def regen_peer_summary() -> str:
    """The peer's aa_summarize_regmean.py logic, applied to the current raw dev files (2x2 file skipped)."""
    lines = []
    for fp in sorted(AA.glob("regmean_*_dev.jsonl")):
        if "2x2" in fp.name:
            continue
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
        mp = AA / f"regmean_{tag}_maps.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            lines.append("")
            for key, maps in m.items():
                ks = np.array([v["kappa_G"] for v in maps.values()])
                lines.append(f"- per-map kappa(G) {key}: median {np.median(ks):.2e}, min {ks.min():.2e}, max {ks.max():.2e}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---- doc checks that need values from several sections ----------------------------------------------------------------
def sec_docs():
    two, mean_dev, sel, sd_idx, inst_t, inst_mean = (STASH[k] for k in ("two", "mean_dev", "sel", "sd_idx", "inst_t", "inst_mean"))
    stest, sdev, edge_dev = STASH["stest"], STASH["sdev"], STASH["edge_dev"]
    sweep_t = STASH["sweep_t"]
    res, tj = STASH["lora_res"], STASH["lora_tj"]
    D = "REPORT_02 s3-s4"
    fr = read_jsonl(MASTER / "results" / "freeze" / "test.jsonl", "master:results/freeze/test.jsonl")
    uni = [r for r in fr if r["cfg"].get("tag") == "uniform|tuned"][0]
    check("D-uni-dev", "REPORT_02 s2", "uniform average, alpha tuned: dev mean_norm", 77.65, 100 * uni["dev_metric"], 2, "master:results/freeze/test.jsonl tag uniform|tuned")
    check("D-uni-test", "REPORT_02 s2", "uniform average, alpha tuned: test mean_norm", 76.68, 100 * uni["metrics"]["mean_norm"], 2)
    check("D-uni-test-acc", "REPORT_02 s2", "uniform average, alpha tuned: test mean_acc", 69.29, 100 * uni["metrics"]["mean_acc"], 2)
    check("D-gain", "REPORT_02 s5.5", "gain of task-separated RegMean over uniform on full FT test (+14.7 normalised)", 14.7, STASH["t_plain"]["norm"] - 100 * uni["metrics"]["mean_norm"], 1)
    # LoRA frozen test (REPORT_02 s2)
    for bank, (u_, s_) in {"knots_release": (51.22, 63.47), "lora_c5a": (53.04, 64.45), "lora": (66.68, 81.89)}.items():
        check(f"D-lora-test-{bank}-uni", "REPORT_02 s2 LoRA table", f"{bank}: frozen test uniform (2 decimals)", u_, 100 * tj[bank]["uniform"]["avg_acc"], 2)
        check(f"D-lora-test-{bank}-shr", "REPORT_02 s2 LoRA table", f"{bank}: frozen test RegMean shrink 0.95 (2 decimals)", s_, 100 * tj[bank]["shrink0.95"]["avg_acc"], 2)
    check_array("D-FR-16.4", "FULL_REPORT s16.4 table", "LoRA frozen test (N 512): uniform / shrinkage 0.95 / kappa cap 1e4 for KnOTS, lora_c5a, vit20 (2 decimals)",
                [51.22, 63.47, 64.30, 53.04, 64.45, 64.96, 66.68, 81.89, 82.09],
                [100 * tj[b][c]["avg_acc"] for b in ("knots_release", "lora_c5a", "lora") for c in ("uniform", "shrink0.95", "kcap1e4_zero")], 2)
    check_range("D-r16-range", "REPORT_02 s2", "rank-16 truncation (KnOTS dev): '61.2-61.7 vs 51.3 for uniform'", "61.2 to 61.7", 61.2, 61.7,
                [100 * 0 + STASH["lora_s16"][1], STASH["lora_p16"][1], STASH["lora_k16"][1]], 1)
    # 2x2 full FT test
    tt = two[("test", 1e5)]
    cells = ("kcap_ridge", "rel_ridge", "kcap_shrink", "fixed_shrink")
    doc = {("kcap_ridge", 16): 87.69, ("kcap_ridge", 512): 91.40, ("rel_ridge", 16): 86.90, ("rel_ridge", 512): 91.38, ("kcap_shrink", 16): 87.28, ("kcap_shrink", 512): 90.59,
           ("fixed_shrink", 16): 87.39, ("fixed_shrink", 512): 91.07}
    check_array("D-2x2-test-cells", D, "full FT 2x2, test: 4 cells x (n = 16 mean, n = 512): ridge-adaptive, ridge-fixed, shrink-adaptive, shrink-fixed", [doc[(c, n)] for c in cells for n in (16, 512)],
                [tt[n]["mean"][c] for c in cells for n in (16, 512)], 2)
    eff_keys = list(tt[16]["eff"].keys())
    doc_eff = {16: [0.41, -0.49, 0.78, -0.11], 512: [0.81, 0.30, 0.03, -0.48]}
    for n in (16, 512):
        check_array(f"D-2x2-test-eff-{n}", D, f"full FT 2x2, test, n = {n}: isotropy(adaptive), isotropy(fixed), adaptivity(ridge), adaptivity(shrink)", doc_eff[n], [tt[n]["eff"][k] for k in eff_keys], 2)
    for lab_, kt, nm_ in (("test", 1e5, "test"), ("dev", 1e5, "dev")):
        r_ = two[(lab_, kt)]
        docs_ = {"test": {"kcap_ridge": ([0.19, 0.43, 0.27], 0.33), "kcap_shrink": ([-0.18, -0.04, -0.12], -0.48), "rel_ridge": ([-0.45, -0.49, -0.52], 0.30)},
                 "dev": {"kcap_ridge": ([0.11, 0.32, 0.26], 0.27), "kcap_shrink": ([-0.29, 0.07, -0.14], -0.22), "rel_ridge": ([-0.56, -0.66, -0.49], 0.12)}}[lab_]
        for cell, (d16, d512) in docs_.items():
            check_array(f"D-edge-{lab_}-{cell}", D, f"edge over cell 3 (a = 0.95) on {lab_}: {cell}, n = 16 subsets 0/1/2 then n = 512", d16 + [d512], list(r_[16]["diff"][cell]) + list(r_[512]["diff"][cell]), 2)
    t4 = two[("test", 1e4)]
    check("D-1e4-16", D, "kappa_t = 1e4 diagonal shrinkage vs a = 0.95, test, n = 16 (-1.00)", -1.00, float(t4[16]["diff"]["kcap_shrink"].mean()), 2)
    check("D-1e4-512", D, "kappa_t = 1e4 diagonal shrinkage vs a = 0.95, test, n = 512 (-1.92)", -1.92, float(t4[512]["diff"]["kcap_shrink"].mean()), 2)
    # strength sweep tables
    kts = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    shs = [0.5, 0.8, 0.9, 0.95, 0.99]
    rho16 = lambda arm, p: float(np.median([sd_idx[(arm, p)][16][s]["rel_rho_median"] for s in sd_idx[(arm, p)][16]]))  # noqa: E731
    crel16 = lambda p: float(np.median([sd_idx[("rel_ridge", p)][16][s]["c_rel"] for s in sd_idx[("rel_ridge", p)][16]]))  # noqa: E731
    check_array("D-sw-kcap16", "REPORT_02 s4 table", "kappa-cap ridge, n = 16 (dev mean_norm over the 7 grid values)", [86.75, 87.77, 88.11, 88.32, 88.37, 88.08, 87.57], [mean_dev[("kcap_ridge", k, 16)] for k in kts], 2)
    check_array("D-sw-rel16", "REPORT_02 s4 table", "fixed ridge, n = 16", [87.08, 87.75, 88.02, 87.96, 87.58, 86.84, 85.47], [mean_dev[("rel_ridge", k, 16)] for k in kts], 2)
    check_array("D-sw-kcap512", "REPORT_02 s4 table", "kappa-cap ridge, n = 512", [90.07, 91.09, 91.32, 91.58, 91.58, 91.51, 91.56], [mean_dev[("kcap_ridge", k, 512)] for k in kts], 2)
    check_array("D-sw-rel512", "REPORT_02 s4 table", "fixed ridge, n = 512", [90.38, 91.06, 91.33, 91.44, 91.43, 91.51, 91.52], [mean_dev[("rel_ridge", k, 512)] for k in kts], 2)
    check_array("D-sw-shr16", "REPORT_02 s4 table", "fixed shrinkage, n = 16", [85.58, 87.16, 87.76, 88.14, 88.14], [mean_dev[("fixed_shrink", a, 16)] for a in shs], 2)
    check_array("D-sw-shr512", "REPORT_02 s4 table", "fixed shrinkage, n = 512", [88.00, 90.43, 91.10, 91.31, 91.55], [mean_dev[("fixed_shrink", a, 512)] for a in shs], 2)
    sig2 = lambda x: float(f"{x:.2g}")  # noqa: E731
    check_array("D-sw-rho", "REPORT_02 s4 table", "'its median rho / (tr G/d)' row of the kappa-cap arm (n = 16, 2 significant digits)", [0.11, 0.036, 0.011, 3.5e-3, 8.4e-4, 0, 0],
                [sig2(rho16("kcap_ridge", k)) for k in kts], 4, sci_fmt=True)
    check_array("D-sw-c-as-rho", "REPORT_02 s4 table", "'its c' row of the fixed ridge, vs the logged rel_rho_median (n = 16, 2 significant digits)", [0.14, 0.051, 0.016, 5.3e-3, 1.6e-3, 5.4e-4, 1.6e-4],
                [sig2(rho16("rel_ridge", k)) for k in kts], 4, sci_fmt=True)
    ok_c = check_array("D-sw-c-true", "REPORT_02 s4 table / s4 text", "'its c' row of the fixed ridge, vs the true c = r_med/kappa_t (`c_rel`, median over the n = 16 sets; the same section says 'c from 0.16 down to 1.6e-4')",
                       [0.14, 0.051, 0.016, 5.3e-3, 1.6e-3, 5.4e-4, 1.6e-4], [sig2(crel16(k)) for k in kts], 4,
                       note="the table row equals rel_rho_median = rho/trace_mean(after adding the ridge) = c/(1+c), which understates c when c is large (0.139 vs c = 0.161 at kappa_t = 1e3)", sev="minor", sci_fmt=True)
    # ties
    check("D-tie-1", "REPORT_02 s4", "n = 512 kappa cap tie: 3e4 = 91.5752", 91.5752, mean_dev[("kcap_ridge", 3e4, 512)], 4)
    check("D-tie-2", "REPORT_02 s4", "n = 512 kappa cap tie: 1e5 = 91.5758", 91.5758, mean_dev[("kcap_ridge", 1e5, 512)], 4)
    check("D-tie-3", "REPORT_02 s4", "n = 16 shrinkage tie: 0.95 = 88.144", 88.144, mean_dev[("fixed_shrink", 0.95, 16)], 3)
    check("D-tie-4", "REPORT_02 s4", "n = 16 shrinkage tie: 0.99 = 88.137", 88.137, mean_dev[("fixed_shrink", 0.99, 16)], 3)
    check("D-rmed", "REPORT_02 s4", "r_med ranges 158-161 (lambda_max/(tr/d) median over maps) - min over the 4 Gram sets", 158, float(sdev["lambda_ratio_median"].min()), 0)
    check("D-rmed-hi", "REPORT_02 s4", "r_med ranges 158-161 - max", 161, float(sdev["lambda_ratio_median"].max()), 0)
    tdev_ = master_df(MA / "twobytwo_C_dev.jsonl", "x", "dev")
    ttest_ = master_df(MA / "twobytwo_C_test.jsonl", "x", "test")
    npair = {"dev": 0, "test": 0}
    for lab, a, b in (("dev", tdev_, sdev), ("test", ttest_, stest)):
        ka = {(r.cell, None if isnan(r.kappa_t) else float(r.kappa_t), None if isnan(r.shrink) else float(r.shrink), int(r.n), int(r.seed)) for r in a.itertuples()}
        for r in b.itertuples():
            if (r.cell, None if isnan(r.kappa_t) else float(r.kappa_t), None if isnan(r.shrink) else float(r.shrink), int(r.n), int(r.seed)) in ka:
                npair[lab] += 1
    check("D-34", "REPORT_02 s4", "'The 34 configurations that repeat the pre-declared run reproduce it exactly' (dev + test)", 34, npair["dev"] + npair["test"], 0, f"dev {npair['dev']} + test {npair['test']}; RESEARCH_LOG 22:45 says 24 (dev only)")
    check("D-med-rho-cap", "REPORT_02 s3", "cap's actual median strength 8.4e-4 of tr(G)/d (n = 16, kappa_t 1e5)", 8.4e-4, rho16("kcap_ridge", 1e5), 5)
    check("D-med-rho-fixed", "REPORT_02 s3", "fixed ridge median strength 1.6e-3 (n = 16, c = r_med/1e5)", 1.6e-3, rho16("rel_ridge", 1e5), 4)
    # frozen test table and differences (REPORT_02 s4, RESEARCH_LOG 22:50)
    tm = STASH["test_mean"]
    check_array("D-test-n16-mean", "REPORT_02 s4 / RESEARCH_LOG 22:50", "frozen test n = 16 means: kappa-cap ridge, fixed ridge, shrinkage", [87.69, 87.51, 87.39], [tm[("kcap_ridge", 16)], tm[("rel_ridge", 16)], tm[("fixed_shrink", 16)]], 2)
    check_array("D-test-n512", "REPORT_02 s4 / RESEARCH_LOG 22:50", "frozen test n = 512: kappa-cap ridge, fixed ridge, shrinkage", [91.40, 91.37, 91.33], [tm[("kcap_ridge", 512)], tm[("rel_ridge", 512)], tm[("fixed_shrink", 512)]], 2)
    for key, (a1, a2), docv, docm in ((16, ("kcap_ridge", "rel_ridge"), [0.10, 0.29, 0.15], 0.18), (16, ("kcap_ridge", "fixed_shrink"), [0.19, 0.43, 0.27], 0.30),
                                      (16, ("rel_ridge", "fixed_shrink"), [0.09, 0.14, 0.12], 0.12)):
        dd = sweep_t[(key, a1, a2)]
        check_array(f"D-diff-{a1}-{a2}", "REPORT_02 s4 / RESEARCH_LOG 22:50", f"test differences n = 16: {a1} - {a2}, subsets 0/1/2", docv, list(dd), 2)
        check(f"D-diff-mean-{a1}-{a2}", "REPORT_02 s4 / RESEARCH_LOG 22:50 / SUMMARY s3 / AA_SIDE_REPORT s9 note / FULL_REPORT s16.6", f"test difference n = 16, mean over subsets: {a1} - {a2}", docm, float(dd.mean()), 2,
              "0.11449 (rounds to +0.11); the docs print +0.12, which makes the isotropy and adaptive parts (0.12 + 0.18) add to the printed +0.30 (exact parts 0.114 + 0.182 = 0.297)" if a1 == "rel_ridge" else "")
    check_array("D-diff-512", "REPORT_02 s4 / RESEARCH_LOG 22:50", "test differences n = 512: kcap - fixed ridge, kcap - shrinkage, fixed ridge - shrinkage", [0.03, 0.07, 0.04],
                [sweep_t[(512, "kcap_ridge", "rel_ridge")][0], sweep_t[(512, "kcap_ridge", "fixed_shrink")][0], sweep_t[(512, "rel_ridge", "fixed_shrink")][0]], 2)
    iso = float(sweep_t[(16, "rel_ridge", "fixed_shrink")].mean())
    ada = float(sweep_t[(16, "kcap_ridge", "rel_ridge")].mean())
    check("D-split-ada", "REPORT_02 s4 / AA_SIDE_REPORT s9 (added note)", "n = 16: edge from kappa-adaptive strength (+0.18)", 0.18, ada, 2)
    check("D-split-pct", "RESEARCH_LOG 22:50", "'about 40 % isotropy and 60 % kappa-adaptive per-map strength'", 40, 100 * iso / (iso + ada), 0, "isotropy share in %% (adaptive share is the rest): %.1f / %.1f; counted OK as 'about'" % (100 * iso / (iso + ada), 100 * ada / (iso + ada)), tol=2.0)
    allsix = [sweep_t[(512, a1, a2)][0] for a1, a2 in (("kcap_ridge", "rel_ridge"), ("kcap_ridge", "fixed_shrink"), ("rel_ridge", "fixed_shrink"))]
    plain512 = float(STASH["t_plain"]["norm"])
    spread = max(tm[("kcap_ridge", 512)], tm[("rel_ridge", 512)], tm[("fixed_shrink", 512)], plain512) - min(tm[("kcap_ridge", 512)], tm[("rel_ridge", 512)], tm[("fixed_shrink", 512)], plain512)
    check("D-512-within", "REPORT_02 s4/s5 / SUMMARY s3", "n = 512: all regularisers within 0.07 of each other and of plain RegMean (91.39)", 0.07, spread, 2, "max - min over the three frozen configs and plain RegMean")
    commits = sorted(set(stest["commit"]))
    check_bool("D-commit", "RESEARCH_LOG 22:50", "'All test records carry commit 6ae7a69 (clean tree)'", "6ae7a69", ", ".join(commits), commits == ["6ae7a69"], note="commit stamps of the 12 strength_sweep test rows", sev="provenance")
    # SUMMARY s3
    t_plain = STASH["t_plain"]
    check("D-sum-1", "SUMMARY s3", "RegMean task-separated: normalised test 91.4", 91.4, t_plain["norm"], 1)
    check("D-sum-2", "SUMMARY s3", "RegMean task-separated: accuracy 82.8 %", 82.8, t_plain["acc"], 1)
    check("D-sum-3", "SUMMARY s3", "uniform/TA test normalised 76.7", 76.7, 100 * uni["metrics"]["mean_norm"], 1)
    c2 = [two[("test", 1e5)][16]["diff"]["kcap_shrink"].min(), two[("test", 1e5)][16]["diff"]["kcap_shrink"].max(), float(two[("test", 1e5)][512]["diff"]["kcap_shrink"][0])]
    check_range("D-sum-4", "SUMMARY s3", "diagonal shrinkage set by a kappa cap 'never helps (-0.04 to -0.48)' (test, kappa_t = 1e5, per subset and n = 512)", "-0.48 to -0.04", -0.48, -0.04, c2, 2)
    kcs = list(two[("test", 1e5)][16]["diff"]["kcap_ridge"])
    check_range("D-sum-5", "SUMMARY s3", "full FT kappa-cap edge on test '+0.2 to +0.4' (3 subsets)", "+0.2 to +0.4", 0.2, 0.4, kcs, 1)
    lo = min(100 * (tj[b]["kcap_minus_shrink"]["point"]) for b in tj)
    hi = max(100 * (tj[b]["kcap_minus_shrink"]["point"]) for b in tj)
    check_range("D-sum-6", "SUMMARY s3 / REPORT_02 s5", "LoRA kappa-cap edge on test '+0.2 to +0.8' (3 banks)", "+0.2 to +0.8", 0.2, 0.8, [lo, hi], 1)
    k16max = float(STASH["idev"][(STASH["idev"].method == "kcap_mean") & (STASH["idev"].n == 16)]["kappa_before_max"].max())
    check_bool("D-sum-7", "SUMMARY s3", "'kappa up to 1e13' at few calibration images", "1e13", f"{k16max:.3e}", 0.5 <= k16max / 1e13 <= 5, note="same order of magnitude (see C5-kG-max16 for the 1.6e13 figure)")


# ---- final: cross-check table, manifest, notes -----------------------------------------------------------------------
def sec_final():
    for dp in (MASTER / "docs" / "REPORT_02_inverse_problem.md", MASTER / "docs" / "RESEARCH_LOG.md", MASTER / "docs" / "SUMMARY.md", MASTER / "docs" / "FULL_REPORT.md", AAREPO / "docs" / "aa" / "AA_SIDE_REPORT.md"):
        register(dp, ("master:docs/" if "condmerge/" in str(dp) and "condmerge_aa" not in str(dp) else "aa:docs/aa/") + dp.name, len(dp.read_text().splitlines()))
    r2 = (MASTER / "docs" / "REPORT_02_inverse_problem.md").read_text()
    log = (MASTER / "docs" / "RESEARCH_LOG.md").read_text()
    if "PENDING" in r2.split("## 3.")[-1].split("## 4.")[0]:
        note_row("X-doc-1", "REPORT_02 s3 vs RESEARCH_LOG 22:40", "REPORT_02 section 3 (2x2 mechanism check)", "'PENDING'", "twobytwo_C_dev/test.jsonl (32 rows each) and the strength sweep exist",
                 "section 3 still reads PENDING")
    last_log = re.findall(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2})[: ]+([^\n]*)", log, flags=re.M)[-1]
    note_row("X-doc-2", "RESEARCH_LOG", "last entry of the log at generation time", f"{last_log[0]}: {last_log[1]}", "strength_sweep_C_test_* complete (4 files x 3 rows)",
             "docs were still being edited while this appendix was built (modification times of the docs listed in X4: %s); the checks use the versions listed in X4" %
             ", ".join(sorted({m["mtime"][-8:-3] for m in MANIFEST if m["label"].split("/")[0] in ("master:docs", "aa:docs")})))
    note_row("X-doc-3", "scripts/aa_summarize_regmean.py", "glob `regmean_*_dev.jsonl`", "summarises per-bank dev files", "also matches regmean_2x2_dev.jsonl (rows have no `method` key)",
             "re-running the summarizer today would raise KeyError; the stored regmean_summary.md was produced before that file existed")
    h2("X1. Cross-check of the docs against the recomputed numbers")
    para("Every row compares one number quoted in a doc (`doc value`, typed from the doc text) with the value recomputed here from the raw files (`recomputed`). Numeric rows are OK when the recomputed value rounds to the doc value "
         "at the doc's own precision (tolerance = half a unit in the last printed digit). `DISCREPANCY` rows are listed again below in X2.")
    rows = [[c["id"], c["doc"], c["claim"], c["doc_val"], c["new_val"], c["status"] + (f" ({c['sev']})" if c["sev"] else ""), c["note"]] for c in CHECKS]
    tab(["id", "doc", "claim", "doc value", "recomputed", "status", "note"], rows, ["l", "l", "l", "r", "r", "l", "l"])
    n_ok = sum(1 for c in CHECKS if c["status"] == "OK")
    n_bad = sum(1 for c in CHECKS if c["status"] == "DISCREPANCY")
    n_note = sum(1 for c in CHECKS if c["status"] == "NOTE")
    para(f"{len(CHECKS)} rows: {n_ok} OK, {n_bad} DISCREPANCY, {n_note} NOTE (observation qualifying a doc statement).")
    h2("X2. Discrepancies and caveats")
    para("Severity: `minor` = within 1.5 units of the last digit the doc prints (rounding or a typo of one unit); `material` = larger; `wording/structure` = a statement whose text does not match the file layout or is too broad.")
    bad = [c for c in CHECKS if c["status"] != "OK"]
    rows = [[c["id"], c["doc"], c["claim"], c["doc_val"], c["new_val"], c["status"] + (f" ({c['sev']})" if c["sev"] else ""), c["note"]] for c in bad]
    tab(["id", "doc", "claim", "doc value", "recomputed", "status", "note"], rows, ["l", "l", "l", "r", "r", "l", "l"])
    h2("X3. Coverage: raw rows vs data shown")
    rows = []
    for c in COVER:
        status = "complete" if c["shown"] is not None and c["shown"] == c["raw"] else ("summary only" if c["shown"] is None else f"MISMATCH ({c['shown']} shown)")
        rows.append([c["file"], c["raw"], "-" if c["shown"] is None else c["shown"], status, c["where"], c["how"]])
    tab(["file", "raw rows / entries", "shown", "status", "where", "how"], rows, ["l", "r", "r", "l", "l", "l"])
    h2("X4. Source file manifest (files read by this generator)")
    seen = set()
    rows = []
    for m in MANIFEST:
        if m["label"] in seen:
            continue
        seen.add(m["label"])
        rows.append([m["label"], m["rows"] if m["rows"] is not None else "-", m["size"], m["mtime"], m["sha"]])
    tab(["file", "rows", "bytes", "mtime (local)", "sha256[:12]"], rows)
    if UNPARSED:
        h2("X5. Could not parse")
        bullets(UNPARSED)
    else:
        para("All source files parsed without error.")


# ====================================================================================================================
# ==== SECTIONS-END ====
# ====================================================================================================================

SECTIONS = [sec_C1, sec_C2, sec_C3, sec_C4, sec_C5, sec_C6, sec_docs, sec_final]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sec in SECTIONS:
        sec()
    body = list(LINES)
    LINES.clear()
    emit("# Appendix C: LoRA banks and RegMean - tables generated from the raw result files")
    emit()
    para(f"Generated by `make_tables.py` on {time.strftime('%Y-%m-%d %H:%M')} (local time) from the files listed in X4; nothing in the tables is typed by hand except the 'doc value' column of X1. "
         "Source paths are written `master:results/...` (= /data/home/azpitarte/condmerge/results/...) and `aa:results/aa/...` (= /data/home/azpitarte/condmerge_aa/results/aa/..., branch aa/side).")
    h3("General legend")
    bullets([
        "**Experts / splits.** 8 CLIP ViT-B/32 task experts: " + ", ".join(TASKS8) + ". `dev` = 20% of the official test set (model selection), `test` = the other 80% (frozen configs only).",
        "**Information budgets.** A = weights only; B = pooled unlabeled inputs without task IDs; C = task-separated unlabeled inputs (expert i sees task i's unlabeled calibration images); D = labeled dev data (oracle / diagnostic only). "
        "LoRA RegMean names: `C` (= budget C), `Bp` (= budget B, pooled through each expert), `B0` (pooled through the base model; RegMean must equal averaging).",
        "**Metrics.** LoRA files (`aa:`): `avg_acc` = ABSOLUTE accuracy, mean over the 8 tasks of correct/total (not normalized); shown in percent. Master full-FT files (`master:`): `mean_norm` = NORMALIZED accuracy "
        "(per task accuracy / that task's own expert accuracy, averaged), `mean_acc` = absolute accuracy, `min_norm` = worst task's normalized accuracy; shown in percent. Differences are in percentage points (pt). "
        "The two sides are never mixed in one column; C5 is normalized-and-absolute, C1-C4 and C6 are absolute.",
        "**The rule.** lambda_i = softmax_i(-beta * kappa2(W_i)), beta = b / (median over maps of the across-expert std of the score); merge W0 + alpha * sum_i lambda_i (W_i - W0). b = 0 is uniform, b = inf is argmin selection.",
        "**Banks.** `knots_release` (the KnOTS-release rank-16 bank, also called `knots_lora_r16_vitb32`), `lora_c5a` (KnOTS recipe, seed 0), `lora_c5a_seed1` (same recipe, seed 1), `lora` (= vit20, 2000 steps). "
        "Only the KnOTS release bank has spectra (C2) and the rule sweep (C3).",
        "**RegMean.** D (G + rho I) = sum_i D_i G~_i + rho R, G~ = a G + (1 - a) diag G; 'kappa cap' = smallest rho with kappa2(G + rho I) <= kappa_t; `rel_ridge` = fixed rho = c tr(G)/d.",
        "**Number formats.** Percent with 2 decimals unless stated; kappa and tiny quantities in scientific notation; '-' = not applicable / not recorded; 'inf' = numerically singular.",
    ])
    h3("Index of tables (rows per table)")
    by_sec = defaultdict(list)
    for sec_, tid, nrows, ncols in TABLE_LOG:
        by_sec[sec_].append((tid, nrows, ncols))
    rows = []
    for sec_ in [k for k in by_sec]:
        rows.append([sec_, "; ".join(f"{tid} ({nr} rows)" for tid, nr, nc in by_sec[sec_])])
    tab(["section", "tables (id, data rows)"], rows, ["l", "l"])
    LINES.extend(body)
    (OUT_DIR / "tables.md").write_text("\n".join(LINES) + "\n")
    (OUT_DIR / "crosschecks.json").write_text(json.dumps({"checks": CHECKS, "tables": [{"section": a, "id": b, "rows": c, "cols": d} for a, b, c, d in TABLE_LOG],
                                                        "unparsed": UNPARSED, "manifest": MANIFEST}, indent=1, default=str))
    print("wrote", OUT_DIR / "tables.md", len("\n".join(LINES).splitlines()), "lines;", len(CHECKS), "check rows;", sum(1 for c in CHECKS if c["status"] == "DISCREPANCY"), "discrepancies;", len(TABLE_LOG), "tables")


if __name__ == "__main__":
    main()
