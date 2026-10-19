#!/usr/bin/env python
"""Appendix A: weight-conditioning experiments on full fine-tunes (FusionBench CLIP ViT-B/32, B/16).

Builds tables A1..A7 (+ legends, integrity checks and a cross-check against docs/REPORT_01 and
docs/FULL_REPORT.md) for the research report.  EVERY number in tables.md is computed here from the
raw result files under /data/home/azpitarte/condmerge (read-only); nothing is typed in by hand
except (i) the *claims quoted from the docs* in the cross-check table X1 (they are the things being
checked) and (ii) free-text captions.  The MD5 of every input file actually read is listed in F1.

Usage:  OMP_NUM_THREADS=4 /data/home/azpitarte/lora_merging_ica/.venv/bin/python make_tables.py
Output: tables.md next to this script.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import io
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

REPO = Path("/data/home/azpitarte/condmerge")
RES = REPO / "results"
CFG = REPO / "configs"
DOCS = REPO / "docs"
ARTIFACTS = REPO / "artifacts"  # symlink to /data/home/azpitarte/condmerge_data (per-example correctness arrays)
OUT_DIR = Path(__file__).resolve().parent
OUT_MD = OUT_DIR / "tables.md"
TASKS8 = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
INF = math.inf

# --------------------------------------------------------------------------------------------
# I/O with provenance (md5 of the exact bytes parsed)
# --------------------------------------------------------------------------------------------
USED: dict[str, dict] = {}


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def read_bytes(p) -> bytes:
    p = Path(p)
    b = p.read_bytes()
    key = rel(p)
    if key not in USED:
        USED[key] = {"bytes": len(b), "md5": hashlib.md5(b).hexdigest(), "lines": b.count(b"\n")}
    return b


def read_json(p):
    return json.loads(read_bytes(p))


def read_jsonl(p) -> list[dict]:
    return [json.loads(x) for x in read_bytes(p).decode().splitlines() if x.strip()]


def read_csv(p) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_bytes(p)))


def read_text(p) -> str:
    return read_bytes(p).decode()


# --------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------
def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d: int = 2) -> str:
    """fraction -> percent string with d decimals"""
    return "n/a" if isnan(x) else f"{100 * x:.{d}f}"


def pp(x, d: int = 2) -> str:
    """fraction difference -> signed percentage points"""
    return "n/a" if isnan(x) else f"{100 * x:+.{d}f}"


def g(x, sig: int = 4) -> str:
    if isnan(x):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{sig}g}"


def kfmt(x) -> str:
    """condition numbers: 4 significant digits, exponent form from 1e4 up"""
    return g(x, 4)


def fnum(x, d: int = 3) -> str:
    return "n/a" if isnan(x) else f"{x:.{d}f}"


def sfnum(x, d: int = 3) -> str:
    return "n/a" if isnan(x) else f"{x:+.{d}f}"


def bfmt(b) -> str:
    return "inf" if (isinstance(b, float) and math.isinf(b)) else f"{b:g}"


def afmt(a) -> str:
    return f"{a:g}"


def esc(s) -> str:
    """Escape for a Markdown table cell: literal pipes, and '<' that could start an HTML tag (e.g. 'shuf<seed>')."""
    return re.sub(r"<(?=[A-Za-z/!?])", r"\\<", str(s).replace("|", "\\|"))


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    n = len(headers)
    align = align or ("l" + "r" * (n - 1))
    assert len(align) == n, (align, headers)
    sep = ["---:" if a == "r" else (":---:" if a == "c" else ":---") for a in align]
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        assert len(r) == n, (len(r), n, r)
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


TABLES: list[dict] = []


def add_table(tid: str, title: str, what: str, sources: list, headers: list[str], rows: list[list],
              align: str | None = None, notes: list[str] | None = None) -> None:
    assert all(t["tid"] != tid for t in TABLES), f"duplicate table id {tid}"
    TABLES.append({"tid": tid, "title": title, "what": what, "sources": [rel(s) if not isinstance(s, str) else s for s in sources],
                   "headers": headers, "rows": rows, "align": align, "notes": notes or []})


CHECKS: list[dict] = []
_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def compare_str(claimed: str, computed: str) -> str | None:
    """'ok' if identical, 'rounding' if every number differs by at most one unit of the claim's last digit, 'DISCREPANCY' if larger,
    None if the two strings have different structure (then the caller must decide)."""
    if claimed == computed:
        return "ok"
    nc, nx = _NUM.findall(claimed), _NUM.findall(computed)
    if not nc or len(nc) != len(nx) or _NUM.sub("#", claimed) != _NUM.sub("#", computed):
        return None
    worst = 0.0
    for a, b in zip(nc, nx):
        dec = len(a.split(".")[1]) if ("." in a and "e" not in a.lower()) else 0
        ulp = 10.0 ** (-dec) if "e" not in a.lower() else max(abs(float(a)) * 0.05, 1e-300)
        worst = max(worst, abs(float(a) - float(b)) / ulp)
    return "ok" if worst < 1e-9 else ("rounding" if worst <= 1.0001 else "DISCREPANCY")


def check(doc: str, claim: str, claimed: str, computed: str, ok: bool | None = None, note: str = "", approx: bool = False) -> str:
    """Record a doc-vs-data comparison.  status: ok | rounding (<= 1 unit in the last quoted digit) | approx (claim is approximate and holds) | DISCREPANCY.
    ok=None -> automatic string/number comparison; ok=True/False -> the caller's judgement (approx=True marks approximate claims that hold)."""
    if ok is None:
        st = compare_str(claimed, computed) or "DISCREPANCY"
    else:
        st = ("approx" if (approx and claimed != computed) else "ok") if ok else "DISCREPANCY"
    CHECKS.append({"doc": doc, "claim": claim, "claimed": claimed, "computed": computed, "status": st, "ok": st in ("ok",), "note": note})
    return st


WARN: list[str] = []


def warn(msg: str) -> None:
    WARN.append(msg)
    print("WARNING:", msg, file=sys.stderr)


# --------------------------------------------------------------------------------------------
# Rule-name legend, parsed from src/condmerge/scores.py (AST) and docs/PROTOCOL.md
# --------------------------------------------------------------------------------------------
def parse_scores_py() -> dict:
    src_path = REPO / "src" / "condmerge" / "scores.py"
    src = read_text(src_path)
    tree = ast.parse(src)
    lines = src.splitlines()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_rule")
    dct = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "table"
                and isinstance(node.value, ast.Dict)):
            dct = node.value
    assert dct is not None
    entries = {}
    for k, v in zip(dct.keys, dct.values):
        name = k.value
        expr = ast.get_source_segment(src, v.body)
        grp = ""
        for ln in range(k.lineno - 2, -1, -1):
            m = re.match(r"^\s*#\s*---\s*(.*?)\s*(---)?\s*$", lines[ln])
            if m:
                grp = m.group(1).strip().rstrip("-").strip()
                break
        entries[name] = {"expr": " ".join(expr.split()), "group": grp}
    prefixes = {}
    for ln in lines:
        m = re.search(r'if name\.startswith\("([^"]+)"\):\s*#\s*(.*)$', ln)
        if m:
            prefixes[m.group(1)] = m.group(2).strip()
    gt = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "global_table")
    prefixes.setdefault("global:", ast.get_docstring(gt))
    mod_doc = ast.get_docstring(tree)
    return {"entries": entries, "prefixes": prefixes, "module_doc": mod_doc}


def parse_protocol_variants() -> dict:
    txt = read_text(DOCS / "PROTOCOL.md")
    out = {}
    for ln in txt.splitlines():
        if ln.startswith("| `"):
            # split on unescaped pipes only ('\|' is a literal pipe inside a cell)
            cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
            if len(cells) >= 2:
                for nm in re.findall(r"`([^`]+)`", cells[0]):
                    out[nm] = cells[1]
    return out


SCORES = parse_scores_py()
PROTO = parse_protocol_variants()

FAMILY_ORDER = [
    "0 baseline (alpha only)",
    "1 literal rule and direct variants",
    "2 robust / base-relative / sigma decomposition",
    "3 shuffled-score null (layerwise)",
    "4 global (one lambda per expert)",
    "5 shuffled-score null (global)",
    "6 magnitude controls",
    "7 data-supported scores (budget B)",
    "8 task-update spectra",
    "9 interaction-aware",
    "10 random-matrix inverse variance (absolute beta)",
]
_FAM_EXACT = {
    "uniform": 0, "metagpt": 0,
    "kappa": 1, "kappa_rev": 1, "kappa_strict": 1, "logkappa": 1, "logkappa_rev": 1, "kappa_z": 1, "kappa_z_rev": 1,
    "logkappa_q90": 2, "logkappa_q90_rev": 2, "kappa_q90": 2, "kappa_rel": 2, "logsmax": 2, "neglogsmin": 2,
    "dnorm": 6, "dnorm_rev": 6, "dnorm_z": 6, "dnorm_z_rev": 6,
    "slogkappa": 7, "slogkappa_rev": 7, "slogkappa_plus_1e-3": 7, "slogkappa_plus_1e-3_rev": 7,
    "sdnorm": 7, "sdnorm_rev": 7, "sdlogkappa_q90": 7, "sdlogkappa_q90_rev": 7,
    "dlogkappa": 8, "dlogkappa_plus_1e-3": 8, "dlogkappa_q90": 8, "dlogkappa_q90_rev": 8, "dstable_rank": 8,
    "dstable_rank_rev": 8, "dentropy_rank": 8, "dentropy_rank_rev": 8,
    "mkgrad": 9, "mkgrad_rev": 9,
    "gd_invvar": 10, "gd_invvar_rev": 10, "gd_noiseshare": 10,
}


def family(rule: str) -> str:
    if rule in _FAM_EXACT:
        return FAMILY_ORDER[_FAM_EXACT[rule]]
    if rule.startswith("shuf"):
        return FAMILY_ORDER[3]
    if rule.startswith("gshuf"):
        return FAMILY_ORDER[5]
    if rule.startswith("global:gd_"):
        return FAMILY_ORDER[10]
    if rule.startswith("global:"):
        return FAMILY_ORDER[4]
    if rule.startswith("wc:"):
        return FAMILY_ORDER[9]
    if rule.startswith("single:"):
        return "single expert"
    raise KeyError(f"no family for rule {rule!r}")


def split_prefix(rule: str):
    """('shuf3', 'kappa'), ('gshuf1', 'global:kappa_z'), ('global', 'logkappa'), ('wc', 'kappa') or (None, rule)."""
    m = re.match(r"^(shuf\d+|gshuf\d+):(.*)$", rule)
    if m:
        return m.group(1), m.group(2)
    if rule.startswith("global:"):
        return "global", rule[len("global:"):]
    if rule.startswith("wc:"):
        return "wc", rule[3:]
    return None, rule


def rule_legend_row(rule: str) -> list:
    pre, base = split_prefix(rule)
    if rule in ("uniform", "metagpt"):
        sc = {"uniform": "(no score; beta = 0, lambda_i = 1/K)", "metagpt": "(no score; lambda_i proportional to ||D_i||^2 over all parameters, experiments/rule_sweep.py:global_norm_coeffs)"}[rule]
        gl = PROTO.get(rule, "Task Arithmetic with alpha tuned" if rule == "uniform" else "")
        return [rule, sc, gl or "baseline"]
    if pre is None:
        e = SCORES["entries"].get(rule)
        if e is None:
            raise KeyError(f"rule {rule!r} not in scores.py table")
        desc = PROTO.get(rule, "")
        return [rule, e["expr"], (desc + " " if desc else "") + f"[scores.py group: {e['group']}]"]
    if pre.startswith("shuf"):
        be = SCORES["entries"][base]
        return [rule, f"scores of `{base}` ({be['expr']}) permuted across experts independently within each map, seed {pre[4:]}",
                SCORES["prefixes"]["shuf"]]
    if pre.startswith("gshuf"):
        sub = rule_legend_row(base)
        return [rule, f"scores of `{base}` with ONE expert permutation (seed {pre[5:]}) applied to all maps; {sub[1]}",
                SCORES["prefixes"]["gshuf"]]
    if pre == "global":
        sub = rule_legend_row(base)
        return [rule, f"one score per expert = parameter-count-weighted mean over maps of `{base}` ({sub[1]})", SCORES["prefixes"]["global:"]]
    if pre == "wc":
        sub = rule_legend_row(base)
        return [rule, f"within-map z-score of `{base}` minus each expert's mean z over maps ({sub[1]})", SCORES["prefixes"]["wc:"]]
    raise KeyError(rule)


def parse_variant_tag(tag: str):
    """'kappa|tuned' -> ('kappa', 'tuned'); 'best_single' -> ('single', 'best_single')."""
    if "|" in tag:
        rule, var = tag.rsplit("|", 1)
        return rule, var
    return tag, ""


# --------------------------------------------------------------------------------------------
# Shared reference data
# --------------------------------------------------------------------------------------------
NPZ_USED: dict[str, str] = {}


def load_correct(key: str) -> dict[str, np.ndarray] | None:
    p = ARTIFACTS / "runs" / "freeze_correct" / f"{key}.npz"
    if not p.exists():
        return None
    b = p.read_bytes()
    NPZ_USED[key] = hashlib.md5(b).hexdigest()
    z = np.load(io.BytesIO(b))
    return {k: z[k] for k in z.files}


_CACHE: dict = {}


def cached(name, fn):
    if name not in _CACHE:
        _CACHE[name] = fn()
    return _CACHE[name]


def expert_acc(arch: str) -> dict:
    p = RES / "experts" / "expert_accuracy.json" if arch == "b32" else RES / "b16" / "experts" / "expert_accuracy.json"
    return cached(("ea", arch), lambda: read_json(p))


def expert_acc_path(arch: str) -> Path:
    return RES / "experts" / "expert_accuracy.json" if arch == "b32" else RES / "b16" / "experts" / "expert_accuracy.json"


def sanity() -> dict:
    return cached("sanity", lambda: read_json(RES / "sanity" / "expert_matrix_dev.json"))


def freeze_b32_records() -> list[dict]:
    return cached("freeze_test", lambda: read_jsonl(RES / "freeze" / "test.jsonl"))


def freeze_b16_records() -> list[dict]:
    return cached("b16_test", lambda: read_jsonl(RES / "b16" / "freeze" / "transfer_test.jsonl"))


def test_sizes(arch: str) -> dict[str, int] | None:
    def go():
        recs = freeze_b32_records() if arch == "b32" else freeze_b16_records()
        tag = "uniform|tuned" if arch == "b32" else "b16xfer|uniform|tuned"
        r = next(x for x in recs if x["cfg"]["tag"] == tag)
        c = load_correct(r["key"])
        return None if c is None else {t: int(len(c[t])) for t in TASKS8}
    return cached(("ntest", arch), go)


def dev_sizes() -> dict[str, int]:
    return sanity()["n"]


def mean_over(d: dict, tasks=TASKS8) -> float:
    return float(np.mean([d[t] for t in tasks]))


# --------------------------------------------------------------------------------------------
# A1: expert / zero-shot accuracy
# --------------------------------------------------------------------------------------------
def build_A1() -> None:
    nd, nt = dev_sizes(), test_sizes("b32")
    pub_e, pub_z = sanity()["published_individual"], sanity()["published_pretrained"]
    ARCH_NAME = {"b32": "ViT-B/32", "b16": "ViT-B/16"}
    for arch in ("b32", "b16"):
        ea = expert_acc(arch)
        zd, zt = ea["zeroshot"]["dev"], ea["zeroshot"]["test"]
        ed, et = ea["expert"]["dev"], ea["expert"]["test"]
        rows = []
        for t in TASKS8:
            rows.append([t, str(nd[t]), str(nt[t]) if nt else "n/a", pct(zd[t]), pct(ed[t]), pct(zt[t]), pct(et[t]),
                         pct(zd[t] / ed[t]), pct(zt[t] / et[t])])
        rows.append(["mean over 8 tasks (n: sum)", str(sum(nd.values())), str(sum(nt.values())) if nt else "n/a",
                     pct(mean_over(zd)), pct(mean_over(ed)), pct(mean_over(zt)), pct(mean_over(et)),
                     pct(np.mean([zd[t] / ed[t] for t in TASKS8])), pct(np.mean([zt[t] / et[t] for t in TASKS8]))])
        add_table(
            "A1a" if arch == "b32" else "A1c",
            f"Zero-shot base and full-fine-tuned expert accuracy, CLIP {ARCH_NAME[arch]}",
            f"Per task, the zero-shot base model and the task's own full-fine-tuned FusionBench expert, on the dev split (20 % of the official test set) and on the "
            f"test split (the other 80 %); metric = absolute top-1 accuracy (%), plus zero-shot accuracy divided by the own-expert accuracy (the 'mean_norm' of the do-nothing model), all in %, 2 decimals.",
            [expert_acc_path(arch), RES / "sanity" / "expert_matrix_dev.json", "artifacts/runs/freeze_correct/<key>.npz (array lengths = n_test)"],
            ["task", "n dev", "n test", "zero-shot dev acc", "expert dev acc", "zero-shot test acc", "expert test acc",
             "zero-shot / expert, dev (norm)", "zero-shot / expert, test (norm)"], rows,
            notes=["Accuracy columns are percent; the last row's 'norm' columns are mean_norm (mean over tasks of acc / own-expert acc) of the zero-shot model. "
                   "Expert accuracy on its own task is the normaliser of every mean_norm in this appendix (same split).",
                   ("n dev from results/sanity/expert_matrix_dev.json; n test = length of the per-example correctness vector saved by experiments/freeze_eval.py. "
                    + ("" if arch == "b32" else "B/16 uses the same images and splits as B/32 (pixel caches are shared); the n values shown are the B/32 ones and the B/16 test length is cross-checked below."))])
        if arch == "b16":
            ntb = test_sizes("b16")
            ok = (ntb == nt) if ntb else None
            check("derived", "B/16 test split sizes equal B/32 test split sizes", "equal", "equal" if ok else f"differ: {ntb}", ok=bool(ok) if ok is not None else True)

    fr = {t: nd[t] / (nd[t] + nt[t]) for t in TASKS8}
    check("PROTOCOL / REPORT_01 §1", "dev = 20 % of each official test split (n_dev / (n_dev + n_test), all 8 tasks)", "20 %", f"{100 * min(fr.values()):.2f}-{100 * max(fr.values()):.2f} %",
          ok=all(abs(v - 0.2) < 0.0005 for v in fr.values()))
    # published comparison (B/32 only): n-weighted dev+test = the full official test set
    ea = expert_acc("b32")
    rows = []
    comb = {}
    for t in TASKS8:
        w_d, w_t = nd[t], nt[t]
        cz = (w_d * ea["zeroshot"]["dev"][t] + w_t * ea["zeroshot"]["test"][t]) / (w_d + w_t)
        ce = (w_d * ea["expert"]["dev"][t] + w_t * ea["expert"]["test"][t]) / (w_d + w_t)
        comb[t] = (cz, ce)
        rows.append([t, pct(cz), f"{pub_z[t]:.1f}", f"{100 * cz - pub_z[t]:+.2f}", pct(ce), f"{pub_e[t]:.1f}", f"{100 * ce - pub_e[t]:+.2f}"])
    mz, me = np.mean([comb[t][0] for t in TASKS8]), np.mean([comb[t][1] for t in TASKS8])
    pz, pe = np.mean([pub_z[t] for t in TASKS8]), np.mean([pub_e[t] for t in TASKS8])
    rows.append(["mean over 8 tasks", pct(mz), f"{pz:.2f}", f"{100 * mz - pz:+.2f}", pct(me), f"{pe:.2f}", f"{100 * me - pe:+.2f}"])
    add_table("A1b", "Reproduction of the published FusionBench ViT-B/32 numbers",
              "Per task, our zero-shot and expert accuracy on the full official test set (dev and test pooled, weighted by n) next to the published values recorded in results/sanity/expert_matrix_dev.json; metric = absolute top-1 accuracy (%).",
              [expert_acc_path("b32"), RES / "sanity" / "expert_matrix_dev.json", "artifacts/runs/freeze_correct/<key>.npz (n test)"],
              ["task", "ours zero-shot (dev+test)", "published zero-shot", "diff (pt)", "ours expert (dev+test)", "published expert", "diff (pt)"], rows,
              notes=["Published values are given with one decimal in the sanity file; the mean row's published entries are means of those eight rounded numbers."])
    check("REPORT_01 §1 / FULL §2", "zero-shot base, mean test accuracy 'matching the published 48.0'", "48.0", f"{100 * mean_over(ea['zeroshot']['test']):.1f}")
    check("REPORT_01 §1 / FULL §2", "FusionBench experts, mean test accuracy", "90.3", f"{100 * mean_over(ea['expert']['test']):.1f}")
    check("REPORT_01 §1 / FULL §2", "FusionBench experts, mean dev accuracy", "90.5", f"{100 * mean_over(ea['expert']['dev']):.1f}")
    check("FULL §2", "zero-shot dev mean accuracy", "48.36", f"{100 * mean_over(ea['zeroshot']['dev']):.2f}")
    check("RESEARCH_LOG 10-18 setup", "published zero-shot mean 48.0 vs mean of the eight published per-task values (1-decimal rounding)", "48.0", f"{pz:.2f}",
          ok=abs(pz - 48.0) <= 0.1, approx=True, note="consistent within the rounding of the per-task published values")

    # cross-task matrices
    for arch, tag in (("b32", "ViT-B/32"), ("b16", "ViT-B/16")):
        ea = expert_acc(arch)
        for sp, tid in (("dev", "d"), ("test", "e")):
            cross = ea["cross"][sp]
            exp = ea["expert"][sp]
            zs = ea["zeroshot"][sp]
            rows = []
            for e in TASKS8:
                rows.append([f"expert {e}"] + [pct(cross[e][t]) for t in TASKS8]
                            + [pct(mean_over(cross[e])), pct(np.mean([cross[e][t] / exp[t] for t in TASKS8]))])
            rows.append(["zero-shot base"] + [pct(zs[t]) for t in TASKS8] + [pct(mean_over(zs)), pct(np.mean([zs[t] / exp[t] for t in TASKS8]))])
            letter = {("b32", "dev"): "d", ("b32", "test"): "e", ("b16", "dev"): "f", ("b16", "test"): "g"}[(arch, sp)]
            add_table(f"A1{letter}", f"Cross-task accuracy of every expert, CLIP {tag}, {sp} split",
                      f"Row = which fine-tuned expert is loaded, column = which task it is evaluated on, on the {sp} split of CLIP {tag}; metric = absolute top-1 accuracy (%), "
                      f"with each row's mean_acc and mean_norm (acc / own-expert acc, averaged over the 8 tasks) as the single-expert-for-everything baseline.",
                      [expert_acc_path(arch)], ["loaded model \\ evaluated on"] + TASKS8 + ["mean_acc", "mean_norm"], rows,
                      notes=["The diagonal is the own-task expert accuracy (the normaliser). The best row by mean_norm is the 'best single expert' baseline; "
                             "the 'best_single' frozen test config in A4 re-evaluates that expert on the test split."])

    # precision sanity
    san = sanity()
    ea = expert_acc("b32")
    rows = []
    for t in TASKS8:
        pr, pr32 = san["pretrained"][t], san["pretrained_fp32"][t]
        de16, de32 = san["expert_matrix"]["dtd"][t], san["dtd_expert_fp32"][t]
        own16, own32 = ea["expert"]["dev"][t], ea["fp32_check"][t]
        rows.append([t, pct(pr), pct(pr32), f"{100 * (pr32 - pr):+.3f}", pct(de16), pct(de32), f"{100 * (de32 - de16):+.3f}",
                     pct(own16), pct(own32), f"{100 * (own32 - own16):+.3f}"])
    mx = max(abs(float(r[3])) for r in rows), max(abs(float(r[6])) for r in rows), max(abs(float(r[9])) for r in rows)
    rows.append(["max |diff| (pt)", "", "", f"{mx[0]:.3f}", "", "", f"{mx[1]:.3f}", "", "", f"{mx[2]:.3f}"])
    add_table("A1h", "Precision check: fp16 autocast vs fp32 evaluation, ViT-B/32, dev split",
              "Per task, accuracy under the fp16-autocast protocol used everywhere and under fp32, for the zero-shot base (all 8 tasks), for the DTD expert (all 8 tasks) and for each expert on its own task (results/experts fp32_check), on the dev split; metric = absolute top-1 accuracy (%), differences in percentage points.",
              [RES / "sanity" / "expert_matrix_dev.json", expert_acc_path("b32")],
              ["task", "base fp16", "base fp32", "diff (pt)", "DTD expert fp16", "DTD expert fp32", "diff (pt)", "own-task expert fp16", "own-task expert fp32", "diff (pt)"], rows,
              notes=["The sanity file's 'expert_matrix' and the cross matrix of results/experts (dev) come from two separate evaluation runs and differ by at most one image per cell (see X1)."])
    check("PROTOCOL / FULL §2", "max |fp16 - fp32| over tasks (pt)", "0.19", f"{max(mx):.2f}", ok=max(mx) <= 0.195, note="claim is an upper bound (<= 0.19)")
    # consistency between the sanity expert matrix and the cross matrix (dev)
    nd_ = san["n"]
    dimg = [abs(san["expert_matrix"][e][t] - ea["cross"]["dev"][e][t]) * nd_[t] for e in TASKS8 for t in TASKS8]
    n_diff = sum(1 for x in dimg if x > 0.5)
    check("derived", "two independent dev evaluations of the 8x8 expert matrix (results/sanity vs results/experts): max difference in images", "<= 1 image", f"{max(dimg):.2f} images ({n_diff} of 64 cells differ)",
          ok=max(dimg) <= 1.01, note="fp16 run-to-run nondeterminism: at most one flipped prediction per cell")


# --------------------------------------------------------------------------------------------
# A2: spectra
# --------------------------------------------------------------------------------------------
SPEC_DIR = {"b32": RES / "spectra", "b16": RES / "b16" / "spectra"}
ARCH_NAME = {"b32": "ViT-B/32", "b16": "ViT-B/16"}
MLP_MODS, ATT_MODS = ("fc1", "fc2"), ("q_proj", "k_proj", "v_proj", "out_proj")
SPEC: dict = {}


def spectra(arch: str) -> dict:
    if arch in SPEC:
        return SPEC[arch]
    d = SPEC_DIR[arch]
    st = read_csv(d / "stats.csv")
    pm = read_csv(d / "per_map.csv")
    summ = read_json(d / "summary.json")
    st["q"] = st[["m", "n"]].min(axis=1)
    base = st[st.kind == "base"].set_index("param")
    full = st[st.kind == "full"]
    upd = st[st.kind == "update"]
    gp = lambda df, col, fn: getattr(df.groupby("param", sort=False)[col], fn)()  # noqa: E731
    M = pm.set_index("param").copy()
    M["m"], M["n"], M["q"] = base.loc[M.index, "m"], base.loc[M.index, "n"], base.loc[M.index, "q"]
    M["base_rank"] = base.loc[M.index, "rank_default"]
    M["base_sing"] = M["base_rank"] < M["q"]
    M["base_fro"] = base.loc[M.index, "fro"]
    M["kmed"] = gp(full, "kappa2_raw", "median")
    M["kmin_chk"], M["kmax_chk"] = gp(full, "kappa2_raw", "min"), gp(full, "kappa2_raw", "max")
    sing = (full.rank_default < full.q).groupby(full.param, sort=False).sum()
    M["n_sing_exp"] = sing
    M["dfro_med"], M["dfro_min"], M["dfro_max"] = gp(upd, "fro", "median"), gp(upd, "fro", "min"), gp(upd, "fro", "max")
    M["drel_med"] = gp(upd, "rel_update_fro", "median")
    M["dsr_med"], M["dsr_min"], M["dsr_max"] = gp(upd, "stable_rank", "median"), gp(upd, "stable_rank", "min"), gp(upd, "stable_rank", "max")
    # consistency of per_map min/max with stats.csv
    assert np.allclose(M["kappa_min"], M["kmin_chk"], rtol=1e-12) and np.allclose(M["kappa_max"], M["kmax_chk"], rtol=1e-12)
    SPEC[arch] = {"st": st, "pm": pm, "summ": summ, "M": M, "base": base, "full": full, "upd": upd, "dir": d}
    return SPEC[arch]


def mod_group(m: str) -> str:
    return "MLP (fc1, fc2)" if m in MLP_MODS else ("attention (q, k, v, out)" if m in ATT_MODS else "patch_embedding")


def build_A2_permap() -> None:
    for arch in ("b32", "b16"):
        S = spectra(arch)
        M, d = S["M"], S["dir"]
        rows_a, rows_b = [], []
        for p, r in M.iterrows():
            rows_a.append([p, str(int(r.layer)), r.module, f"{int(r.m)}x{int(r.n)}", kfmt(r.kappa0), f"{int(r.base_rank)}/{int(r.q)}",
                           "yes" if r.base_sing else "no", kfmt(r.kappa_min), kfmt(r.kmed), kfmt(r.kappa_max), g(r.rel_spread, 3),
                           g(r.max_rel_shift, 3), str(int(r.n_sing_exp)), fnum(r.fp32_noise_sd_over_spread, 4), r.argmin_expert])
            rows_b.append([p, g(r.base_fro, 4), f"{g(r.dfro_med, 3)} [{g(r.dfro_min, 3)}, {g(r.dfro_max, 3)}]", g(r.drel_med, 3),
                           f"{g(r.dsr_med, 3)} [{g(r.dsr_min, 3)}, {g(r.dsr_max, 3)}]", str(int(r.update_rank_default_min)),
                           fnum(r.first_order_r, 3), g(r.first_order_ratio_med, 3), fnum(r.spearman_kappa_vs_dnorm, 3),
                           fnum(r.spearman_smin_vs_dnorm, 3), fnum(r.spearman_smax_vs_dnorm, 3), g(r.base_smin_over_dnorm_med, 3)])
        suf = "" if arch == "b32" else "'"
        add_table(
            f"A2a" if arch == "b32" else "A2c", f"Per linear map: condition numbers, rank flags and fp32 rounding share, CLIP {ARCH_NAME[arch]} full fine-tunes",
            f"One row per linear map (the {len(M)} matrices the rule acts on): the base model's kappa2 (kappa0), the min/median/max of kappa2(W_i) over the 8 experts, their relative spread, numerical-rank flags and the fp32-rounding share, "
            f"from the SVDs of the stored weights (no accuracy involved; condition numbers are dimensionless).",
            [d / "per_map.csv", d / "stats.csv"],
            ["param", "layer", "module", "shape m x n", "kappa0 (base kappa2_raw)", "base numerical rank / q", "base fp32-singular", "kappa min over 8 experts",
             "kappa median", "kappa max", "rel_spread = sd_i(kappa_i)/kappa0", "max_rel_shift = max_i|kappa_i-kappa0|/kappa0",
             "experts rank-deficient (of 8)", "fp32 noise sd / across-expert sd", "argmin expert"], rows_a, align="lrllrrlrrrrrrrl",
            notes=["kappa = kappa2_raw = s_1/s_q from a float64 SVD of the stored float32 matrix, q = min(m, n); layer -1 = patch embedding. 'rank' is rank_default = #{s_i > max(m,n) * eps32 * s_1}; "
                   "'fp32-singular' means rank_default < q, which is exactly where kappa2 (kappa_strict) is +inf. rel_spread and max_rel_shift are defined in experiments/analyze_spectra.py (population s.d., ddof = 0). "
                   "'fp32 noise sd / across-expert sd' = s.d. over experts of (kappa from an fp32 SVD minus kappa from an fp64 SVD) divided by the s.d. over experts of kappa (the 'fp32 rounding share'; a value near or above 1 means the rounding error of kappa is as large as its spread across experts)."])
        add_table(
            f"A2b" if arch == "b32" else "A2d", f"Per linear map: task-update norms, stable rank and first-order probe, CLIP {ARCH_NAME[arch]} full fine-tunes",
            f"One row per linear map: the base Frobenius norm, the Frobenius norm of the task updates D_i = W_i - W0 (median over the 8 experts, [min, max]), the update stable rank, and how well the first-order rank-2 probe and the update norm relate to kappa across experts "
            f"(weights only, no accuracy; norms are in weight units, ranks and correlations are dimensionless).",
            [d / "per_map.csv", d / "stats.csv"],
            ["param", "||W0||_F", "||D_i||_F median [min, max]", "||D_i||_F / ||W0||_F, median", "update stable rank median [min, max]",
             "min update numerical rank", "first-order Pearson r (pred vs actual kappa shift)", "median pred/actual", "Spearman(kappa_i, ||D_i||)",
             "Spearman(smin_i, ||D_i||)", "Spearman(smax_i, ||D_i||)", "base smin / median ||D_i||"], rows_b, align="lrrrrrrrrrrr",
            notes=["Stable rank = ||D||_F^2 / ||D||_2^2. Correlations are Spearman across the 8 experts within the map (experiments/analyze_spectra.py); 'first-order' = kappa0 (u1' D v1/s1 - uq' D vq/sq), "
                   "the linear rank-2 probe of MATH_NOTES N2, compared with the actual kappa shift across the 8 experts (Pearson r)."])


def recompute_summary(M: pd.DataFrame) -> dict:
    out = {
        "n_maps": len(M),
        "kappa0_range": [float(M.kappa0.min()), float(M.kappa0.max())],
        "median_rel_spread": float(np.median(M.rel_spread)),
        "median_max_rel_shift": float(np.median(M.max_rel_shift)),
        "median_fp32_noise_over_spread": float(np.nanmedian(M.fp32_noise_sd_over_spread)),
        "first_order_r_median": float(np.nanmedian(M.first_order_r)),
        "spearman_kappa_dnorm_median": float(np.nanmedian(M.spearman_kappa_vs_dnorm)),
        "argmin_selection_counts": dict(collections.Counter(M.argmin_expert)),
    }
    by = {}
    for mod, gdf in M.groupby("module", sort=False):
        by[mod] = {"n": len(gdf), "kappa0_median": float(np.median(gdf.kappa0)), "rel_spread_median": float(np.median(gdf.rel_spread)),
                   "first_order_r_median": float(np.nanmedian(gdf.first_order_r)),
                   "spearman_kappa_dnorm_median": float(np.nanmedian(gdf.spearman_kappa_vs_dnorm)),
                   "spearman_smin_dnorm_median": float(np.nanmedian(gdf.spearman_smin_vs_dnorm)),
                   "argmin_selection": dict(collections.Counter(gdf.argmin_expert))}
    out["by_module"] = by
    return out


def _close(a, b, tol=1e-9) -> bool:
    if isinstance(a, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return bool(np.isclose(a, b, rtol=tol, atol=tol))


def build_A2_summary() -> None:
    rows, rows_mod = [], []
    S = {a: spectra(a) for a in ("b32", "b16")}
    for a in S:
        rc = recompute_summary(S[a]["M"])
        ok = _close(rc, S[a]["summ"])
        check("derived", f"{rel(S[a]['dir'] / 'summary.json')} equals the values recomputed from per_map.csv", "match", "match" if ok else "MISMATCH", ok=ok)
    labels = [("n_maps", "number of linear maps"), ("kappa0_min", "kappa0 range: min"), ("kappa0_max", "kappa0 range: max"),
              ("median_rel_spread", "median over maps of rel_spread (sd_i kappa_i / kappa0)"),
              ("median_max_rel_shift", "median over maps of max_i |kappa_i - kappa0| / kappa0"),
              ("median_fp32_noise_over_spread", "median over maps of fp32-noise s.d. / across-expert s.d."),
              ("first_order_r_median", "median over maps of first-order Pearson r"),
              ("spearman_kappa_dnorm_median", "median over maps of Spearman(kappa_i, ||D_i||_F)")]
    for key, lab in labels:
        vals = []
        for a in ("b32", "b16"):
            sm = S[a]["summ"]
            if key == "n_maps":
                vals.append(str(sm["n_maps"]))
            elif key == "kappa0_min":
                vals.append(kfmt(sm["kappa0_range"][0]))
            elif key == "kappa0_max":
                vals.append(kfmt(sm["kappa0_range"][1]))
            else:
                vals.append(g(sm[key], 4))
        rows.append([lab] + vals)
    for e in TASKS8:
        rows.append([f"argmin-kappa winner count: {e}"] + [str(S[a]["summ"]["argmin_selection_counts"].get(e, 0)) for a in ("b32", "b16")])
    add_table("A2e", "Headline spectra numbers (summary.json), ViT-B/32 and ViT-B/16",
              "Summary statistics over the 73 linear maps of the condition numbers of the 8 full-fine-tuned experts for CLIP ViT-B/32 and ViT-B/16; weights only, dimensionless (counts, ratios, correlations).",
              [SPEC_DIR["b32"] / "summary.json", SPEC_DIR["b16"] / "summary.json"],
              ["statistic", "ViT-B/32 (results/spectra/summary.json)", "ViT-B/16 (results/b16/spectra/summary.json)"], rows,
              notes=["Both files were also recomputed from per_map.csv (check in X1: all values match). 'argmin-kappa winner' = the expert with the smallest kappa2 in a map (what the b = inf limit of the literal rule selects)."])
    for a in ("b32", "b16"):
        sm = S[a]["summ"]
        for mod, v in sm["by_module"].items():
            sel = ", ".join(f"{e} {v['argmin_selection'][e]}" for e in TASKS8 if e in v["argmin_selection"])
            rows_mod.append([ARCH_NAME[a], mod, str(v["n"]), kfmt(v["kappa0_median"]), g(v["rel_spread_median"], 3), fnum(v["first_order_r_median"], 3),
                             fnum(v["spearman_kappa_dnorm_median"], 3), fnum(v["spearman_smin_dnorm_median"], 3), sel])
    add_table("A2f", "Spectra summary by module type (summary.json by_module)",
              "Per module type of the 73 linear maps, medians over maps of the base kappa0, the relative across-expert spread of kappa, the first-order-probe correlation and the Spearman correlations with the update norm, for ViT-B/32 and ViT-B/16; weights only, dimensionless.",
              [SPEC_DIR["b32"] / "summary.json", SPEC_DIR["b16"] / "summary.json"],
              ["backbone", "module", "n maps", "kappa0 median", "rel_spread median", "first-order r median", "Spearman(kappa, ||D||) median",
               "Spearman(smin, ||D||) median", "argmin-kappa winners (expert count)"], rows_mod, align="llrrrrrrl")


def build_A2_regimes() -> None:
    rows = []
    for a in ("b32", "b16"):
        S = spectra(a)
        M = S["M"].copy()
        M["grp"] = M.module.map(mod_group)
        groups = [("all 73 maps", M)] + [(gname, gdf) for gname, gdf in M.groupby("grp", sort=False)]
        full = S["full"]
        for gname, gdf in groups:
            ps = set(gdf.index)
            fg = full[full.param.isin(ps)]
            n_pairs_sing = int((fg.rank_default < fg.q).sum())
            maps_any = int((gdf.n_sing_exp > 0).sum())
            maps_all = int((gdf.n_sing_exp == 8).sum())
            rows.append([ARCH_NAME[a], gname, str(len(gdf)), kfmt(gdf.kappa0.min()), kfmt(gdf.kappa0.median()), kfmt(gdf.kappa0.max()),
                         str(int(((gdf.kappa0 >= 15) & (gdf.kappa0 <= 60)).sum())), g(gdf.rel_spread.median(), 3), g(gdf.max_rel_shift.median(), 3),
                         fnum(gdf.first_order_r.median(), 3), str(int(gdf.base_sing.sum())), f"{maps_any} (all 8: {maps_all})",
                         f"{n_pairs_sing}/{len(fg)}", fnum(gdf.fp32_noise_sd_over_spread.median(), 4), str(int((gdf.fp32_noise_sd_over_spread > 0.5).sum()))])
    add_table("A2g", "Two regimes: MLP vs attention vs patch-embedding maps (derived from per_map.csv and stats.csv)",
              "Per group of linear maps (all / MLP / attention / patch embedding) and per backbone, the range of the base condition number, the relative spread of kappa across the 8 experts, first-order-probe quality and the counts of fp32-singular matrices; weights only, dimensionless.",
              [SPEC_DIR["b32"] / "per_map.csv", SPEC_DIR["b32"] / "stats.csv", SPEC_DIR["b16"] / "per_map.csv", SPEC_DIR["b16"] / "stats.csv"],
              ["backbone", "group", "n maps", "kappa0 min", "kappa0 median", "kappa0 max", "# maps with kappa0 in [15, 60]", "rel_spread median", "max_rel_shift median",
               "first-order r median", "# base maps fp32-singular", "# maps with >= 1 rank-deficient expert", "# (map, expert) rank-deficient / total",
               "fp32 noise share median", "# maps with fp32 noise share > 0.5"], rows, align="llrrrrrrrrrrrrr",
              notes=["'rank-deficient' = numerical rank at tolerance max(m,n)*eps32*s_1 below q (kappa_strict = +inf). The interval [15, 60] is the 'MLP maps: kappa0 is 15-60' statement of REPORT_01 §3, tested in X1."])


def build_A2_first_order() -> None:
    rows = []
    for a in ("b32", "b16"):
        S = spectra(a)
        d = S["dir"]
        fo = read_csv(d / "first_order.csv")
        pm = S["M"]
        fo["module"] = fo.param.map(pm["module"])
        fo["grp"] = fo.module.map(mod_group)
        fo["relerr"] = (fo.pred_shift - fo.actual_shift).abs() / fo.actual_shift.abs()
        fo["sign_ok"] = np.sign(fo.pred_shift) == np.sign(fo.actual_shift)
        fo["act_rel"] = fo.actual_shift.abs() / fo.kappa0
        # per-map Pearson r, recomputed
        rr = {}
        for p, gdf in fo.groupby("param", sort=False):
            if gdf.pred_shift.std() > 0 and gdf.actual_shift.std() > 0:
                rr[p] = float(np.corrcoef(gdf.actual_shift, gdf.pred_shift)[0, 1])
            else:
                rr[p] = float("nan")
        r_series = pd.Series(rr)
        okr = np.allclose(r_series.reindex(pm.index).fillna(-9).values, pm.first_order_r.fillna(-9).values, atol=1e-9)
        check("derived", f"first_order.csv per-map Pearson r reproduces per_map.csv ({ARCH_NAME[a]})", "match", "match" if okr else "MISMATCH", ok=okr)
        groups = [("all 73 maps", list(pm.index))] + [(gn, list(pm.index[pm.module.map(mod_group) == gn])) for gn in
                                                       ["MLP (fc1, fc2)", "attention (q, k, v, out)", "patch_embedding"]]
        for mod in ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]:
            groups.append((f"module {mod}", list(pm.index[pm.module == mod])))
        for gname, ps in groups:
            sub = fo[fo.param.isin(ps)]
            rs = r_series.reindex(ps)
            ratio = pm.loc[ps, "first_order_ratio_med"]
            pooled = float(np.corrcoef(sub.pred_shift / sub.kappa0, sub.actual_shift / sub.kappa0)[0, 1]) if len(sub) > 2 else float("nan")
            rows.append([ARCH_NAME[a], gname, str(len(ps)), fnum(rs.median(), 3), f"{fnum(rs.min(), 3)} .. {fnum(rs.max(), 3)}",
                         f"{int((rs >= 0.9).sum())}/{int(rs.notna().sum())}", g(ratio.median(), 3), fnum(sub.sign_ok.mean(), 3),
                         g(sub.relerr.median(), 3), g(sub.act_rel.median(), 3), fnum(pooled, 3)])
    add_table("A2i", "First-order rank-2 probe: fit quality against the actual kappa shift (first_order.csv)",
              "Per group of linear maps, how well the first-order prediction kappa0 (u1' D v1/s1 - uq' D vq/sq) of kappa2(W0 + D_i) - kappa2(W0) matches the actual shift across the 8 experts, for ViT-B/32 and ViT-B/16; weights only, dimensionless.",
              [SPEC_DIR["b32"] / "first_order.csv", SPEC_DIR["b32"] / "per_map.csv", SPEC_DIR["b16"] / "first_order.csv", SPEC_DIR["b16"] / "per_map.csv"],
              ["backbone", "group", "n maps", "median per-map Pearson r", "per-map r: min .. max", "# maps with r >= 0.9", "median per-map pred/actual",
               "fraction of (map, expert) with equal sign", "median |pred-actual|/|actual|", "median |actual shift|/kappa0", "pooled Pearson r of (pred/kappa0, actual/kappa0)"],
              rows, align="llrrrrrrrrr",
              notes=["Per-map r is the Pearson correlation over the 8 experts of the prediction and the actual shift (the 'first_order_r' of per_map.csv, reproduced exactly here); the 'pooled' column pools all (map, expert) pairs of the group after dividing by kappa0 so that maps with very different kappa0 are comparable."])


def build_A2_precision() -> None:
    rows = []
    for a in ("b32", "b16"):
        S = spectra(a)
        pr = read_csv(S["dir"] / "precision.csv")
        pr["module"] = pr.param.map(S["M"]["module"])
        pr["grp"] = pr.module.map(mod_group)
        pr["rel"] = (pr.kappa_fp32 - pr.kappa_fp64).abs() / pr.kappa_fp64
        for gname, gdf in [("all", pr)] + [(k, v) for k, v in pr.groupby("module", sort=False)]:
            rows.append([ARCH_NAME[a], gname, str(len(gdf)), g(gdf.rel.median(), 3), g(gdf.rel.quantile(0.9), 3), g(gdf.rel.max(), 3),
                         f"{int((gdf.rel > 0.01).sum())}", f"{int((gdf.rel > 0.1).sum())}", f"{int((gdf.rel > 1.0).sum())}"])
    add_table("A2j", "fp32 vs fp64 SVD: relative error of kappa2 (precision.csv)",
              "For every (linear map, expert) pair, the relative difference |kappa_fp32 - kappa_fp64| / kappa_fp64 between the condition number from an fp32 SVD and from the fp64 SVD of the same stored float32 matrix, summarised by module type, for ViT-B/32 and ViT-B/16; dimensionless.",
              [SPEC_DIR["b32"] / "precision.csv", SPEC_DIR["b16"] / "precision.csv", SPEC_DIR["b32"] / "per_map.csv", SPEC_DIR["b16"] / "per_map.csv"],
              ["backbone", "module group", "n (map, expert)", "median rel. error", "90th percentile", "max", "# > 1 %", "# > 10 %", "# > 100 %"], rows, align="llrrrrrrr")


def build_A2_overlap() -> None:
    rows, mats = [], {}
    for a in ("b32", "b16"):
        S = spectra(a)
        ov = read_csv(S["dir"] / "overlap.csv")
        lcol = [c for c in ov.columns if c.startswith("left_overlap_k")][0]
        rcol = [c for c in ov.columns if c.startswith("right_overlap_k")][0]
        k = int(lcol.split("k")[-1])
        ov["m"], ov["n"] = ov.param.map(S["M"]["m"]), ov.param.map(S["M"]["n"])
        ov["chance_l"], ov["chance_r"] = k / ov.m, k / ov.n
        for gname, gdf in [("all", ov)] + [(kk, v) for kk, v in ov.groupby("module", sort=False)]:
            rows.append([ARCH_NAME[a], gname, str(len(gdf)), fnum(gdf.cos.mean(), 4), fnum(gdf.cos.abs().mean(), 4), fnum(gdf.cos.min(), 3), fnum(gdf.cos.max(), 3),
                         fnum(gdf[lcol].mean(), 4), fnum(gdf["chance_l"].mean(), 4), fnum(gdf[rcol].mean(), 4), fnum(gdf["chance_r"].mean(), 4)])
        mm = np.full((8, 8), np.nan)
        for i, e1 in enumerate(TASKS8):
            for j, e2 in enumerate(TASKS8):
                if i < j:
                    mm[i, j] = ov[(ov.a == e1) & (ov.b == e2)].cos.mean()
        mats[a] = (mm, k)
        trows = []
        for i, e1 in enumerate(TASKS8):
            trows.append([e1] + ["" if (i >= j) else fnum(mm[i, j], 4) for j in range(8)])
        add_table("A2l" if a == "b32" else "A2m", f"Mean cosine between task updates, per expert pair, {ARCH_NAME[a]}",
                  f"Cosine similarity between the flattened task updates D_a and D_b, averaged over the 73 linear maps (unweighted), for each of the 28 expert pairs of the {ARCH_NAME[a]} bank; dimensionless.",
                  [S["dir"] / "overlap.csv"], ["expert a \\ expert b"] + TASKS8, trows, align="l" + "r" * 8)
    add_table("A2k", "Overlap of task updates by module type (overlap.csv)",
              f"For all 28 expert pairs x 73 maps, the cosine between flattened task updates and the mean squared cosine between the top-{k} left (and right) singular subspaces of the two updates, summarised by module type for ViT-B/32 and ViT-B/16, with the chance level k/dim of random subspaces; dimensionless.",
              [SPEC_DIR["b32"] / "overlap.csv", SPEC_DIR["b16"] / "overlap.csv", SPEC_DIR["b32"] / "per_map.csv", SPEC_DIR["b16"] / "per_map.csv"],
              ["backbone", "module group", "n (pair, map)", "mean cos", "mean |cos|", "min cos", "max cos", "mean left top-k overlap", "left chance k/m",
               "mean right top-k overlap", "right chance k/n"], rows, align="llrrrrrrrrr",
              notes=[f"k = {k} (columns {lcol} / {rcol} of scripts/compute_spectra.py); 'overlap' = ||U_a^T U_b||_F^2 / k for the top-k singular vectors of the two updates."])


def build_A2_extras() -> None:
    S = spectra("b32")
    M = S["M"]
    # noise_est.csv
    ne = read_csv(SPEC_DIR["b32"] / "noise_est.csv")
    ne["module"] = ne.param.map(M["module"])
    rows = []
    for gname, gdf in [("all", ne)] + [(k, v) for k, v in ne.groupby("module", sort=False)]:
        rows.append([gname, str(len(gdf)), g(gdf.sigma_hat.median(), 3), fnum(gdf.signal_frac.median(), 3), f"{fnum(gdf.signal_frac.min(), 3)} .. {fnum(gdf.signal_frac.max(), 3)}",
                     g(gdf.n_spikes.median(), 4), g(gdf.snr_top.median(), 3), fnum(gdf.noise_energy.sum() / gdf.total_energy.sum(), 3)])
    for e in TASKS8:
        gdf = ne[ne.expert == e]
        rows.append([f"expert {e}", str(len(gdf)), g(gdf.sigma_hat.median(), 3), fnum(gdf.signal_frac.median(), 3), f"{fnum(gdf.signal_frac.min(), 3)} .. {fnum(gdf.signal_frac.max(), 3)}",
                     g(gdf.n_spikes.median(), 4), g(gdf.snr_top.median(), 3), fnum(gdf.noise_energy.sum() / gdf.total_energy.sum(), 3)])
    add_table("A2n", "Random-matrix noise estimates of the task updates (noise_est.csv), ViT-B/32",
              "Gavish-Donoho noise level sigma_hat, number of spikes above the bulk edge, fraction of update energy in the spikes and top-singular-value signal-to-noise for each (map, expert) task update D_i, summarised by module type and by expert; weights only, ViT-B/32 (no B/16 file exists).",
              [SPEC_DIR["b32"] / "noise_est.csv", SPEC_DIR["b32"] / "per_map.csv"],
              ["group", "n (map, expert)", "median sigma_hat", "median signal_frac", "signal_frac min .. max", "median n_spikes", "median snr_top", "sum(noise energy) / sum(total energy)"],
              rows, align="lrrrrrrr",
              notes=["sigma_hat = median singular value of D / median singular value of an m x n unit-variance Gaussian matrix; spikes = singular values above sigma_hat (sqrt m + sqrt n); signal_frac = energy fraction in spikes (scripts/noise_estimates.py)."])
    # supported_stats.csv
    sp = read_csv(SPEC_DIR["b32"] / "supported_stats.csv")
    sp["grp"] = sp.module
    rows = []
    for gname, sub in [("all", sp)] + [(k, v) for k, v in sp.groupby("module", sort=False)]:
        b = sub[sub.kind == "base_sup"].set_index("param")
        f = sub[sub.kind == "full_sup"]
        sd = f.groupby("param", sort=False).kappa2_raw.std(ddof=0)
        rows.append([gname, str(len(b)), g(b.input_kappa.median(), 3), fnum((b["input_rank_1e-6"] / b.n).median(), 3), g(b.kappa2_raw.median(), 4),
                     g((sd / b.kappa2_raw).median(), 3), str(int((b.rank_default < np.minimum(b.m, b.n)).sum()))])
    add_table("A2o", "Data-supported spectra (supported_stats.csv): kappa of W C^{1/2}, ViT-B/32",
              "Condition numbers of the weights restricted to the calibration-input subspace (W_i C^{1/2}, budget B: pooled unlabelled calibration inputs), summarised by module type; weights plus unlabelled inputs, ViT-B/32 only, dimensionless.",
              [SPEC_DIR["b32"] / "supported_stats.csv"],
              ["module group", "n maps", "median input kappa (C)", "median input rank(1e-6) / input dim", "median base kappa2_raw(W0 C^{1/2})",
               "median rel. across-expert spread of kappa2_raw(W_i C^{1/2}) (sd/base)", "# base maps rank-deficient (default tol)"], rows, align="lrrrrrr",
              notes=["C = pooled calibration second moment of the map's input (base-model activations on the union of the 8 calib splits, no task ids; scripts/compute_supported_spectra.py)."])
    # merged_kappa_grad_a1.csv
    mk = read_csv(SPEC_DIR["b32"] / "merged_kappa_grad_a1.csv")
    mk["module"] = mk.param.map(M["module"])
    rows = []
    for gname, gdf in [("all", mk)] + [(k, v) for k, v in mk.groupby("module", sort=False)]:
        per = gdf.groupby("param", sort=False)
        km = per.kappa_merged.first()
        k0 = M.loc[km.index, "kappa0"]
        sdg = per.grad.std(ddof=0)
        win = per.apply(lambda x: x.loc[x.grad.idxmin(), "expert"], include_groups=False)
        cnt = collections.Counter(win)
        rows.append([gname, str(len(km)), g(km.median(), 4), g((km / k0).median(), 4), g((sdg / km).median(), 3), fnum((per.grad.apply(lambda x: (x > 0).mean())).median(), 3),
                     ", ".join(f"{e} {cnt[e]}" for e in TASKS8 if e in cnt)])
    add_table("A2p", "Gradient of the merged map's kappa2 w.r.t. each expert's weight (merged_kappa_grad_a1.csv)",
              "At the uniform merge (alpha = 1) of the 8 ViT-B/32 experts, the condition number of the merged matrix and the gradient d kappa2(W_merged)/d lambda_i for each expert (the score of the `mkgrad` rule), summarised by module type; weights only, dimensionless.",
              [SPEC_DIR["b32"] / "merged_kappa_grad_a1.csv", SPEC_DIR["b32"] / "per_map.csv"],
              ["module group", "n maps", "median kappa2(merged)", "median kappa2(merged) / kappa0", "median sd_i(grad) / kappa2(merged)", "median fraction of experts with grad > 0",
               "expert with the smallest grad (= largest mkgrad weight): count of maps"], rows, align="lrrrrrl",
              notes=["Gradient formula: alpha * kappa_m * (u1' D_i v1 / s1 - uq' D_i vq / sq) at the merged matrix (scripts/merged_kappa_grad.py)."])


def build_A2() -> None:
    build_A2_permap()
    build_A2_summary()
    build_A2_regimes()
    build_A2_first_order()
    build_A2_precision()
    build_A2_overlap()
    build_A2_extras()


# --------------------------------------------------------------------------------------------
# A3: dev rule sweep (results/sweeps/rule_sweep_dev*.jsonl)
# --------------------------------------------------------------------------------------------
ALPHAS = (1.0, 1.6, 2.0, 2.4, 2.8, 3.2)  # asserted against the data in sweeps()
SWEEP_GLOB = "rule_sweep_dev*.jsonl"
SW: dict = {}


def sweeps() -> dict:
    """Same loading logic as experiments/analyze_sweep.py: sorted shard files, 8-expert records, dict keyed (b, alpha), later file wins."""
    if SW:
        return SW
    files = sorted((RES / "sweeps").glob(SWEEP_GLOB))
    recs = []
    for f in files:
        for r in read_jsonl(f):
            r["_file"] = f.name
            recs.append(r)
    other = [f.name for f in (RES / "sweeps").glob("*.jsonl") if not f.name.startswith("rule_sweep_dev") and not f.name.startswith("subsets_dev_")]
    if other:
        warn(f"unexpected jsonl files in results/sweeps: {other}")
    n_all = len(recs)
    recs = [r for r in recs if len(r["cfg"]["experts"]) == 8]
    assert len(recs) == n_all
    keys = collections.Counter(r["key"] for r in recs)
    by: dict = collections.defaultdict(dict)
    dups = 0
    for r in recs:
        c = r["cfg"]
        k = (c["b"], c["alpha"])
        if k in by[c["rule"]]:
            dups += 1
        by[c["rule"]][k] = r
    assert all(c["split"] == "dev" and c["bias_policy"] == "tied" for c in (r["cfg"] for r in recs))
    assert tuple(sorted({k[1] for k in by["kappa"]})) == ALPHAS, "alpha grid differs from the constant used for table layout"
    SW.update({"files": files, "recs": recs, "by": dict(by), "dups": dups, "n": len(recs),
               "bs": sorted({k[0] for d in by.values() for k in d if k[0] > 0}, key=lambda x: (math.isinf(x), x))})
    return SW


def n_eff_of(r: dict) -> float | None:
    cs = r.get("coeff_summary") or {}
    return cs.get("n_eff_mean")


def lam_bar_neff(r: dict) -> float:
    lb = np.array(list(r["lambda_bar"].values()), dtype=float)
    p = lb[lb > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def rule_n_eff(rule: str, r: dict) -> tuple[float, bool]:
    """(n_eff_mean over maps, derived?)  Uniform / MetaGPT have no coeff_summary: derive from lambda_bar (global weights, one lambda for all maps)."""
    v = n_eff_of(r)
    if v is not None:
        return float(v), False
    return lam_bar_neff(r), True


def best_of(d: dict, metric: str = "mean_norm", alpha: float | None = None):
    items = [(k, r) for k, r in d.items() if alpha is None or k[1] == alpha]
    if not items:
        return None
    best = max((r for _, r in items), key=lambda r: r["metrics"][metric])
    ties = sum(1 for _, r in items if abs(r["metrics"][metric] - best["metrics"][metric]) < 1e-12)
    return best, ties


def bdisp(rule: str, r: dict) -> str:
    if rule in ("uniform", "metagpt"):
        return "-"
    b = r["cfg"]["b"]
    return f"beta={b:g} (abs.)" if r["cfg"].get("abs_beta") else bfmt(b)


def rule_summary(rule: str) -> dict:
    d = sweeps()["by"][rule]
    best, ties = best_of(d)
    b1, ties1 = best_of(d, alpha=1.0)
    scale = best["cfg"].get("scale")
    return {"rule": rule, "family": family(rule), "n": len(d), "best": best, "ties": ties, "best1": b1, "ties1": ties1,
            "scale": scale, "abs": bool(best["cfg"].get("abs_beta"))}


def build_A3() -> None:
    S = sweeps()
    by, BS = S["by"], S["bs"]
    rules = sorted(by, key=lambda r: (FAMILY_ORDER.index(family(r)), r))
    sums = {r: rule_summary(r) for r in rules}
    uni = sums["uniform"]
    u_best, u_a1 = uni["best"]["metrics"]["mean_norm"], uni["best1"]["metrics"]["mean_norm"]
    # ---- A3a: best over the grid
    rows = []
    for fam in FAMILY_ORDER:
        rs = [r for r in rules if family(r) == fam]
        rs.sort(key=lambda r: -sums[r]["best"]["metrics"]["mean_norm"])
        for r in rs:
            x = sums[r]
            bm = x["best"]
            ne, der = rule_n_eff(r, bm)
            grid = ("6 alphas" if r in ("uniform", "metagpt") else
                    (f"{len({k[0] for k in by[r]})} abs. betas x {len({k[1] for k in by[r]})} alphas" if x["abs"] else
                     f"{len({k[0] for k in by[r]})} b x {len({k[1] for k in by[r]})} alphas"))
            rows.append([fam.split(" ", 1)[1], r, str(x["n"]), grid, g(x["scale"], 4) if x["scale"] is not None else "-", bdisp(r, bm), afmt(bm["cfg"]["alpha"]),
                         pct(bm["metrics"]["mean_norm"]), pct(bm["metrics"]["mean_acc"]), pct(bm["metrics"]["min_norm"]),
                         fnum(ne, 2) + ("*" if der else ""), str(x["ties"]), pp(bm["metrics"]["mean_norm"] - u_best)])
    add_table("A3a", "Dev rule sweep: best (b, alpha) over the shared grid, every rule with 8 experts",
              f"For each of the {len(rules)} rules in the 8-expert dev sweep, the configuration with the highest dev mean_norm over its (b, alpha) grid, with its mean_norm (mean over tasks of acc / own-expert acc, %), mean_acc (%), min_norm (%) and coefficient concentration n_eff; "
              f"dev split (20 % of each test set), {S['n']} records.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"],
              ["family", "rule", "# configs", "grid", "scale (beta = b/scale)", "best b", "best alpha", "mean_norm (%)", "mean_acc (%)", "min_norm (%)",
               "n_eff (mean over maps) at best", "# configs tied at best", "diff vs uniform best (pp)"], rows, align="llrlrrrrrrrrr",
              notes=[f"Grid: b in {{{', '.join(bfmt(b) for b in BS)}}} x alpha in {{{', '.join(afmt(a) for a in ALPHAS)}}} = {len(BS) * len(ALPHAS)} configs per beta-rule; uniform and metagpt have no beta ({len(ALPHAS)} alphas). "
                     f"gd_invvar and global:gd_invvar use ABSOLUTE beta in {{{', '.join(f'{x:g}' for x in sorted({k[0] for k in by['gd_invvar']}))}}} (abs_beta = true; {len(by['gd_invvar'])} configs), so their 'best b' is a beta value, not a multiplier. "
                     "beta = b / scale with scale = median over maps of the across-expert s.d. of the rule's scores (cfg 'scale'); b = inf is per-map argmin selection. Ties are resolved as experiments/analyze_sweep.py does (first maximum in file order); the tie count shows whether another config has an identical mean_norm. "
                     "* n_eff not recorded for uniform/metagpt: derived as exp(H(lambda_bar)) from the recorded global lambda (uniform = 8 by definition). 'diff vs uniform best' = this rule's best minus the best of uniform over its alpha grid (the Task-Arithmetic baseline).",
                     "Full per-config data are in the shard files; per-rule full grids are summarised here only (the kappa and uniform grids are in A3d)."])
    # ---- A3b: alpha = 1
    rows = []
    for fam in FAMILY_ORDER:
        rs = [r for r in rules if family(r) == fam and sums[r]["best1"] is not None]
        rs.sort(key=lambda r: -sums[r]["best1"]["metrics"]["mean_norm"])
        for r in rs:
            x = sums[r]
            b1 = x["best1"]
            ne, der = rule_n_eff(r, b1)
            n1 = sum(1 for (b, a) in by[r] if a == 1.0)
            rows.append([fam.split(" ", 1)[1], r, str(n1), bdisp(r, b1), pct(b1["metrics"]["mean_norm"]), pct(b1["metrics"]["mean_acc"]), pct(b1["metrics"]["min_norm"]),
                         fnum(ne, 2) + ("*" if der else ""), str(x["ties1"]), pp(b1["metrics"]["mean_norm"] - u_a1)])
    add_table("A3b", "Dev rule sweep at alpha = 1 (the literal normalised average): best b per rule",
              "For each rule, the b with the highest dev mean_norm at alpha = 1 (W = sum_i lambda_i W_i), with mean_norm, mean_acc, min_norm (all %), n_eff and the difference to uniform at alpha = 1; dev split.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"],
              ["family", "rule", "# alpha=1 configs", "best b at alpha = 1", "mean_norm (%)", "mean_acc (%)", "min_norm (%)", "n_eff (mean over maps)", "# tied", "diff vs uniform alpha=1 (pp)"],
              rows, align="llrrrrrrrr",
              notes=["This is the quantity of the 'alpha = 1' column in REPORT_01 §2. 'best b' for gd_invvar / global:gd_invvar is an absolute beta. * as in A3a."])
    # ---- A3c: shuffled nulls
    rows = []
    null_defs = [("kappa", [f"shuf{s}:kappa" for s in range(5)]), ("logkappa_q90_rev", [f"shuf{s}:logkappa_q90_rev" for s in range(5)]),
                 ("global:kappa_z", [f"gshuf{s}:global:kappa_z" for s in range(3)])]
    NULLS = {}
    for base_rule, nrs in null_defs:
        obs = sums[base_rule]["best"]["metrics"]["mean_norm"]
        obs1 = sums[base_rule]["best1"]["metrics"]["mean_norm"]
        v = np.array([sums[n]["best"]["metrics"]["mean_norm"] for n in nrs])
        v1 = np.array([sums[n]["best1"]["metrics"]["mean_norm"] for n in nrs])
        NULLS[base_rule] = {"rules": nrs, "tuned": v, "a1": v1, "obs": obs, "obs1": obs1}
        rows.append([base_rule, f"{nrs[0]} .. {nrs[-1]}", str(len(nrs)), pct(obs), pct(v.mean()), f"{100 * v.std(ddof=0):.2f}", f"{100 * v.std(ddof=1):.2f}",
                     pct(v.min()), pct(v.max()), f"{int((v >= obs).sum())}/{len(v)}", pp(obs - v.mean()),
                     pct(obs1), pct(v1.mean()), f"{pct(v1.min())} .. {pct(v1.max())}", f"{int((v1 >= obs1).sum())}/{len(v1)}"])
    add_table("A3c", "Shuffled-score nulls: dev-best of the observed rule against its shuffles",
              "For each base rule, the dev-best mean_norm (%) of the observed scores next to the same quantity for its shuffled-score nulls (scores permuted across experts: independently per map for shuf, one expert permutation for all maps for gshuf), "
              "tuned over the full (b, alpha) grid and at alpha = 1; dev split, units %.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"],
              ["base rule", "null rules", "# nulls", "observed best (%)", "null mean (%)", "null s.d., ddof=0", "null s.d., ddof=1", "null min (%)", "null max (%)",
               "# nulls >= observed", "observed - null mean (pp)", "observed best at alpha=1 (%)", "null mean at alpha=1 (%)", "null range at alpha=1 (%)", "# nulls >= observed at alpha=1"],
              rows, align="llrrrrrrrrrrrrr",
              notes=["analyze_sweep.py reports the s.d. with ddof=0 (the '+-0.19' of REPORT_01 is that value); both are given. The 'gshuf' nulls in the dev sweep are shuffles of global:kappa_z (not of global:logkappa).",
                     "Only shuf*:kappa, shuf*:logkappa_q90_rev and gshuf*:global:kappa_z exist as dev sweeps; B/32 test has shuf*:kappa and gshuf*:global:kappa_z only (A4)."])
    # ---- A3d: full grids for kappa and uniform
    for metric, mt in (("mean_norm", "mean_norm (%)"), ("mean_acc", "mean_acc (%)"), ("min_norm", "min_norm (%)")):
        d = by["kappa"]
        allv = [(r["metrics"][metric], k) for k, r in d.items()]
        gmax = max(allv)[0]
        rows = []
        for b in BS:
            row = [bfmt(b)]
            for a in ALPHAS:
                v = d[(b, a)]["metrics"][metric]
                row.append(("**" + pct(v) + "**") if v == gmax else pct(v))
            rows.append(row)
        sid = {"mean_norm": "A3d", "mean_acc": "A3e", "min_norm": "A3f"}[metric]
        add_table(sid, f"Literal rule `kappa`: full b x alpha grid of {metric}",
                  f"Dev {metric} (%) of the literal rule lambda_i = softmax(-beta kappa2(W_i)) for every b (rows; beta = b/scale, scale = {g(d[(BS[0], 1.0)]['cfg']['scale'], 6)}) and alpha (columns) of the shared grid; dev split.",
                  [RES / "sweeps" / "rule_sweep_dev_literal.jsonl"], ["b \\ alpha"] + [afmt(a) for a in ALPHAS], rows,
                  notes=["Bold = grid maximum." if metric == "mean_norm" else "Bold = maximum of this metric over the grid."])
    d = by["uniform"]
    rows = []
    for metric, lab in (("mean_norm", "mean_norm (%)"), ("mean_acc", "mean_acc (%)"), ("min_norm", "min_norm (%)")):
        rows.append([lab] + [pct(d[(0.0, a)]["metrics"][metric]) for a in ALPHAS])
    rows.append(["Task-Arithmetic scale s = alpha / 8"] + [f"{a / 8:.3f}" for a in ALPHAS])
    add_table("A3g", "Uniform averaging (beta = 0): dev metrics over the alpha grid",
              "Dev mean_norm, mean_acc and min_norm (%) of uniform lambda_i = 1/8 (Task Arithmetic with scale s = alpha/8) for each alpha of the grid; dev split.",
              [RES / "sweeps" / "rule_sweep_dev.jsonl"], ["metric \\ alpha"] + [afmt(a) for a in ALPHAS], rows)
    # ---- A3h: kappa vs its null across the whole grid
    d = by["kappa"]
    nr = null_defs[0][1]
    rows = []
    for b in BS:
        row = [bfmt(b)]
        for a in ALPHAS:
            k = d[(b, a)]["metrics"]["mean_norm"]
            nv = np.array([by[n][(b, a)]["metrics"]["mean_norm"] for n in nr])
            row.append(f"{100 * (k - nv.mean()):+.2f} ({int((nv >= k).sum())}/5)")
        rows.append(row)
    add_table("A3h", "Literal rule against its 5 shuffled-score nulls, cell by cell",
              "For every (b, alpha) cell, dev mean_norm of `kappa` minus the mean of `shuf0..4:kappa` at the same cell, in percentage points, with the number of the 5 shuffles that score at least as high as `kappa` in parentheses; dev split.",
              [RES / "sweeps" / "rule_sweep_dev_literal.jsonl", RES / "sweeps" / "rule_sweep_dev_shuffled.jsonl"], ["b \\ alpha"] + [afmt(a) for a in ALPHAS], rows,
              notes=["A cell reads '+0.12 (2/5)': kappa is 0.12 pp above the null mean and 2 of the 5 shuffles are >= kappa."])
    # ---- A3i: alpha = 1 curves of all rules by b
    rows = []
    for fam in FAMILY_ORDER:
        for r in sorted([r for r in rules if family(r) == fam and not sums[r]["abs"] and r not in ("uniform", "metagpt")]):
            rows.append([r] + [pct(by[r][(b, 1.0)]["metrics"]["mean_norm"]) for b in BS])
    rows.append(["uniform (b=0)"] + [pct(by["uniform"][(0.0, 1.0)]["metrics"]["mean_norm"])] + [""] * (len(BS) - 1))
    rows.append(["metagpt (no beta)"] + [pct(by["metagpt"][(0.0, 1.0)]["metrics"]["mean_norm"])] + [""] * (len(BS) - 1))
    add_table("A3i", "Dev mean_norm at alpha = 1 as a function of b, every rule (all rules; the alpha = 1 panel of results/figures/pilot_rule_vs_b.png plus the others)",
              "For every beta-rule, dev mean_norm (%) at alpha = 1 for each temperature multiplier b (inf = per-map argmin); dev split.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"], ["rule"] + [f"b={bfmt(b)}" for b in BS], rows,
              notes=["The uniform and metagpt rows have a single alpha = 1 value (no b). gd_invvar and global:gd_invvar (absolute beta) are in A3k."])
    # ---- A3o: best over alpha at each b
    rows = []
    for fam in FAMILY_ORDER:
        for r in sorted([r for r in rules if family(r) == fam and not sums[r]["abs"] and r not in ("uniform", "metagpt")]):
            cells = []
            for b in BS:
                bm, _ = best_of({k: v for k, v in by[r].items() if k[0] == b})
                cells.append(f"{pct(bm['metrics']['mean_norm'])} ({afmt(bm['cfg']['alpha'])})")
            rows.append([r] + cells)
    bu, _ = best_of(by["uniform"])
    rows.append(["uniform (b=0)", f"{pct(bu['metrics']['mean_norm'])} ({afmt(bu['cfg']['alpha'])})"] + [""] * (len(BS) - 1))
    bg, _ = best_of(by["metagpt"])
    rows.append(["metagpt (no beta)", f"{pct(bg['metrics']['mean_norm'])} ({afmt(bg['cfg']['alpha'])})"] + [""] * (len(BS) - 1))
    add_table("A3o", "Dev mean_norm, best over alpha at each b, every rule (the 'best alpha per b' panel of results/figures/pilot_rule_vs_b.png and all other rules)",
              "For every beta-rule and temperature multiplier b, the highest dev mean_norm (%) over the alpha grid with the maximising alpha in parentheses; dev split.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"], ["rule"] + [f"b={bfmt(b)}" for b in BS], rows,
              notes=["uniform and metagpt have a single value (alpha tuned, no b). gd_invvar and global:gd_invvar (absolute beta) are in A3k."])
    # ---- A3j: n_eff by b (alpha-independent)
    rows = []
    for fam in FAMILY_ORDER:
        for r in sorted([r for r in rules if family(r) == fam and not sums[r]["abs"] and r not in ("uniform", "metagpt")]):
            rows.append([r] + [fnum(by[r][(b, 1.0)]["coeff_summary"]["n_eff_mean"], 2) for b in BS])
    add_table("A3j", "Coefficient concentration: n_eff (mean over the 73 maps) as a function of b, every rule",
              "For every beta-rule, the effective number of experts n_eff = exp(H(lambda)) of the per-map coefficients, averaged over the 73 linear maps, for each b (it does not depend on alpha); 8 = uniform, 1 = one expert per map; dev sweep records, dimensionless.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"], ["rule"] + [f"b={bfmt(b)}" for b in BS], rows)
    # ---- A3k: absolute-beta rules
    rows = []
    for r in [r for r in rules if sums[r]["abs"]]:
        for bb in sorted({k[0] for k in by[r]}):
            for a in ALPHAS:
                rec = by[r][(bb, a)]
                rows.append([r, f"{bb:g}", afmt(a), pct(rec["metrics"]["mean_norm"]), pct(rec["metrics"]["mean_acc"]), pct(rec["metrics"]["min_norm"]),
                             fnum(rec["coeff_summary"]["n_eff_mean"], 2)])
    add_table("A3k", "Inverse-variance (Gavish-Donoho) rules with ABSOLUTE beta: all 36 configurations",
              "For gd_invvar and global:gd_invvar (score c = 2 log sigma_hat, so beta = 1 means lambda proportional to sigma_hat^-2), every absolute beta in {0.5, 1, 2} x alpha configuration of the dev sweep: mean_norm, mean_acc, min_norm (%) and n_eff; dev split.",
              [RES / "sweeps" / "rule_sweep_dev_w3abs.jsonl"], ["rule", "absolute beta", "alpha", "mean_norm (%)", "mean_acc (%)", "min_norm (%)", "n_eff (mean over maps)"], rows, align="lrrrrrr",
              notes=["beta = 1 is the 'derived' inverse-variance temperature of MATH_NOTES N7."])
    # ---- A3l: kappa concentration and lambda_bar by b
    d = by["kappa"]
    rows = []
    for b in BS:
        r = d[(b, 1.0)]
        cs, lb = r["coeff_summary"], r["lambda_bar"]
        rows.append([bfmt(b), g(r["beta"], 4), fnum(cs["n_eff_mean"], 2), fnum(cs["n_eff_median"], 2), fnum(cs["n_eff_min"], 2), fnum(cs["max_mean"], 3), fnum(cs["max_max"], 3)]
                    + [fnum(lb[e], 4) for e in TASKS8])
    add_table("A3l", "Literal rule `kappa`: coefficient concentration and average expert weights as a function of b",
              "For each b, the recorded concentration of the literal rule's per-map coefficients (n_eff mean/median/min over the 73 maps, mean and max over maps of the largest lambda) and lambda_bar = parameter-weighted mean lambda of each expert; dev sweep records (alpha-independent), dimensionless.",
              [RES / "sweeps" / "rule_sweep_dev_literal.jsonl"],
              ["b", "beta = b/scale", "n_eff mean", "n_eff median", "n_eff min", "max lambda, mean over maps", "max lambda, max over maps"] + [f"lambda_bar {e}" for e in TASKS8],
              rows, align="l" + "r" * 14,
              notes=["Which maps are concentrated is shown in A3n: the 24 MLP maps and the patch-embedding map stay at n_eff = 8 for every finite b, so all of the concentration comes from attention maps."])
    # ---- A3m: consistency with the stored summaries
    chk_rows = []
    for fn in ("summary_wave1.json", "summary_w1w2.json"):
        st = read_json(RES / "sweeps" / fn)
        diffs = []
        for rule, v in st.items():
            if rule not in sums:
                diffs.append(f"{rule}: not in recomputed sweeps")
                continue
            x = sums[rule]
            bm = x["best"]
            same = (math.isclose(v["best"]["mean_norm"], bm["metrics"]["mean_norm"], abs_tol=1e-12) and v["best"]["b"] == bm["cfg"]["b"]
                    and v["best"]["alpha"] == bm["cfg"]["alpha"] and math.isclose(v["best"]["mean_acc"], bm["metrics"]["mean_acc"], abs_tol=1e-12)
                    and math.isclose(v["best"]["min_norm"], bm["metrics"]["min_norm"], abs_tol=1e-12)
                    and ((v["best_alpha1"] is None and x["best1"] is None) or (v["best_alpha1"]["b"] == x["best1"]["cfg"]["b"]
                         and math.isclose(v["best_alpha1"]["mean_norm"], x["best1"]["metrics"]["mean_norm"], abs_tol=1e-12))))
            if v["n_configs"] != x["n"]:
                diffs.append(f"{rule}: stored n_configs {v['n_configs']} vs {x['n']} in the final shards" + ("" if same else " (and best differs)"))
            elif not same:
                diffs.append(f"{rule}: best / best-at-alpha-1 differ")
        missing = [r for r in sums if r not in st]
        chk_rows.append([fn, str(len(st)), str(len(st) - len(diffs)), "; ".join(diffs) if diffs else "none", f"{len(missing)} rules not covered: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")])
    add_table("A3m", "Consistency of the stored sweep summaries with the final shard files",
              "For each stored summary file, whether every rule's best (b, alpha), metrics, best-at-alpha-1 and config count equal what this script recomputes from the final shards; dev split, no new numbers.",
              [RES / "sweeps" / "summary_wave1.json", RES / "sweeps" / "summary_w1w2.json", RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)"],
              ["stored file", "# rules in file", "# rules identical", "differences", "recomputed rules not in the file"], chk_rows, align="lrrll",
              notes=["The stored summaries are snapshots written by experiments/analyze_sweep.py --json-out while the campaign was running; entries whose config count is below the final count were written from partial shards. "
                     "The frozen configurations (A4d) were checked against the final shards, not against these files."])
    # --- doc cross-checks from the sweep
    k = sums["kappa"]
    u = sums["uniform"]
    mg = sums["metagpt"]
    R1 = "REPORT_01 §2 / FULL §5"
    check(R1, "uniform at alpha = 1: mean_norm (%)", "74.80", pct(u["best1"]["metrics"]["mean_norm"]))
    check(R1, "uniform best: mean_norm (%) (alpha 1.6)", "77.65", pct(u["best"]["metrics"]["mean_norm"]))
    check(R1, "uniform best: alpha", "1.6", afmt(u["best"]["cfg"]["alpha"]))
    check(R1, "uniform best: mean_acc (%)", "70.28", pct(u["best"]["metrics"]["mean_acc"]))
    check(R1, "kappa at alpha = 1, best mean_norm (%)", "74.80", pct(k["best1"]["metrics"]["mean_norm"]))
    check(R1, "kappa at alpha = 1, argmax b (report says b = 2)", "2", bfmt(k["best1"]["cfg"]["b"]),
          note=f"value at b=2 is {pct(by['kappa'][(2.0, 1.0)]['metrics']['mean_norm'])}, at b=4 {pct(by['kappa'][(4.0, 1.0)]['metrics']['mean_norm'])} (frozen kappa|alpha1 uses b=4)")
    check(R1, "kappa best over grid: mean_norm (%)", "77.74", pct(k["best"]["metrics"]["mean_norm"]))
    check(R1, "kappa best over grid: (b, alpha)", "(0.25, 1.6)", f"({bfmt(k['best']['cfg']['b'])}, {afmt(k['best']['cfg']['alpha'])})")
    v1 = NULLS["kappa"]["a1"]
    check(R1, "5 shuffled kappa, alpha = 1 range (%)", "74.65-75.03", f"{pct(v1.min())}-{pct(v1.max())}")
    vt = NULLS["kappa"]["tuned"]
    check(R1, "5 shuffled kappa, best-over-grid mean (%)", "77.74", pct(vt.mean()))
    check(R1, "5 shuffled kappa, best-over-grid s.d. (ddof=0)", "0.19", f"{100 * vt.std(ddof=0):.2f}")
    check(R1, "5 shuffled kappa, best-over-grid range (%)", "77.54-78.10", f"{pct(vt.min())}-{pct(vt.max())}")
    for rule, c1, ct in [("kappa_rev", "74.81", "77.35"), ("logkappa", "74.88", "77.71"), ("kappa_strict", "74.88", "77.52"), ("kappa_z", "74.40", "76.73"),
                         ("global:kappa_z", "74.60", "76.65"), ("metagpt", "74.55", "76.98"), ("dnorm", "74.00", "77.26")]:
        check(R1, f"{rule}: alpha = 1 best (%)", c1, pct(sums[rule]["best1"]["metrics"]["mean_norm"]))
        check(R1, f"{rule}: best over grid (%)", ct, pct(sums[rule]["best"]["metrics"]["mean_norm"]))
    finite = [b for b in BS if not math.isinf(b)]
    vals = [by["kappa"][(b, 1.0)]["metrics"]["mean_norm"] for b in finite]
    check(R1, "kappa at alpha = 1 is flat from b = 0.25 to 128: range (%)", "74.6-74.8", f"{100 * min(vals):.1f}-{100 * max(vals):.1f}")
    check(R1, "n_eff of kappa at b = 0.25 (of 8)", "6.9", f"{by['kappa'][(0.25, 1.0)]['coeff_summary']['n_eff_mean']:.1f}")
    check(R1, "n_eff of kappa at b = 128 (of 8)", "3.6", f"{by['kappa'][(128.0, 1.0)]['coeff_summary']['n_eff_mean']:.1f}")
    check(R1, "kappa argmin limit (b = inf, alpha = 1) mean_norm (%)", "65.0", f"{100 * by['kappa'][(INF, 1.0)]['metrics']['mean_norm']:.1f}")
    kz4 = by["kappa_z"][(4.0, 1.0)]["metrics"]["mean_norm"]
    kz8 = by["kappa_z"][(8.0, 1.0)]["metrics"]["mean_norm"]
    check(R1, "kappa_z at b = 4 (alpha = 1) mean_norm (%)", "66.1", f"{100 * kz4:.1f}", note=f"b=8 gives {100 * kz8:.1f}; the quoted 66.1 matches b=8, not b=4")
    dr = by["dnorm_rev"]
    check(R1, "per-map largest-update selection (dnorm_rev, b = inf, alpha = 1) mean_norm (%)", "36", f"{100 * dr[(INF, 1.0)]['metrics']['mean_norm']:.0f}",
          note="dnorm_rev b=inf: alpha 1 / 1.6 / 2.0 = " + " / ".join(f"{100 * dr[(INF, a)]['metrics']['mean_norm']:.1f}" for a in (1.0, 1.6, 2.0))
          + "; dnorm (smallest update) b=inf: " + " / ".join(f"{100 * by['dnorm'][(INF, a)]['metrics']['mean_norm']:.1f}" for a in (1.0, 1.6, 2.0))
          + f"; 36.2 occurs only for dnorm_rev at b=8, alpha=2.0 ({100 * dr[(8.0, 2.0)]['metrics']['mean_norm']:.1f})")
    R2 = "RESEARCH_LOG 10-18 setup"
    check(R2, "uniform alpha = 1: mean_acc (%)", "67.3", f"{100 * u['best1']['metrics']['mean_acc']:.1f}")
    check(R2, "uniform best: mean_norm / mean_acc (%)", "77.7 / 70.3", f"{100 * u['best']['metrics']['mean_norm']:.1f} / {100 * u['best']['metrics']['mean_acc']:.1f}")
    check(R2, "MetaGPT alpha = 1: mean_norm / mean_acc (%)", "74.6 / 67.2", f"{100 * mg['best1']['metrics']['mean_norm']:.1f} / {100 * mg['best1']['metrics']['mean_acc']:.1f}")
    check(R2, "MetaGPT best (alpha 1.6): mean_norm / mean_acc (%)", "77.0 / 69.8", f"{100 * mg['best']['metrics']['mean_norm']:.1f} / {100 * mg['best']['metrics']['mean_acc']:.1f}")
    check(R2, "Task-Arithmetic scale s = alpha / K at the best alpha", "0.2", f"{u['best']['cfg']['alpha'] / 8:.1f}")
    gd = by["gd_invvar"]
    check("RESEARCH_LOG wave 2/3 / FULL §9", "Gavish-Donoho inverse-variance rule at its derived beta = 1: dev mean_norm (%) ('75.5 dev')", "75.5",
          f"{100 * max(gd[(1.0, a)]['metrics']['mean_norm'] for a in ALPHAS):.1f}",
          note=f"75.5 is the best over beta in {{0.5,1,2}} x alpha (beta = 0.5, alpha 1.6: {pct(gd[(0.5, 1.6)]['metrics']['mean_norm'])}); at beta = 1 alpha=1 it is {pct(gd[(1.0, 1.0)]['metrics']['mean_norm'])}, alpha=1.6 {pct(gd[(1.0, 1.6)]['metrics']['mean_norm'])}")
    nonnull = [r for r in rules if not r.startswith(("shuf", "gshuf")) and r not in ("uniform",)]
    check("REPORT_01 §5 / FULL §7,9", "'about 45' rules / variants: number of distinct non-null, non-uniform rules in the dev sweeps", "~45", str(len(nonnull)),
          ok=abs(len(nonnull) - 45) <= 5, approx=True, note="41 including metagpt; 5 shuf*:kappa, 5 shuf*:logkappa_q90_rev and 3 gshuf* nulls and uniform are excluded")
    n66 = sum(1 for r in rules if sums[r]["n"] == 66)
    abs_rules = [r for r in rules if sums[r]["abs"]]
    check("REPORT_01 §1 / FULL §1", "'Every rule gets the same 66-config grid' (uniform and MetaGPT tune alpha only: 6 configs)", "all beta-rules: 66 configs",
          f"{n66} of {len(rules) - 2} beta-rules have 66; " + ", ".join(f"{r} {sums[r]['n']}" for r in abs_rules) + " (absolute beta in {0.5, 1, 2} x 6 alphas)", ok=(len(abs_rules) == 0),
          note="minor: the Gavish-Donoho rules gd_invvar and global:gd_invvar use an absolute-beta grid of 18 configs; the other 49 beta-rules have the shared 66")
    sup = [r for r in rules if family(r).startswith("7 ") or r.startswith("wc:sd") or r.startswith("global:s")]
    check("REPORT_01 §1", "'All rules are weights-only (A)'", "0 rules using calibration inputs", f"{len(sup)} of {len(rules)} swept rules use the pooled calibration second moment (budget B): " + ", ".join(sup),
          ok=(len(sup) == 0), note="minor wording: true for the literal rule and its direct variants; scores.py labels the s*/sd* scores 'data-supported statistics, budget (B)'")
    mono, rise = {}, {}
    for r in ("kappa_z", "dnorm", "dnorm_rev", "global:kappa_z"):
        v = [by[r][(b, 1.0)]["metrics"]["mean_norm"] for b in BS]
        rise[r] = max([v[i + 1] - v[i] for i in range(len(v) - 1)] + [0.0])
        mono[r] = rise[r] <= 1e-12
    soft = all(x <= 0.001 for x in rise.values())  # rises of at most 0.1 pp = plateau noise
    check("REPORT_01 §2 / FULL §5", "rules that concentrate every map degrade monotonically with b at alpha = 1 (kappa_z, dnorm, dnorm_rev, global:kappa_z)", "monotone decrease",
          ", ".join(f"{r}: {'monotone' if m else 'largest rise ' + pp(rise[r]) + ' pp'}" for r, m in mono.items()), ok=soft, approx=not all(mono.values()),
          note="strictly monotone except for plateau fluctuations of at most 0.1 pp once the coefficients are one-hot (large b)")
    SW["sums"], SW["NULLS"] = sums, NULLS


# --------------------------------------------------------------------------------------------
# A4 / A5: frozen test (B/32) and pre-registered transfer (B/16)
# --------------------------------------------------------------------------------------------
import datetime as _dt


def git_first_commit_time(path: str) -> str | None:
    """ISO committer date of the commit that first added `path` (read-only git log)."""
    try:
        out = subprocess.run(["git", "--no-optional-locks", "log", "--diff-filter=A", "--format=%cI", "--", path], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
        return out[-1] if out else None
    except Exception:
        return None


def parse_local(ts: str) -> _dt.datetime:
    """Record timestamps are local wall-clock strings without zone; the commits show -04:00, which we assume for the records."""
    d = _dt.datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone(_dt.timedelta(hours=-4)))
    return d


def reproduce_bootstrap(recs: list[dict], ref_tag: str, exp_test: dict, n_boot: int = 2000) -> dict | None:
    """Re-run experiments/compare_frozen.py (paired task-stratified bootstrap, rng seed 0) from the per-example correctness arrays.
    Uses bootstrap counts @ correctness instead of fancy-indexing; the draws are identical (same rng stream, same order)."""
    by_tag = {}
    for r in recs:
        by_tag[r["cfg"]["tag"]] = r
    ref = by_tag[ref_tag]
    ref_c = load_correct(ref["key"])
    if ref_c is None:
        return None
    tasks = list(ref_c)
    arrs = {}
    for tag, r in by_tag.items():
        c = load_correct(r["key"])
        if c is None or set(c) != set(tasks):
            continue
        arrs[tag] = c
    tags = sorted(arrs)
    rng = np.random.default_rng(0)
    acc_diff = {tag: np.zeros(n_boot) for tag in tags}
    point = {tag: 0.0 for tag in tags}
    for t in tasks:
        n_t = len(ref_c[t])
        idx = rng.integers(0, n_t, size=(n_boot, n_t))
        counts = np.empty((n_boot, n_t), dtype=np.float32)
        for k in range(n_boot):
            counts[k] = np.bincount(idx[k], minlength=n_t)
        del idx
        C = np.stack([arrs[tag][t].astype(np.float32) for tag in tags], axis=1)  # (n_t, n_tags)
        boot = (counts @ C).astype(np.float64) / n_t  # (n_boot, n_tags)
        del counts
        ref_boot = boot[:, tags.index(ref_tag)]
        for j, tag in enumerate(tags):
            acc_diff[tag] += (boot[:, j] - ref_boot) / exp_test[t]
            point[tag] += (arrs[tag][t].mean() - ref_c[t].mean()) / exp_test[t]
    out = {}
    for tag in tags:
        d = acc_diff[tag] / len(tasks)
        out[tag] = {"diff": point[tag] / len(tasks), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def pertask_rows(recs_by_tag: dict, tags: list[str], exp_test: dict, zs_test: dict, norm: bool) -> list[list]:
    rows = []
    for tag in tags:
        r = recs_by_tag[tag]
        acc = r["metrics"]["acc"]
        vals = [acc[t] / exp_test[t] if norm else acc[t] for t in TASKS8]
        mean_acc = np.mean([acc[t] for t in TASKS8])
        mean_norm = np.mean([acc[t] / exp_test[t] for t in TASKS8])
        min_norm = min(acc[t] / exp_test[t] for t in TASKS8)
        # consistency with the stored metrics
        assert math.isclose(mean_acc, r["metrics"]["mean_acc"], abs_tol=1e-9) and math.isclose(mean_norm, r["metrics"]["mean_norm"], abs_tol=1e-9) \
            and math.isclose(min_norm, r["metrics"]["min_norm"], abs_tol=1e-9), tag
        rows.append([tag] + [pct(v) for v in vals] + [pct(np.mean(vals)) if norm else pct(mean_acc), pct(mean_norm), pct(min_norm)])
    return rows


def build_A4() -> None:
    comp = read_json(RES / "freeze" / "test_comparison.json")
    recs = freeze_b32_records()
    frozen = {w: read_json(RES / "freeze" / f"frozen_configs_{w}.json") for w in ("w1", "w2", "w3")}
    ref = comp["ref"]
    by_tag = {}
    for r in recs:
        by_tag[r["cfg"]["tag"]] = r
    where = collections.defaultdict(list)
    for w, lst in frozen.items():
        for c in lst:
            where[c["tag"]].append(w)
    res = comp["results"]
    cnt0 = collections.Counter(r["cfg"]["tag"] for r in recs)
    dups0 = sorted(t for t, n in cnt0.items() if n > 1)
    rows = []
    for tag, v in sorted(res.items(), key=lambda kv: (-kv[1]["diff_vs_ref"], kv[0])):
        cfg = by_tag[tag]["cfg"]
        rule, var = parse_variant_tag(tag)
        lo, hi = v["ci95"]
        sig = "+" if lo > 0 else ("-" if hi < 0 else "")
        dev = v["dev_metric"]
        b = cfg["b"]
        bd = "-" if cfg["rule"] in ("uniform", "metagpt") or str(cfg["rule"]).startswith("single:") else (f"beta={b:g} (abs.)" if cfg.get("abs_beta") else bfmt(float(b)))
        rows.append([tag, cfg["rule"], var or "(best single)", bd, afmt(cfg["alpha"]), "+".join(where[tag]), pct(dev), pct(v["test_mean_norm"]), pct(v["test_mean_acc"]),
                     pct(v["test_min_norm"]), pp(v["diff_vs_ref"]), pp(lo), pp(hi), sig, pp(dev - v["test_mean_norm"])])
    add_table("A4a", f"Frozen ViT-B/32 test: all {len(res)} rows of test_comparison.json, sorted by difference to the reference `{ref}`",
              f"Every dev-frozen configuration ('tuned' = best (b, alpha) over the shared grid, 'alpha1' = best b at alpha = 1, plus the best single expert) evaluated once on the TEST split (80 % of each test set) of the 8 ViT-B/32 experts; "
              f"metrics = mean_norm / mean_acc / min_norm in % and the difference in mean_norm to the reference `{ref}` in percentage points with a 95 % paired task-stratified bootstrap CI (2000 resamples of test images).",
              [RES / "freeze" / "test_comparison.json", RES / "freeze" / "test.jsonl", RES / "freeze" / "frozen_configs_w1.json", RES / "freeze" / "frozen_configs_w2.json", RES / "freeze" / "frozen_configs_w3.json"],
              ["tag (rule|variant)", "rule", "variant", "b", "alpha", "frozen in", "dev mean_norm (%)", "test mean_norm (%)", "test mean_acc (%)", "test min_norm (%)",
               "diff vs ref (pp)", "CI95 low (pp)", "CI95 high (pp)", "CI excludes 0", "dev - test mean_norm (pp)"], rows, align="lllrrlrrrrrrrcr",
              notes=[f"Reference = `{ref}` (uniform lambda, alpha = 1.6 chosen on dev), whose own diff is 0 with a degenerate CI. Sorting is by diff, largest first. 'dev - test' mixes selection optimism with the (random) difference between the two image subsets; it is shown so that the optimism of 'tuned' vs 'alpha1' rows can be compared. "
                     "Pipes in tags are escaped as \\| for Markdown.",
                     f"test.jsonl holds {len(recs)} records for these {len(res)} tags: {', '.join(dups0)} were evaluated twice (frozen in both w2 and w3) with identical results (checked in A4d); compare_frozen.py keeps one per tag."])
    # per-task accuracy
    ea = expert_acc("b32")
    exp_test, zs_test = ea["expert"]["test"], ea["zeroshot"]["test"]
    sel = ["uniform|tuned", "kappa|tuned", "logkappa_q90_rev|tuned", "global:logkappa|tuned", "best_single", "uniform|alpha1", "kappa|alpha1"]
    rows = pertask_rows(by_tag, sel, exp_test, zs_test, norm=False)
    rows.append(["(own-task expert accuracy)"] + [pct(exp_test[t]) for t in TASKS8] + [pct(mean_over(exp_test)), "100.00", "100.00"])
    rows.append(["(zero-shot base)"] + [pct(zs_test[t]) for t in TASKS8] + [pct(mean_over(zs_test)), pct(np.mean([zs_test[t] / exp_test[t] for t in TASKS8])),
                                                                           pct(min(zs_test[t] / exp_test[t] for t in TASKS8))])
    add_table("A4b", "Frozen ViT-B/32 test: per-task accuracy of the key configurations",
              "Absolute test accuracy (%) per task for the reference, the literal rule, the two B/32 candidates, the best single expert and (extra) the alpha = 1 versions of uniform and kappa, on the TEST split of the 8 ViT-B/32 experts; last three columns = mean_acc, mean_norm, min_norm (%).",
              [RES / "freeze" / "test.jsonl", expert_acc_path("b32")], ["tag"] + TASKS8 + ["mean_acc", "mean_norm", "min_norm"], rows,
              notes=["mean_acc, mean_norm and min_norm are recomputed from the per-task accuracies and agree with the stored metrics (asserted in the script)."])
    rows = pertask_rows(by_tag, sel, exp_test, zs_test, norm=True)
    rows.append(["(zero-shot base)"] + [pct(zs_test[t] / exp_test[t]) for t in TASKS8] + [pct(np.mean([zs_test[t] / exp_test[t] for t in TASKS8])),
                                                                                      pct(np.mean([zs_test[t] / exp_test[t] for t in TASKS8])), pct(min(zs_test[t] / exp_test[t] for t in TASKS8))])
    add_table("A4c", "Frozen ViT-B/32 test: per-task NORMALISED accuracy of the key configurations",
              "Test accuracy divided by the task's own-expert test accuracy (%) per task for the same configurations as A4b, on the TEST split; the 'mean' column is the mean over tasks of these (= mean_norm).",
              [RES / "freeze" / "test.jsonl", expert_acc_path("b32")], ["tag"] + TASKS8 + ["mean over tasks", "mean_norm", "min_norm"], rows)
    # ----- integrity checks
    integ = []
    sw = sweeps()
    bad = []
    n_chk = 0
    for w, lst in frozen.items():
        for c in lst:
            n_chk += 1
            if c["tag"] == "best_single":
                continue
            rule = c["rule"]
            d = sw["by"][rule]
            if c["tag"].endswith("|tuned"):
                bm, _ = best_of(d)
            elif c["tag"].endswith("|alpha1"):
                bm, _ = best_of(d, alpha=1.0)
            else:
                bad.append(f"{w}:{c['tag']} (unknown variant)")
                continue
            ok = (bm["cfg"]["b"] == float(c["b"]) and bm["cfg"]["alpha"] == float(c["alpha"]) and math.isclose(bm["metrics"]["mean_norm"], c["dev_metric"], abs_tol=1e-12))
            if not ok:
                bad.append(f"{w}:{c['tag']} frozen (b={c['b']}, a={c['alpha']}, dev={c['dev_metric']:.6f}) vs sweep best (b={bm['cfg']['b']}, a={bm['cfg']['alpha']}, dev={bm['metrics']['mean_norm']:.6f})")
    integ.append(["frozen (b, alpha, dev_metric) = dev-best of the final sweeps", f"{n_chk - sum(1 for lst in frozen.values() for c in lst if c['tag'] == 'best_single') - len(bad)}/{n_chk - sum(1 for lst in frozen.values() for c in lst if c['tag'] == 'best_single')} match",
                  "; ".join(bad) if bad else "all frozen entries of w1/w2/w3 equal the argmax recomputed from results/sweeps (first maximum in file order)"])
    # test_comparison.json vs test.jsonl metrics
    mm = [t for t, v in res.items() if not (math.isclose(v["test_mean_norm"], by_tag[t]["metrics"]["mean_norm"], abs_tol=1e-12) and math.isclose(v["test_mean_acc"], by_tag[t]["metrics"]["mean_acc"], abs_tol=1e-12)
                                         and math.isclose(v["test_min_norm"], by_tag[t]["metrics"]["min_norm"], abs_tol=1e-12) and math.isclose(v["dev_metric"], by_tag[t].get("dev_metric") or 0, abs_tol=1e-12))]
    integ.append(["test_comparison.json metrics (test mean_norm / mean_acc / min_norm, dev_metric) = test.jsonl records", f"{len(res) - len(mm)}/{len(res)} tags", "; ".join(mm) if mm else "all"])
    # best single from the cross matrix
    cross = ea["cross"]["dev"]
    single = {e: float(np.mean([cross[e][t] / ea["expert"]["dev"][t] for t in TASKS8])) for e in TASKS8}
    e_best = max(single, key=single.get)
    SW["best_single_expert"] = e_best
    bs = [c for c in frozen["w1"] if c["tag"] == "best_single"][0]
    integ.append(["frozen best_single = best row of the dev cross matrix", "match" if (bs["rule"] == f"single:{e_best}" and math.isclose(bs["dev_metric"], single[e_best], abs_tol=1e-12)) else "MISMATCH",
                  f"{bs['rule']} dev {pct(bs['dev_metric'])} vs recomputed {e_best} {pct(single[e_best])}"])
    # test cfg vs frozen cfg
    mism = []
    allfrozen = {}
    for w, lst in frozen.items():
        for c in lst:
            allfrozen.setdefault(c["tag"], c)
    for tag, r in by_tag.items():
        c = allfrozen[tag]
        cf = r["cfg"]
        if not (cf["rule"] == c["rule"] and float(cf["b"]) == float(c["b"]) and float(cf["alpha"]) == float(c["alpha"]) and bool(cf.get("abs_beta", False)) == bool(c.get("abs_beta", False))
                and cf["split"] == "test" and len(cf["experts"]) == 8 and math.isclose(r.get("dev_metric") or 0, c["dev_metric"], abs_tol=1e-12)):
            mism.append(tag)
    integ.append(["evaluated test config = frozen config (rule, b, alpha, abs_beta, split=test, 8 experts, dev_metric)", f"{len(by_tag) - len(mism)}/{len(by_tag)} match", "; ".join(mism) if mism else "all tags"])
    # duplicates
    cnt = collections.Counter(r["cfg"]["tag"] for r in recs)
    dups = {t: n for t, n in cnt.items() if n > 1}
    same = all(len({json.dumps(x["metrics"], sort_keys=True) for x in recs if x["cfg"]["tag"] == t}) == 1 for t in dups)
    integ.append(["duplicate tags in test.jsonl", f"{len(recs)} records / {len(cnt)} tags", f"{len(dups)} tags evaluated twice ({', '.join(sorted(dups))}); metrics identical: {same}"])
    # freeze-before-test timeline
    times = {}
    for w, fn in (("w1", "frozen_configs_w1.json"), ("w2", "frozen_configs_w2.json"), ("w3", "frozen_configs_w3.json")):
        times[w] = git_first_commit_time(f"results/freeze/{fn}")
    if all(times.values()):
        viol = []
        margins = []
        for tag in by_tag:
            ws = where[tag]
            t_freeze = min(parse_local(times[w].replace("-04:00", "")) if False else _dt.datetime.fromisoformat(times[w]) for w in ws)
            t_test = min(parse_local(r["time"]) for r in recs if r["cfg"]["tag"] == tag)
            margins.append((t_test - t_freeze).total_seconds())
            if t_test < t_freeze:
                viol.append(tag)
        integ.append(["every tag's first test evaluation is later than the commit that introduced its frozen config", f"{len(by_tag) - len(viol)}/{len(by_tag)} OK",
                      f"commit times: w1 {times['w1']}, w2 {times['w2']}, w3 {times['w3']}; smallest margin {min(margins):.0f} s; "
                      + ("violations: " + ", ".join(viol) if viol else "no violations") + " (record times assumed UTC-04:00, like the commits)"])
    else:
        integ.append(["freeze-before-test timeline", "not checked", "git log unavailable"])
    # bootstrap reproduction
    repro = reproduce_bootstrap(recs, ref, exp_test)
    if repro is None:
        integ.append(["bootstrap CIs reproduced from per-example correctness", "not checked", "artifacts/runs/freeze_correct not readable"])
    else:
        md = max(abs(repro[t]["diff"] - res[t]["diff_vs_ref"]) for t in res if t in repro)
        mc = max(max(abs(repro[t]["ci"][0] - res[t]["ci95"][0]), abs(repro[t]["ci"][1] - res[t]["ci95"][1])) for t in res if t in repro)
        integ.append(["bootstrap diff and 95 % CI reproduced from the per-example correctness arrays (rng seed 0, 2000 resamples)", f"{len([t for t in res if t in repro])}/{len(res)} tags",
                      f"max |diff - stored| = {md:.2e}, max |CI endpoint - stored| = {mc:.2e}"])
        check("derived", "test_comparison.json diff/CI reproducible from freeze_correct arrays (max abs deviation, fraction)", "< 1e-9", f"{max(md, mc):.1e}", ok=max(md, mc) < 1e-9)
    add_table("A4d", "Integrity checks of the frozen ViT-B/32 test", "Consistency of the frozen configurations, the evaluated test records, the commit order and the stored bootstrap intervals; no new accuracy numbers.",
              [RES / "freeze" / "frozen_configs_w1.json", RES / "freeze" / "frozen_configs_w2.json", RES / "freeze" / "frozen_configs_w3.json", RES / "freeze" / "test.jsonl", RES / "freeze" / "test_comparison.json",
               RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)", "artifacts/runs/freeze_correct/<key>.npz", "git log (commit dates only)"],
              ["check", "result", "detail"], integ, align="lll")
    # ---- doc cross-checks
    R1 = "REPORT_01 §5 / SUMMARY §1 / FULL §7"
    d_ = lambda tag: res[tag]  # noqa: E731
    check(R1, "reference uniform|tuned test mean_norm (%)", "76.68", pct(d_("uniform|tuned")["test_mean_norm"]))
    check(R1, "kappa|tuned test mean_norm (%)", "76.61", pct(d_("kappa|tuned")["test_mean_norm"]))
    k = d_("kappa|tuned")
    check(R1, "kappa|tuned diff [CI] (pp)", "-0.08 [-0.20, +0.04]", f"{pp(k['diff_vs_ref'])} [{pp(k['ci95'][0])}, {pp(k['ci95'][1])}]")
    sh = [d_(f"shuf{s}:kappa|tuned") for s in range(5)]
    check(R1, "5 shuffled kappa|tuned: test mean_norm range (%)", "76.55-76.84", f"{pct(min(x['test_mean_norm'] for x in sh))}-{pct(max(x['test_mean_norm'] for x in sh))}")
    check(R1, "5 shuffled kappa|tuned: diff range (pp)", "-0.13 ... +0.15", f"{pp(min(x['diff_vs_ref'] for x in sh))} ... {pp(max(x['diff_vs_ref'] for x in sh))}")
    for tag, c1, c2 in [("kappa_rev|tuned", "76.46", "-0.23"), ("logkappa|tuned", "76.66", "-0.02"), ("kappa_strict|tuned", "76.38", "-0.30"), ("kappa_z|tuned", "76.00", "-0.69"),
                        ("global:kappa_z|tuned", "75.75", "-0.93"), ("metagpt|tuned", "75.71", "-0.97"), ("dnorm|tuned", "", "-0.21"), ("global:logkappa_rev|tuned", "", "-1.35")]:
        if c1:
            check(R1, f"{tag}: test mean_norm (%)", c1, pct(d_(tag)["test_mean_norm"]))
        check(R1, f"{tag}: diff vs reference (pp)", c2, pp(d_(tag)["diff_vs_ref"]))
    check(R1, "logkappa_q90_rev|tuned diff [CI] (pp)", "+0.47 [+0.24, +0.70]", f"{pp(d_('logkappa_q90_rev|tuned')['diff_vs_ref'])} [{pp(d_('logkappa_q90_rev|tuned')['ci95'][0])}, {pp(d_('logkappa_q90_rev|tuned')['ci95'][1])}]")
    check(R1, "logkappa_q90_rev|tuned test mean_norm (%)", "77.15", pct(d_("logkappa_q90_rev|tuned")["test_mean_norm"]))
    check(R1, "global:logkappa|tuned diff [CI] (pp)", "+0.27 [+0.03, +0.50]", f"{pp(d_('global:logkappa|tuned')['diff_vs_ref'])} [{pp(d_('global:logkappa|tuned')['ci95'][0])}, {pp(d_('global:logkappa|tuned')['ci95'][1])}]")
    check(R1, "global:logkappa|tuned test mean_norm (%)", "76.95", pct(d_("global:logkappa|tuned")["test_mean_norm"]))
    check(R1, "kappa_rev|tuned diff [CI] (pp) (FULL §7)", "-0.23 [-0.47, +0.02]", f"{pp(d_('kappa_rev|tuned')['diff_vs_ref'])} [{pp(d_('kappa_rev|tuned')['ci95'][0])}, {pp(d_('kappa_rev|tuned')['ci95'][1])}]")
    check(R1, "metagpt|tuned diff [CI] (pp) (FULL §7)", "-0.97 [-1.26, -0.68]", f"{pp(d_('metagpt|tuned')['diff_vs_ref'])} [{pp(d_('metagpt|tuned')['ci95'][0])}, {pp(d_('metagpt|tuned')['ci95'][1])}]")
    check(R1, "best_single test mean_norm (%) (FULL §7)", "53.94", pct(d_("best_single")["test_mean_norm"]))
    check(R1, "best_single diff [CI] (pp) (FULL §7)", "-22.74 [-23.28, -22.16]", f"{pp(d_('best_single')['diff_vs_ref'])} [{pp(d_('best_single')['ci95'][0])}, {pp(d_('best_single')['ci95'][1])}]")
    ka, ua = by_tag["kappa|alpha1"]["metrics"]["mean_norm"], by_tag["uniform|alpha1"]["metrics"]["mean_norm"]
    check(R1, "kappa at alpha = 1 vs uniform at alpha = 1: test mean_norm (%)", "73.29 vs 73.39", f"{pct(ka)} vs {pct(ua)}")
    check(R1, "kappa at alpha = 1 minus uniform at alpha = 1 (pp)", "-0.10", pp(ka - ua),
          note=f"exact difference {100 * (ka - ua):+.4f}; the quoted -0.10 is the difference of the rounded values {pct(ka)} - {pct(ua)}")
    gsn = [d_(f"gshuf{s}:global:kappa_z|tuned")["test_mean_norm"] for s in range(3)]
    gs = sorted(d_(f"gshuf{s}:global:kappa_z|tuned")["diff_vs_ref"] for s in range(3))
    rn = sorted(round(100 * x, 2) - round(100 * d_("uniform|tuned")["test_mean_norm"], 2) for x in gsn)
    check("RESEARCH_LOG wave 1 / FULL §7", "'The (gshuf) global shuffled nulls give -0.76 / -0.68 / +0.01' (pp, ascending)", "-0.76 / -0.68 / +0.01", " / ".join(pp(x) for x in gs),
          note="the nulls are gshuf{0,1,2}:global:kappa_z (shuffles of global:kappa_z, not of the candidate global:logkappa); in seed order "
               + " / ".join(pp(d_(f"gshuf{s}:global:kappa_z|tuned")["diff_vs_ref"]) for s in range(3))
               + "; the quoted values equal differences of the ROUNDED test_mean_norm values: " + " / ".join(f"{x:+.2f}" for x in rn))
    thr = max(x["diff_vs_ref"] for x in sh)
    above = sorted([t for t in res if t.endswith("|tuned") and res[t]["diff_vs_ref"] > thr and not t.startswith(("shuf", "gshuf"))], key=lambda t: -res[t]["diff_vs_ref"])
    check("REPORT_01 §5", "Only logkappa_q90_rev and global:logkappa are above the B/32 shuffle range (point estimates above the largest shuffle of kappa|tuned)",
          "logkappa_q90_rev, global:logkappa", ", ".join(a.replace("|tuned", "") for a in above), ok=[a.replace("|tuned", "") for a in above] == ["logkappa_q90_rev", "global:logkappa"],
          note="third rule: wc:kappa_rev|tuned " + pp(res["wc:kappa_rev|tuned"]["diff_vs_ref"]) + " [" + pp(res["wc:kappa_rev|tuned"]["ci95"][0]) + ", " + pp(res["wc:kappa_rev|tuned"]["ci95"][1]) + "] (evaluated in wave 3, not carried to B/16)")
    rules_t = {r["cfg"]["rule"] for r in recs}
    n_sel = len(rules_t - {"uniform", "single:cars"})
    check("REPORT_01 §5 / FULL §7", "'Test results for all 49 dev-selected rules' (distinct rules in test.jsonl excluding uniform and best_single)", "49", str(n_sel))
    check("FULL §7", "test_comparison.json has 101 entries", "101", str(len(res)))
    only_w3 = sorted(t for t in where if where[t] == ["w3"])
    commits = {}
    for w in ("w1", "w2", "w3"):
        try:
            commits[w] = subprocess.run(["git", "--no-optional-locks", "log", "--diff-filter=A", "--format=%h", "--", f"results/freeze/frozen_configs_{w}.json"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()[-1]
        except Exception:
            commits[w] = "?"
    SW["freeze_commits"] = commits
    check("REPORT_01 §5", "configurations frozen in commits 'edbdcf5, ee218d2' before any test evaluation", "edbdcf5, ee218d2", ", ".join(commits.values()),
          ok=(list(commits.values()) == ["edbdcf5", "ee218d2"]), note=f"{len(only_w3)} tags exist only in the third freeze file ({', '.join(sorted({t.split('|')[0] for t in only_w3}))}); FULL_REPORT §7 cites all three commits")
    check("FULL §19", "freeze_eval bug (metagpt / best-single silently uniform) fixed before the reported test numbers: metagpt|tuned and best_single differ from uniform|tuned", "differ",
          f"metagpt {pct(res['metagpt|tuned']['test_mean_norm'])} and best_single {pct(res['best_single']['test_mean_norm'])} vs uniform {pct(res['uniform|tuned']['test_mean_norm'])}",
          ok=(res["metagpt|tuned"]["test_mean_norm"] != res["uniform|tuned"]["test_mean_norm"] and res["best_single"]["test_mean_norm"] != res["uniform|tuned"]["test_mean_norm"]))
    mk = [res["mkgrad|tuned"]["diff_vs_ref"], res["mkgrad_rev|tuned"]["diff_vs_ref"]]
    check("FULL §9 / RESEARCH_LOG wave 2/3", "merged-kappa gradient `mkgrad`, both signs: 'about 1 pt below uniform' (test, tuned)", "about -1", f"{pp(mk[0])} / {pp(mk[1])}", ok=all(-1.6 <= 100 * x <= -0.8 for x in mk), approx=True)
    pos = sum(1 for t in res if t.endswith("|tuned") and res[t]["diff_vs_ref"] > 0)
    tot = sum(1 for t in res if t.endswith("|tuned"))
    check("REPORT_01 §5", "'most are below uniform' (tuned variants with a positive point estimate)", "minority", f"{pos} of {tot}", ok=pos < tot / 2)
    SW["A4"] = {"by_tag": by_tag, "res": res}


def build_A5() -> None:
    comp = read_json(RES / "b16" / "freeze" / "transfer_comparison.json")
    cfgs = read_json(CFG / "b16_transfer_configs.json")
    meta = read_json(CFG / "b16_transfer_meta.json")
    recs = freeze_b16_records()
    p0, p1 = read_jsonl(RES / "b16" / "freeze" / "transfer_test_part0.jsonl"), read_jsonl(RES / "b16" / "freeze" / "transfer_test_part1.jsonl")
    ref = comp["ref"]
    res = comp["results"]
    by_tag = {r["cfg"]["tag"]: r for r in recs}
    cfg_by_tag = {c["tag"]: c for c in cfgs}
    rows = []
    for tag, v in sorted(res.items(), key=lambda kv: (-kv[1]["diff_vs_ref"], kv[0])):
        c = cfg_by_tag[tag]
        lo, hi = v["ci95"]
        rows.append([tag, c["rule"], "-" if c["rule"] in ("uniform", "metagpt") else bfmt(float(c["b"])), afmt(c["alpha"]), c["source"],
                     "-" if v["dev_metric"] is None else pct(v["dev_metric"]), pct(v["test_mean_norm"]), pct(v["test_mean_acc"]), pct(v["test_min_norm"]),
                     pp(v["diff_vs_ref"]), pp(lo), pp(hi), "+" if lo > 0 else ("-" if hi < 0 else "")])
    add_table("A5a", f"Pre-registered ViT-B/16 transfer: all {len(res)} rows of transfer_comparison.json, sorted by difference to the reference `{ref}`",
              "The ViT-B/32 dev-frozen rules and (b, alpha) (no B/16 tuning), and matched-config shuffled-score nulls, evaluated once on the ViT-B/16 TEST split (80 % of each test set); metrics = mean_norm / mean_acc / min_norm in %, "
              "difference in mean_norm to the reference in percentage points with a 95 % paired task-stratified bootstrap CI.",
              [RES / "b16" / "freeze" / "transfer_comparison.json", RES / "b16" / "freeze" / "transfer_test.jsonl", CFG / "b16_transfer_configs.json", CFG / "b16_transfer_meta.json"],
              ["tag", "rule", "b", "alpha", "source (configs/b16_transfer_configs.json)", "ViT-B/32 dev mean_norm of the frozen config (%)", "B/16 test mean_norm (%)", "B/16 test mean_acc (%)",
               "B/16 test min_norm (%)", "diff vs ref (pp)", "CI95 low (pp)", "CI95 high (pp)", "CI excludes 0"], rows, align="lllrlrrrrrrrc",
              notes=[f"Reference = `{ref}` (uniform, alpha = 1.6 frozen on ViT-B/32 dev). Pre-registration record (configs/b16_transfer_meta.json): \"{meta['preregistered']}\"; protocol: \"{meta['protocol']}\". "
                     "Null rows (dev metric '-') are the shuffled-score versions of a candidate evaluated at exactly the candidate's frozen (b, alpha)."])
    # candidate vs nulls
    rows = []
    cand_defs = [("kappa|tuned", [f"b16xfer|shuf{s}:kappa@kappa|tuned" for s in range(5)]),
                 ("logkappa_q90_rev|tuned", [f"b16xfer|shuf{s}:logkappa_q90_rev@logkappa_q90_rev|tuned" for s in range(5)]),
                 ("global:logkappa|tuned", [f"b16xfer|gshuf{s}:global:logkappa@global:logkappa|tuned" for s in range(5)])]
    NUL = {}
    for cand, nulls in cand_defs:
        ct = "b16xfer|" + cand
        cd = res[ct]["diff_vs_ref"]
        nd = np.array([res[n]["diff_vs_ref"] for n in nulls])
        NUL[cand] = (cd, nd)
        rows.append([cand, pp(cd), f"[{pp(res[ct]['ci95'][0])}, {pp(res[ct]['ci95'][1])}]", ", ".join(pp(x) for x in nd), pp(nd.min()), pp(nd.max()), pp(nd.mean()),
                     f"{int((nd >= cd).sum())}/{len(nd)}", "yes" if nd.min() <= cd <= nd.max() else "no (outside)"])
    add_table("A5b", "ViT-B/16 transfer: each candidate against its matched shuffled-score nulls",
              "For the literal rule and the two B/32 candidates, the B/16 test difference (pp, mean_norm vs uniform|tuned) next to the differences of the 5 shuffled-score nulls evaluated at the same frozen (b, alpha); ViT-B/16 TEST split.",
              [RES / "b16" / "freeze" / "transfer_comparison.json"],
              ["candidate (b16xfer|...)", "diff (pp)", "CI95 (pp)", "null diffs in seed order (pp)", "null min", "null max", "null mean", "# nulls >= candidate", "candidate inside null range"], rows, align="lrlrrrrrl",
              notes=["Null for `kappa|tuned`: shuf0..4:kappa; for `logkappa_q90_rev|tuned`: shuf0..4:logkappa_q90_rev; for `global:logkappa|tuned`: gshuf0..4:global:logkappa (one expert permutation for all maps).",
                     f"`global:logkappa|tuned` is above all {len(NUL['global:logkappa|tuned'][1])} of its matched nulls on B/16 ({pp(NUL['global:logkappa|tuned'][0])} vs null max {pp(NUL['global:logkappa|tuned'][1].max())}) but is not distinguishable from uniform (CI above); "
                     f"`logkappa_q90_rev|tuned` is below all {len(NUL['logkappa_q90_rev|tuned'][1])} of its nulls."])
    # per-task B/16
    ea = expert_acc("b16")
    exp_t, zs_t = ea["expert"]["test"], ea["zeroshot"]["test"]
    sel = ["b16xfer|uniform|tuned", "b16xfer|kappa|tuned", "b16xfer|logkappa_q90_rev|tuned", "b16xfer|global:logkappa|tuned", "b16xfer|kappa_rev|tuned", "b16xfer|metagpt|tuned",
           "b16xfer|uniform|alpha1", "b16xfer|kappa|alpha1", "b16xfer|logkappa_q90_rev|alpha1"]
    rows = pertask_rows(by_tag, sel, exp_t, zs_t, norm=False)
    rows.append(["(own-task expert accuracy)"] + [pct(exp_t[t]) for t in TASKS8] + [pct(mean_over(exp_t)), "100.00", "100.00"])
    rows.append(["(zero-shot base)"] + [pct(zs_t[t]) for t in TASKS8] + [pct(mean_over(zs_t)), pct(np.mean([zs_t[t] / exp_t[t] for t in TASKS8])), pct(min(zs_t[t] / exp_t[t] for t in TASKS8))])
    add_table("A5c", "ViT-B/16 transfer: per-task test accuracy of the key configurations",
              "Absolute test accuracy (%) per task of the reference, the literal rule, the candidates and a few comparison rules on the ViT-B/16 TEST split; last three columns = mean_acc, mean_norm, min_norm (%).",
              [RES / "b16" / "freeze" / "transfer_test.jsonl", expert_acc_path("b16")], ["tag"] + TASKS8 + ["mean_acc", "mean_norm", "min_norm"], rows)
    # integrity
    integ = []
    keys_all = {r["key"] for r in recs}
    keys_parts = {r["key"] for r in p0} | {r["key"] for r in p1}
    integ.append(["transfer_test.jsonl = union of part0 and part1", "match" if keys_all == keys_parts and len(recs) == len(p0) + len(p1) else "MISMATCH", f"{len(recs)} records = {len(p0)} + {len(p1)}"])
    cp0, cp1 = read_json(CFG / "b16_transfer_part0.json"), read_json(CFG / "b16_transfer_part1.json")
    ok_parts = sorted(json.dumps(c, sort_keys=True) for c in cp0 + cp1) == sorted(json.dumps(c, sort_keys=True) for c in cfgs)
    integ.append(["configs/b16_transfer_part0.json + part1.json = b16_transfer_configs.json (as multisets)", "match" if ok_parts else "MISMATCH", f"{len(cp0)} + {len(cp1)} = {len(cfgs)} configs"])
    mm = [t for t, v in res.items() if not (math.isclose(v["test_mean_norm"], by_tag[t]["metrics"]["mean_norm"], abs_tol=1e-12) and math.isclose(v["test_mean_acc"], by_tag[t]["metrics"]["mean_acc"], abs_tol=1e-12)
                                         and math.isclose(v["test_min_norm"], by_tag[t]["metrics"]["min_norm"], abs_tol=1e-12))]
    integ.append(["transfer_comparison.json metrics = transfer_test.jsonl records", f"{len(res) - len(mm)}/{len(res)} tags", "; ".join(mm) if mm else "all"])
    miss = [c["tag"] for c in cfgs if c["tag"] not in by_tag]
    integ.append(["every config of b16_transfer_configs.json was evaluated", f"{len(cfgs) - len(miss)}/{len(cfgs)}", ", ".join(miss) if miss else "all"])
    mism = []
    for tag, r in by_tag.items():
        c, cf = cfg_by_tag[tag], r["cfg"]
        if not (cf["rule"] == c["rule"] and float(cf["b"]) == float(c["b"]) and float(cf["alpha"]) == float(c["alpha"]) and cf["split"] == "test"):
            mism.append(tag)
    integ.append(["evaluated config = pre-registered config (rule, b, alpha, split)", f"{len(by_tag) - len(mism)}/{len(by_tag)}", ", ".join(mism) if mism else "all"])
    # frozen B/32 source: dev_metric equals B/32 frozen dev metric
    frozen32 = {}
    for w in ("w1", "w2", "w3"):
        for c in read_json(RES / "freeze" / f"frozen_configs_{w}.json"):
            frozen32.setdefault((c["rule"], float(c["b"]), float(c["alpha"])), c["dev_metric"])
    bad = []
    for c in cfgs:
        if c["dev_metric"] is not None:
            k = (c["rule"], float(c["b"]), float(c["alpha"]))
            if k not in frozen32 or not math.isclose(frozen32[k], c["dev_metric"], abs_tol=1e-12):
                bad.append(c["tag"])
    integ.append(["B/16 configs with a dev metric = the B/32 frozen config (rule, b, alpha, dev_metric)", f"{sum(1 for c in cfgs if c['dev_metric'] is not None) - len(bad)}/{sum(1 for c in cfgs if c['dev_metric'] is not None)}", ", ".join(bad) if bad else "all"])
    t_cfg = git_first_commit_time("configs/b16_transfer_configs.json")
    if t_cfg:
        t_first = min(parse_local(r["time"]) for r in recs)
        t_c = _dt.datetime.fromisoformat(t_cfg)
        integ.append(["pre-registration commit precedes the first B/16 evaluation", "OK" if t_first > t_c else "VIOLATION", f"configs committed {t_cfg}; first B/16 test record {min(r['time'] for r in recs)} (assumed UTC-04:00)"])
    repro = reproduce_bootstrap(recs, ref, exp_t)
    if repro is not None:
        md = max(abs(repro[t]["diff"] - res[t]["diff_vs_ref"]) for t in res if t in repro)
        mc = max(max(abs(repro[t]["ci"][0] - res[t]["ci95"][0]), abs(repro[t]["ci"][1] - res[t]["ci95"][1])) for t in res if t in repro)
        integ.append(["bootstrap diff and 95 % CI reproduced from the per-example correctness arrays (rng seed 0, 2000 resamples)", f"{len([t for t in res if t in repro])}/{len(res)} tags",
                      f"max |diff - stored| = {md:.2e}, max |CI endpoint - stored| = {mc:.2e}"])
        check("derived", "transfer_comparison.json diff/CI reproducible from freeze_correct arrays (max abs deviation)", "< 1e-9", f"{max(md, mc):.1e}", ok=max(md, mc) < 1e-9)
    add_table("A5d", "Integrity checks of the ViT-B/16 transfer", "Consistency of the pre-registered configurations, the evaluated records, the commit order and the stored bootstrap intervals; no new accuracy numbers.",
              [RES / "b16" / "freeze" / "transfer_test.jsonl", RES / "b16" / "freeze" / "transfer_test_part0.jsonl", RES / "b16" / "freeze" / "transfer_test_part1.jsonl", CFG / "b16_transfer_configs.json",
               RES / "b16" / "freeze" / "transfer_comparison.json", CFG / "b16_transfer_part0.json", CFG / "b16_transfer_part1.json", "artifacts/runs/freeze_correct/<key>.npz", "git log (commit dates only)"], ["check", "result", "detail"], integ, align="lll")
    # ---- doc cross-checks
    R1 = "REPORT_01 §5 / SUMMARY §1 / FULL §10"
    ref_v = res[ref]["test_mean_norm"]
    check(R1, "B/16 uniform|tuned test mean_norm (%)", "82.01", pct(ref_v))
    for cand, c1, c2 in [("kappa|tuned", "81.90", "-0.11"), ("logkappa_q90_rev|tuned", "81.00", "-1.01"), ("global:logkappa|tuned", "82.03", "+0.02"), ("kappa_rev|tuned", "79.87", "-2.15"),
                         ("metagpt|tuned", "", "-0.83")]:
        t = "b16xfer|" + cand
        if c1:
            check(R1, f"B/16 {cand}: test mean_norm (%)", c1, pct(res[t]["test_mean_norm"]))
        check(R1, f"B/16 {cand}: diff vs uniform|tuned (pp)", c2, pp(res[t]["diff_vs_ref"]))
    nk, nq, ng = NUL["kappa|tuned"][1], NUL["logkappa_q90_rev|tuned"][1], NUL["global:logkappa|tuned"][1]
    check(R1, "B/16 shuffles of kappa: diff range (pp)", "-0.76 ... +0.24", f"{pp(nk.min())} ... {pp(nk.max())}")
    check(R1, "B/16 shuffles of logkappa_q90_rev: diff range (pp)", "-0.95 ... +0.32", f"{pp(nq.min())} ... {pp(nq.max())}")
    check(R1, "B/16 gshuf of global:logkappa: diff range (pp)", "-1.04 ... -0.08", f"{pp(ng.min())} ... {pp(ng.max())}")
    check("FULL §10", "B/16 at alpha = 1: uniform / kappa / q90_rev test mean_norm (%)", "78.16 / 78.01 / 78.05",
          " / ".join(pct(res[f"b16xfer|{x}|alpha1"]["test_mean_norm"]) for x in ("uniform", "kappa", "logkappa_q90_rev")))
    SW["A5"] = {"res": res}


# --------------------------------------------------------------------------------------------
# A6: pairs
# --------------------------------------------------------------------------------------------
def refine(lams: np.ndarray, f: np.ndarray) -> float:
    """3-point parabola refinement of the grid argmax (copied logic of experiments/analyze_pairs.py)."""
    k = int(np.argmax(f))
    if 0 < k < len(lams) - 1:
        x0, x1, x2 = lams[k - 1:k + 2]
        y0, y1, y2 = f[k - 1:k + 2]
        den = (y0 - 2 * y1 + y2)
        if den < 0:
            h = x1 - x0
            return float(np.clip(x1 + 0.5 * h * (y0 - y2) / den, x0, x2))
    return float(lams[k])


def build_A6() -> None:
    from scipy.stats import binomtest, spearmanr
    pa = read_json(RES / "pairs" / "pair_analysis.json")
    recs = read_jsonl(RES / "pairs" / "pair_grid_dev.jsonl")
    ea = expert_acc("b32")
    ref, zs, cross = ea["expert"]["dev"], ea["zeroshot"]["dev"], ea["cross"]["dev"]
    S = spectra("b32")
    full, upd, base = S["full"], S["upd"], S["base"]
    kap = full.pivot(index="param", columns="expert", values="kappa2_raw")[TASKS8].sort_index()
    dn = upd.pivot(index="param", columns="expert", values="fro")[TASKS8].loc[kap.index]
    w = (base.loc[kap.index, "m"] * base.loc[kap.index, "n"]).values.astype(float)
    logk = np.log(kap.values)
    dn_tot = np.sqrt((dn.values ** 2).sum(0))
    curves = collections.defaultdict(dict)
    for r in recs:
        c = r["cfg"]
        curves[(tuple(c["pair"]), c["alpha"])][c["lam"]] = r["metrics"]["mean_norm"]
    assert all(c["mode"] == "grid" and c["split"] == "dev" for c in (r["cfg"] for r in recs))
    out = []
    for (pair, alpha), d in sorted(curves.items()):
        lams = np.array(sorted(d))
        f = np.array([d[x] for x in lams])
        assert len(lams) == 11
        i, j = TASKS8.index(pair[0]), TASKS8.index(pair[1])
        ls = refine(lams, f)
        out.append({"pair": list(pair), "alpha": alpha, "i": i, "j": j, "lam_star": ls, "lam_grid": float(lams[np.argmax(f)]), "f_star": float(f.max()), "f_half": float(d[0.5]), "gain": float(f.max() - d[0.5]),
                    "f0": d[0.0], "f1": d[1.0], "curve": [d[x] for x in lams],
                    "mean_logkappa_ratio": float(np.sum(w * (logk[:, i] - logk[:, j])) / w.sum()), "frac_kappa_i_gt_j": float(np.sum(w * (logk[:, i] > logk[:, j])) / w.sum()),
                    "log_dnorm_ratio": float(np.log(dn_tot[i] / dn_tot[j])),
                    "transfer_asym": float(cross[pair[0]][pair[1]] / ref[pair[1]] - cross[pair[1]][pair[0]] / ref[pair[0]]),
                    "gain_zs_asym": float((ref[pair[0]] - zs[pair[0]]) - (ref[pair[1]] - zs[pair[1]]))})
    assert sorted({o["alpha"] for o in out}) == [1.0, 1.4] and len({tuple(o["pair"]) for o in out}) == 28 and all(len(o["curve"]) == 11 for o in out)  # design parameters typed in the captions
    # compare with the stored pair_analysis.json (pairs)
    stored = {(tuple(o["pair"]), o["alpha"]): o for o in pa["pairs"]}
    out.sort(key=lambda o: (o["i"], o["j"], o["alpha"]))
    assert len(stored) == len(out) == 56
    maxd = 0.0
    for o in out:
        so = stored[(tuple(o["pair"]), o["alpha"])]
        for k in ("lam_star", "lam_grid", "f_star", "f_half", "gain", "mean_logkappa_ratio", "frac_kappa_i_gt_j", "log_dnorm_ratio", "transfer_asym", "gain_zs_asym"):
            maxd = max(maxd, abs(o[k] - so[k]))
    check("derived", "pair_analysis.json 'pairs' (56 rows) reproduced from pair_grid_dev.jsonl + stats.csv + expert_accuracy.json (max abs deviation)", "< 1e-9", f"{maxd:.1e}", ok=maxd < 1e-9)
    # tests
    tests = {}
    for alpha in sorted({o["alpha"] for o in out}):
        oa = [o for o in out if o["alpha"] == alpha]
        y = np.array([o["lam_star"] - 0.5 for o in oa])
        A = np.zeros((len(oa) + 1, 8))
        for r_, o in enumerate(oa):
            A[r_, o["i"]], A[r_, o["j"]] = 1, -1
        A[-1] = 1
        u, *_ = np.linalg.lstsq(A, np.append(y, 0), rcond=None)
        resid = y - A[:-1] @ u
        t = {"n_pairs": len(oa), "mean_abs_dev_from_half": float(np.mean(np.abs(y))), "median_gain": float(np.median([o["gain"] for o in oa])), "worth_u": dict(zip(TASKS8, map(float, u))),
             "worth_model_R2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)), "yhat": A[:-1] @ u, "y": y}
        for pred in ("mean_logkappa_ratio", "frac_kappa_i_gt_j", "log_dnorm_ratio", "transfer_asym", "gain_zs_asym"):
            sp = spearmanr(np.array([o[pred] for o in oa]), y)
            t[f"spearman_{pred}"] = [float(sp.statistic), float(sp.pvalue)]
        inf_ = [o for o in oa if abs(o["lam_star"] - 0.5) >= 0.05]
        agree = sum((o["mean_logkappa_ratio"] > 0) == (o["lam_star"] < 0.5) for o in inf_)
        t["sign_test_kappa"] = {"n_informative": len(inf_), "agree": int(agree), "p_two_sided": float(binomtest(int(agree), len(inf_)).pvalue)}
        lk_e = (w[:, None] * logk).sum(0) / w.sum()
        t["expert_level_spearman_worth_vs_meanlogkappa"] = [float(v) for v in spearmanr(u, lk_e)]
        t["expert_level_spearman_worth_vs_dnorm"] = [float(v) for v in spearmanr(u, dn_tot)]
        t["expert_level_spearman_worth_vs_gain_over_zs"] = [float(v) for v in spearmanr(u, [ref[e] - zs[e] for e in TASKS8])]
        tests[alpha] = t
        for o, yh, yy in zip(oa, t["yhat"], y):
            o["yhat"], o["y"] = float(yh), float(yy)
    md = 0.0
    for a, t in tests.items():
        st_ = pa["tests"][str(a)]
        for k, v in t.items():
            if k in ("yhat", "y"):
                continue
            if k == "worth_u":
                md = max(md, max(abs(v[e] - st_[k][e]) for e in TASKS8))
            elif isinstance(v, dict):
                md = max(md, abs(v["p_two_sided"] - st_[k]["p_two_sided"]), abs(v["agree"] - st_[k]["agree"]), abs(v["n_informative"] - st_[k]["n_informative"]))
            elif isinstance(v, list):
                md = max(md, max(abs(x - y_) for x, y_ in zip(v, st_[k])))
            else:
                md = max(md, abs(v - st_[k]))
    check("derived", "pair_analysis.json 'tests' (R2, Spearman, sign test, worth u) reproduced from the same raw files (max abs deviation)", "< 1e-9", f"{md:.1e}", ok=md < 1e-9)
    # ---- tables
    rows_a, rows_b, rows_c = [], [], []
    for o in out:
        pr = o["pair"]
        inf_flag = abs(o["lam_star"] - 0.5) >= 0.05
        agree = ((o["mean_logkappa_ratio"] > 0) == (o["lam_star"] < 0.5)) if inf_flag else None
        rows_a.append([pr[0], pr[1], afmt(o["alpha"]), fnum(o["lam_star"], 3), f"{o['lam_grid']:.1f}", pct(o["f0"]), pct(o["f_half"]), pct(o["f_star"]), pct(o["f1"]), pp(o["gain"]),
                       "yes" if inf_flag else "no", "-" if agree is None else ("agrees" if agree else "disagrees")])
        rows_b.append([pr[0], pr[1], afmt(o["alpha"]), sfnum(o["mean_logkappa_ratio"], 4), fnum(o["frac_kappa_i_gt_j"], 3), sfnum(o["log_dnorm_ratio"], 3), sfnum(o["transfer_asym"], 3),
                       sfnum(o["gain_zs_asym"], 3), sfnum(o["y"], 3), sfnum(o["yhat"], 3), sfnum(o["y"] - o["yhat"], 3)])
        rows_c.append([pr[0], pr[1], afmt(o["alpha"])] + [pct(x) for x in o["curve"]])
    add_table("A6a", "Pairs: favourable weight lambda* and the shape of each pair's dev curve (28 pairs x 2 alphas)",
              "For every unordered pair of experts and alpha in {1.0, 1.4}, one global weight lambda on expert i (1 - lambda on expert j) for ALL parameters, W = W0 + alpha (lambda D_i + (1-lambda) D_j), scanned over lambda in {0, 0.1, ..., 1}; "
              "metric = dev mean_norm of the two tasks (%), lambda* = parabola-refined argmax, gain = max - value at lambda = 0.5 in percentage points; dev split.",
              [RES / "pairs" / "pair_grid_dev.jsonl", RES / "pairs" / "pair_analysis.json"],
              ["expert i (weight lambda)", "expert j (weight 1-lambda)", "alpha", "lambda* (refined)", "lambda at grid max", "mean_norm at lambda=0 (all j) (%)", "at lambda=0.5 (%)", "at grid max (%)",
               "at lambda=1 (all i) (%)", "gain = max - f(0.5) (pp)", "informative (|lambda*-0.5| >= 0.05)", "professor's direction (higher mean log-kappa gets lower lambda)"], rows_a, align="llrrrrrrrrll",
              notes=["Recomputed from pair_grid_dev.jsonl with the algorithm of experiments/analyze_pairs.py and compared with pair_analysis.json (all fields agree to < 1e-9; see X1). The favourable-lambda* column is the quantity analysed in A6d."])
    add_table("A6b", "Pairs: predictors of lambda* and the per-expert worth-model fit (28 pairs x 2 alphas)",
              "For every pair and alpha, the quantities compared with lambda* (all oriented as expert i minus expert j) and the worth-model fit lambda* - 0.5 ~ u_i - u_j (u = per-expert worth, least squares with sum u = 0); the predictors come from the experts' weights (kappa, update norms) and dev accuracies and are dimensionless, the lambda deviations are in lambda units; dev split.",
              [RES / "pairs" / "pair_analysis.json", RES / "pairs" / "pair_grid_dev.jsonl", RES / "spectra" / "stats.csv", expert_acc_path("b32")],
              ["expert i", "expert j", "alpha", "mean log-kappa ratio (param-weighted, i - j)", "fraction of maps (param-weighted) with kappa_i > kappa_j", "log(||D_i||_F / ||D_j||_F) over all maps",
               "transfer asymmetry = acc(i on j)/ref_j - acc(j on i)/ref_i", "gain-over-zero-shot asymmetry (acc units)", "observed lambda* - 0.5", "worth model u_i - u_j", "residual"], rows_b, align="llrrrrrrrrr",
              notes=["Predictors are those listed in the docstring of experiments/analyze_pairs.py; the fraction of maps is weighted by parameter count (m*n) and is independent of alpha."])
    add_table("A6c", "Pairs: full dev curves, mean_norm (%) for lambda = 0, 0.1, ..., 1 (28 pairs x 2 alphas)",
              "The 11 evaluated points of every pair curve: dev mean_norm (%) of the two-expert merge W0 + alpha (lambda D_i + (1-lambda) D_j) for lambda from 0 (all expert j) to 1 (all expert i); dev split of the two tasks.",
              [RES / "pairs" / "pair_grid_dev.jsonl"], ["expert i", "expert j", "alpha"] + [f"lambda={x / 10:.1f}" for x in range(11)], rows_c, align="llr" + "r" * 11,
              notes=["616 records = 28 pairs x 2 alphas x 11 lambdas; the grid is in pair_grid_dev.jsonl."])
    # A6d summary
    keys = [("n_pairs", "number of pairs", lambda t: str(t["n_pairs"])),
            ("mean_abs_dev_from_half", "mean |lambda* - 0.5|", lambda t: fnum(t["mean_abs_dev_from_half"], 4)),
            ("median_gain", "median gain f(max) - f(0.5) (pp)", lambda t: pp(t["median_gain"], 3)),
            ("worth_model_R2", "worth-model R^2 (lambda*-0.5 ~ u_i - u_j)", lambda t: fnum(t["worth_model_R2"], 4))]
    T = {a: pa["tests"][str(a)] for a in (1.0, 1.4)}  # display the stored values (equal to the recomputed ones, checked above)
    rows = [[lab, *(fn(T[a]) for a in (1.0, 1.4))] for _, lab, fn in keys]
    labs = {"mean_logkappa_ratio": "mean log-kappa ratio (i - j)", "frac_kappa_i_gt_j": "fraction of maps with kappa_i > kappa_j", "log_dnorm_ratio": "log update-norm ratio (i - j)",
            "transfer_asym": "transfer asymmetry", "gain_zs_asym": "gain-over-zero-shot asymmetry"}
    for k, lab in labs.items():
        rows.append([f"pair-level Spearman rho (p): lambda*-0.5 vs {lab}"] + [f"{T[a]['spearman_' + k][0]:+.3f} (p={T[a]['spearman_' + k][1]:.3g})" for a in (1.0, 1.4)])
    rows.append(["sign test of the professor's direction: agree / informative pairs (two-sided binomial p)"] +
                [f"{T[a]['sign_test_kappa']['agree']}/{T[a]['sign_test_kappa']['n_informative']} (p={T[a]['sign_test_kappa']['p_two_sided']:.3g})" for a in (1.0, 1.4)])
    for k, lab in [("expert_level_spearman_worth_vs_meanlogkappa", "mean log kappa of the expert (param-weighted)"), ("expert_level_spearman_worth_vs_dnorm", "total update norm"),
                   ("expert_level_spearman_worth_vs_gain_over_zs", "expert gain over zero-shot")]:
        rows.append([f"expert-level Spearman rho (p, n = 8): worth u vs {lab}"] + [f"{T[a][k][0]:+.3f} (p={T[a][k][1]:.3g})" for a in (1.0, 1.4)])
    add_table("A6d", "Pairs: summary statistics (pair_analysis.json 'tests')",
              "Summary of the 28-pair analysis at alpha = 1.0 and 1.4: size and gain of the favourable-lambda deviations, the worth-model fit, Spearman correlations of lambda* - 0.5 with each predictor across pairs, the sign test of the professor's direction and the expert-level correlations of worth; dev split, dimensionless.",
              [RES / "pairs" / "pair_analysis.json"], ["statistic", "alpha = 1.0", "alpha = 1.4"], rows, align="lrr",
              notes=["Values shown are the stored pair_analysis.json values; the script recomputes all of them from the raw files and they agree to < 1e-9 (X1). p-values are two-sided (scipy spearmanr / binomtest)."])
    rows = [[e, sfnum(T[1.0]["worth_u"][e], 4), sfnum(T[1.4]["worth_u"][e], 4)] for e in TASKS8]
    add_table("A6e", "Pairs: per-expert worth u (positive = deserves more weight)", "Least-squares expert worth u (sum zero) from lambda* - 0.5 ~ u_i - u_j over the 28 pairs at alpha = 1.0 and 1.4, in units of lambda (dimensionless); dev split.",
              [RES / "pairs" / "pair_analysis.json"], ["expert", "u at alpha = 1.0", "u at alpha = 1.4"], rows, align="lrr")
    # ---- doc cross-checks
    R1 = "REPORT_01 §3 / FULL §8"
    t1 = tests[1.0]
    check(R1, "worth-model R^2 (alpha = 1)", "0.91", f"{t1['worth_model_R2']:.2f}")
    ord_ = sorted(TASKS8, key=lambda e: -t1["worth_u"][e])
    check(R1, "'DTD and Cars deserve more weight, MNIST and EuroSAT less' (top-2 / bottom-2 worth at alpha = 1)", "dtd, cars | mnist, eurosat", f"{', '.join(ord_[:2])} | {', '.join(ord_[:-3:-1])}")
    check(R1, "worth vs kappa: expert-level Spearman rho (alpha = 1)", "-0.02", f"{t1['expert_level_spearman_worth_vs_meanlogkappa'][0]:.2f}")
    check(R1, "worth vs kappa: p (alpha = 1)", "0.96", f"{t1['expert_level_spearman_worth_vs_meanlogkappa'][1]:.2f}")
    check(R1, "sign test (alpha = 1)", "10/19", f"{t1['sign_test_kappa']['agree']}/{t1['sign_test_kappa']['n_informative']}")
    check(R1, "larger updates deserve less weight: pair-level rho (alpha = 1)", "-0.49", f"{t1['spearman_log_dnorm_ratio'][0]:.2f}")
    check(R1, "larger updates deserve less weight: p (alpha = 1)", "0.008", f"{t1['spearman_log_dnorm_ratio'][1]:.3f}")
    fr = [o["frac_kappa_i_gt_j"] for o in out if o["alpha"] == 1.0]
    outside_w = [(o["pair"], o["frac_kappa_i_gt_j"]) for o in out if o["alpha"] == 1.0 and not (0.4 - 0.005 <= o["frac_kappa_i_gt_j"] <= 0.6 + 0.005)]
    check(R1, "'fraction of maps with kappa_i > kappa_j is 0.4-0.6 for every pair' (param-weighted fraction, 28 pairs)", "0.4-0.6", f"{min(fr):.2f}-{max(fr):.2f}", ok=(len(outside_w) == 0),
          note=f"{len(outside_w)} of 28 pairs outside [0.4, 0.6]: " + "; ".join(f"{a}>{b} {f:.2f}" for (a, b), f in sorted(outside_w, key=lambda x: -x[1])))
    # unweighted fraction as extra info
    unw = []
    for i in range(8):
        for j in range(i + 1, 8):
            unw.append(float(np.mean(logk[:, i] > logk[:, j])))
    pairs_ = [(TASKS8[i], TASKS8[j]) for i in range(8) for j in range(i + 1, 8)]
    outside_u = [(pr, f) for pr, f in zip(pairs_, unw) if not (0.4 - 0.005 <= f <= 0.6 + 0.005)]
    check(R1, "same, unweighted (fraction of the 73 maps)", "0.4-0.6", f"{min(unw):.2f}-{max(unw):.2f}", ok=(len(outside_u) == 0),
          note=f"{len(outside_u)} of 28 pairs outside [0.4, 0.6]")
    SW["A6"] = {"tests": tests}


# --------------------------------------------------------------------------------------------
# A7: pre-registered subsets
# --------------------------------------------------------------------------------------------
def build_A7() -> None:
    subsets = read_json(CFG / "subsets.json")
    summ = read_json(RES / "sweeps" / "subsets_summary.json")
    RULES = ["uniform", "kappa", "kappa_rev", "logkappa", "shuf0:kappa", "shuf1:kappa", "shuf2:kappa"]
    data = {}
    rows_a, rows_b, rows_c = [], [], []
    for name, experts in subsets["subsets"].items():
        recs = read_jsonl(RES / "sweeps" / f"subsets_dev_{name}.jsonl")
        assert all(r["cfg"]["experts"] == experts and r["cfg"]["split"] == "dev" for r in recs), name
        by = collections.defaultdict(dict)
        for r in recs:
            by[r["cfg"]["rule"]][(r["cfg"]["b"], r["cfg"]["alpha"])] = r
        assert sorted(by) == sorted(RULES), (name, sorted(by))
        info = {}
        for rule in RULES:
            bm, ties = best_of(by[rule])
            b1, _ = best_of(by[rule], alpha=1.0)
            info[rule] = {"best": bm, "best1": b1, "n": len(by[rule])}
        data[name] = {"experts": experts, "info": info, "grid": (sorted({k[0] for k in by["kappa"]}, key=lambda x: (math.isinf(x), x)), sorted({k[1] for k in by["kappa"]}))}
        K = len(experts)
        rows_a.append([name, str(K), ", ".join(experts), f"subsets_dev_{name}.jsonl", str(len(recs))])
        u, u1 = info["uniform"]["best"]["metrics"]["mean_norm"], info["uniform"]["best1"]["metrics"]["mean_norm"]
        for rule in RULES:
            x = info[rule]
            bm, b1 = x["best"], x["best1"]
            rows_b.append([name, str(K), rule, "-" if rule == "uniform" else bfmt(bm["cfg"]["b"]), afmt(bm["cfg"]["alpha"]), pct(bm["metrics"]["mean_norm"]), pct(bm["metrics"]["mean_acc"]),
                           pct(bm["metrics"]["min_norm"]), pp(bm["metrics"]["mean_norm"] - u), "-" if rule == "uniform" else bfmt(b1["cfg"]["b"]), pct(b1["metrics"]["mean_norm"]), pp(b1["metrics"]["mean_norm"] - u1)])
        sh = [info[f"shuf{s}:kappa"]["best"]["metrics"]["mean_norm"] for s in range(3)]
        sh1 = [info[f"shuf{s}:kappa"]["best1"]["metrics"]["mean_norm"] for s in range(3)]
        kb = info["kappa"]["best"]["metrics"]["mean_norm"]
        rank = 1 + sum(1 for v in sh if v > kb)
        data[name].update({"u": u, "u1": u1, "sh": sh, "sh1": sh1, "rank": rank})
        rows_c.append([name, str(K), pct(u), pct(kb), pp(kb - u), pct(np.mean(sh)), pp(np.mean(sh) - u), pct(max(sh)), pp(info["kappa_rev"]["best"]["metrics"]["mean_norm"] - u),
                       pp(info["logkappa"]["best"]["metrics"]["mean_norm"] - u), f"{rank}/4",
                       pp(info["kappa"]["best1"]["metrics"]["mean_norm"] - u1), pp(np.mean(sh1) - u1), pp(info["kappa_rev"]["best1"]["metrics"]["mean_norm"] - u1)])
    nm = list(data)
    assert len(nm) == 7 and sum(n.startswith("rand4_") for n in nm) == 3 and sum(n.startswith("rand6_") for n in nm) == 2 and {"natural4", "synthetic4"} <= set(nm)
    assert all(len(data[n]["experts"]) == (4 if n.endswith("4") or n.startswith("rand4_") else 6) for n in nm)
    g_ = data["natural4"]["grid"]
    add_table("A7a", "Pre-registered expert subsets (configs/subsets.json)",
              "The seven pre-registered subsets of the 8 ViT-B/32 experts (two representative 4-expert sets, three random 4-expert sets, two random 6-expert sets) on which the literal rule was compared with uniform on the dev split; definition only, no results.",
              [CFG / "subsets.json", RES / "sweeps" / "subsets_dev_*.jsonl"], ["subset", "K", "experts", "result file", "# records"], rows_a, align="lrlll",
              notes=[f"Pre-registration record: \"{subsets['preregistered']}\"; purpose: \"{subsets['purpose']}\". The subset sweeps use a reduced grid: b in {{{', '.join(bfmt(b) for b in g_[0] if b > 0)}}} for the beta-rules, alpha in {{{', '.join(afmt(a) for a in g_[1])}}}; "
                     "uniform has 3 configs (alpha only), every other rule 12. Normalisation: each task's accuracy divided by that task's own-expert dev accuracy (as in A1), averaged over the subset's tasks."])
    add_table("A7b", "Subsets: per subset and rule, dev-best (reduced grid) and best at alpha = 1, with the difference to uniform",
              "For each pre-registered subset and rule, the best dev mean_norm (% over the subset's tasks) over the reduced (b, alpha) grid with its mean_acc and min_norm (%), and the best at alpha = 1, each with the difference to uniform's value on the same subset in percentage points; dev split.",
              [RES / "sweeps" / "subsets_dev_*.jsonl"], ["subset", "K", "rule", "best b", "best alpha", "mean_norm (%)", "mean_acc (%)", "min_norm (%)", "diff vs uniform best (pp)",
                                                        "best b at alpha=1", "mean_norm at alpha=1 (%)", "diff vs uniform alpha=1 (pp)"], rows_b, align="lrlrrrrrrrrr")
    add_table("A7c", "Subsets: literal rule, its shuffled nulls and variants relative to uniform, per subset",
              "Per subset, dev-best mean_norm of uniform and of the literal rule, the mean and maximum over the 3 shuffled-score nulls, and the differences to uniform (pp) for kappa, the shuffle mean, kappa_rev and logkappa (tuned over the reduced grid), the rank of kappa among kappa and its 3 shuffles, and the alpha = 1 differences; dev split.",
              [RES / "sweeps" / "subsets_dev_*.jsonl"],
              ["subset", "K", "uniform best (%)", "kappa best (%)", "kappa - uniform (pp)", "shuffle mean (%)", "shuffle mean - uniform (pp)", "shuffle max (%)", "kappa_rev - uniform (pp)", "logkappa - uniform (pp)",
               "kappa rank among kappa + 3 shuffles (1 = best)", "kappa - uniform at alpha=1 (pp)", "shuffle mean - uniform at alpha=1 (pp)", "kappa_rev - uniform at alpha=1 (pp)"], rows_c, align="lrrrrrrrrrrrrr")
    # summary across subsets
    names = list(data)
    def diffs(fn):
        return np.array([fn(data[n]) for n in names])
    d_tuned = {"kappa": diffs(lambda d: d["info"]["kappa"]["best"]["metrics"]["mean_norm"] - d["u"]),
               "kappa_rev": diffs(lambda d: d["info"]["kappa_rev"]["best"]["metrics"]["mean_norm"] - d["u"]),
               "logkappa": diffs(lambda d: d["info"]["logkappa"]["best"]["metrics"]["mean_norm"] - d["u"]),
               "shuf_mean": diffs(lambda d: np.mean(d["sh"]) - d["u"])}
    d_a1 = {"kappa": diffs(lambda d: d["info"]["kappa"]["best1"]["metrics"]["mean_norm"] - d["u1"]),
            "kappa_rev": diffs(lambda d: d["info"]["kappa_rev"]["best1"]["metrics"]["mean_norm"] - d["u1"]),
            "logkappa": diffs(lambda d: d["info"]["logkappa"]["best1"]["metrics"]["mean_norm"] - d["u1"]),
            "shuf_mean": diffs(lambda d: np.mean(d["sh1"]) - d["u1"])}
    rows = []
    for k in ("kappa", "kappa_rev", "logkappa", "shuf_mean"):
        v, v1 = d_tuned[k], d_a1[k]
        rows.append([k, pp(v.mean()), f"{int((v > 0).sum())}/{len(v)}", pp(v.min()), pp(v.max()), pp(v1.mean()), f"{int((v1 > 0).sum())}/{len(v1)}", pp(v1.min()), pp(v1.max())])
    add_table("A7d", "Subsets: summary over the 7 subsets (what subsets_summary.json reports)",
              "Across the 7 pre-registered subsets, the mean over subsets of (rule's dev-best mean_norm - uniform's dev-best mean_norm) in percentage points and the number of subsets with a positive difference, for the tuned configuration (reduced grid) and at alpha = 1; dev split.",
              [RES / "sweeps" / "subsets_summary.json", RES / "sweeps" / "subsets_dev_*.jsonl"],
              ["rule", "mean diff vs uniform, tuned (pp)", "# subsets positive", "min (pp)", "max (pp)", "mean diff at alpha=1 (pp)", "# positive at alpha=1", "min at alpha=1 (pp)", "max at alpha=1 (pp)"], rows, align="lrrrrrrrr",
              notes=["'shuf_mean' = mean over the 3 shuffled-score nulls shuf0..2:kappa of their dev-best mean_norm. Positive counts use a strict > 0.",
                     f"At alpha = 1 the literal rule averages {pp(d_a1['kappa'].mean())} vs uniform ({int((d_a1['kappa'] > 0).sum())}/7 subsets positive) while the shuffle mean averages {pp(d_a1['shuf_mean'].mean())} ({int((d_a1['shuf_mean'] > 0).sum())}/7); "
                     "the docs quote only the tuned columns. At alpha = 1 each rule picks its best b among 4 values (uniform has no choice)."])
    # compare with stored subsets_summary.json
    md = 0.0
    for n in names:
        st_ = summ["subsets"][n]
        md = max(md, abs(st_["shuf_mean"] - np.mean(data[n]["sh"])), abs(st_["kappa_rank_among_kappa_and_shuffled"] - data[n]["rank"]))
        for rule in RULES:
            md = max(md, abs(st_["best"][rule] - data[n]["info"][rule]["best"]["metrics"]["mean_norm"]), abs(st_["best_alpha1"][rule] - data[n]["info"][rule]["best1"]["metrics"]["mean_norm"]))
    for k in ("kappa", "kappa_rev", "logkappa", "shuf_mean"):
        md = max(md, float(np.max(np.abs(np.array(summ["diffs_vs_uniform"][k]) - d_tuned[k]))))
    check("derived", "subsets_summary.json reproduced from the subset records (max abs deviation)", "< 1e-9", f"{md:.1e}", ok=md < 1e-9)
    R1 = "REPORT_01 §5 / SUMMARY §1 / FULL §8"
    check(R1, "kappa minus uniform averaged over subsets (pp)", "-0.05", pp(d_tuned["kappa"].mean()))
    check(R1, "kappa: subsets positive", "3/7", f"{int((d_tuned['kappa'] > 0).sum())}/7")
    check(R1, "shuffled mean minus uniform (pp)", "-0.29", pp(d_tuned["shuf_mean"].mean()))
    check(R1, "shuffled mean: subsets positive", "0/7", f"{int((d_tuned['shuf_mean'] > 0).sum())}/7")
    check(R1, "kappa_rev minus uniform (pp)", "-0.82", pp(d_tuned["kappa_rev"].mean()))
    check(R1, "kappa_rev: subsets positive", "1/7", f"{int((d_tuned['kappa_rev'] > 0).sum())}/7")
    n4 = sum(1 for n in names if len(data[n]["experts"]) == 4)
    n6 = sum(1 for n in names if len(data[n]["experts"]) == 6)
    check("REPORT_01 §5", "subset composition: '4 representative/random 4-expert subsets and 2 random 6-expert subsets'", "4 + 2 (= 6 subsets, but '3/7' follows)", f"{n4} four-expert + {n6} six-expert = {len(names)} subsets",
          ok=(n4 == 4 and n6 == 2), note="configs/subsets.json: natural4, synthetic4 (representative) and rand4_0/1/2 are five 4-expert sets; FULL_REPORT §8 lists them correctly")
    SW["A7"] = {"data": data}


# --------------------------------------------------------------------------------------------
# A2h: which singular value moves; spectra claim checks
# --------------------------------------------------------------------------------------------
def build_A2_sigma() -> None:
    from scipy.stats import spearmanr
    rows = []
    for a in ("b32", "b16"):
        S = spectra(a)
        base, full, M = S["base"], S["full"].copy(), S["M"]
        full["s1_rel"] = (full.smax.values - base.loc[full.param, "smax"].values) / base.loc[full.param, "smax"].values
        full["sq_rel"] = (full.smin.values - base.loc[full.param, "smin"].values) / base.loc[full.param, "smin"].values
        full["grp"] = full.param.map(M["module"]).map(mod_group)
        rho_min, rho_max = {}, {}
        for p, gdf in full.groupby("param", sort=False):
            rho_min[p] = spearmanr(gdf.kappa2_raw, gdf.smin).statistic
            rho_max[p] = spearmanr(gdf.kappa2_raw, gdf.smax).statistic
        rm, rx = pd.Series(rho_min), pd.Series(rho_max)
        S["rho_smin"], S["rho_smax"] = rm, rx
        grp_of = M["module"].map(mod_group)
        for gname in ["all 73 maps", "MLP (fc1, fc2)", "attention (q, k, v, out)", "patch_embedding"]:
            gdf = full if gname.startswith("all") else full[full.grp == gname]
            ps = list(M.index) if gname.startswith("all") else list(M.index[grp_of == gname])
            rows.append([ARCH_NAME[a], gname, str(len(gdf)), g(gdf.s1_rel.abs().median(), 3), g(gdf.s1_rel.abs().max(), 3), g(gdf.sq_rel.abs().median(), 3),
                         g(gdf.sq_rel.abs().quantile(0.9), 3), g(gdf.sq_rel.abs().max(), 3), fnum(rm[ps].mean(), 3), fnum(rm[ps].median(), 3), fnum(rx[ps].mean(), 3), fnum(rx[ps].median(), 3)])
    add_table("A2h", "Which singular value moves: sigma_1 and sigma_q of each expert relative to the base, and within-map rank correlation of kappa with them",
              "For every (linear map, expert) pair, the relative change of the largest and smallest singular value of the full weight W_i = W0 + D_i with respect to the base W0, and the within-map Spearman correlation across the 8 experts of kappa2(W_i) with sigma_min(W_i) and with sigma_max(W_i), grouped by module type, for ViT-B/32 and ViT-B/16; weights only, dimensionless.",
              [SPEC_DIR["b32"] / "stats.csv", SPEC_DIR["b16"] / "stats.csv"],
              ["backbone", "group", "n (map, expert)", "median |s1_i/s1_0 - 1|", "max |s1_i/s1_0 - 1|", "median |sq_i/sq_0 - 1|", "90th pct |sq_i/sq_0 - 1|", "max |sq_i/sq_0 - 1|",
               "mean over maps of Spearman(kappa, smin)", "median over maps of Spearman(kappa, smin)", "mean over maps of Spearman(kappa, smax)", "median over maps of Spearman(kappa, smax)"],
              rows, align="llrrrrrrrrrr",
              notes=["sigma_q = smallest singular value (smin in stats.csv), s1 = smax. The 'mean over maps of Spearman(kappa, smin)' of the all-maps row is the within-layer rho quoted in the docs (peer claim, recomputed here from the master stats)."])


def build_spectra_checks() -> None:
    S = spectra("b32")
    M, full, base = S["M"], S["full"], S["base"]
    pm = S["pm"]
    D1 = "REPORT_01 §3 / FULL §4"
    k0 = M.kappa0
    check(D1, "kappa0 spans 14 ... 1.0e8 across the 73 maps: lower end", "14", f"{k0.min():.0f}")
    check(D1, "kappa0 upper end (1.0e8)", "1.0e8", f"{k0.max():.1e}".replace("e+0", "e"))
    mlp, att = M[M.module.isin(MLP_MODS)], M[M.module.isin(ATT_MODS)]
    check(D1, "number of MLP maps / attention maps", "24 / 48", f"{len(mlp)} / {len(att)}")
    n_in = int(((mlp.kappa0 >= 15) & (mlp.kappa0 <= 60)).sum())
    check(D1, "'MLP maps: kappa0 is 15-60' (all 24 inside?)", "24 of 24 in 15-60", f"{n_in} of {len(mlp)} in 15-60; range {mlp.kappa0.min():.1f}-{mlp.kappa0.max():.1f}", ok=(n_in == len(mlp)),
          note="outside: " + ", ".join(f"{p.replace('encoder.layers.', 'L').replace('.mlp.', ' ').replace('.weight', '')} {v:.1f}" for p, v in mlp.kappa0.items() if not (15 <= v <= 60)))
    check(D1, "MLP maps: experts differ from the base by 'about 0.1 %' (median rel_spread)", "about 0.1 %", f"{100 * mlp.rel_spread.median():.2f} %", ok=0.05 <= 100 * mlp.rel_spread.median() <= 0.2, approx=True)
    fo_fc1, fo_fc2 = S["summ"]["by_module"]["fc1"]["first_order_r_median"], S["summ"]["by_module"]["fc2"]["first_order_r_median"]
    check(D1, "first-order probe, MLP maps: median r per module type", "0.94-0.98", f"{min(fo_fc1, fo_fc2):.2f}-{max(fo_fc1, fo_fc2):.2f}")
    check(D1, "attention maps: kappa0 is 1e3-1e8", "1e3-1e8", f"{att.kappa0.min():.4g}-{att.kappa0.max():.3g}", ok=(att.kappa0.min() > 900 and att.kappa0.max() < 2e8), approx=True)
    check(D1, "attention maps: 'fine-tuning moves kappa by about 100 %' (median rel_spread)", "about 100 %", f"{100 * att.rel_spread.median():.0f} %", ok=0.8 <= att.rel_spread.median() <= 1.3, approx=True)
    nsing = int(M.base_sing.sum())
    check(D1, "27 of 73 base maps numerically rank-deficient at fp32 tolerance", "27", str(nsing))
    sh = M.fp32_noise_sd_over_spread.sort_values(ascending=False)
    over = [p for p, v in sh.items() if v > 0.5]
    short = lambda p: p.replace("encoder.layers.", "L").replace(".self_attn.", " ").replace(".weight", "")  # noqa: E731
    l1q = M.loc["encoder.layers.1.self_attn.q_proj.weight", "fp32_noise_sd_over_spread"]
    claimed = {"encoder.layers.0.self_attn.k_proj.weight", "encoder.layers.1.self_attn.k_proj.weight", "encoder.layers.0.self_attn.q_proj.weight"}
    check(D1, "'In L0/L1 k_proj and L0 q_proj, fp32 rounding noise is 50-90 % of the across-expert spread': maps with share > 50 %",
          "L0 k_proj, L1 k_proj, L0 q_proj (50-90 %)", "; ".join(f"{short(p)} {100 * sh[p]:.0f} %" for p in over), ok=(set(over) == claimed),
          note=f"also L2 k_proj; RESEARCH_LOG's 'L0/L1 q/k' would include L1 q_proj, whose share is {100 * l1q:.1f} %")
    check(D1, "median Spearman(kappa, ||D||) across maps", "0.05", f"{S['summ']['spearman_kappa_dnorm_median']:.2f}")
    sp_ = None
    # sigma-min correlation (needs A2h to have run)
    if "rho_smin" in S:
        check("REPORT_01 §3 / FULL §3-4", "within-layer Spearman(kappa, sigma_min) = -0.91 (mean over the 73 maps; peer claim recomputed from master stats)", "-0.91", f"{S['rho_smin'].mean():.2f}")
    cnt = S["summ"]["argmin_selection_counts"]
    check(D1, "argmin-kappa winners spread over all 8 experts", "8 distinct", f"{len(cnt)} distinct (counts {min(cnt.values())}..{max(cnt.values())})", ok=(len(cnt) == 8))
    f = full.copy()
    f["s1_rel"] = (f.smax.values - base.loc[f.param, "smax"].values) / base.loc[f.param, "smax"].values
    check("FULL §3 (side-track math note)", "sigma_1(W_i) ~ sigma_1(W0) for every expert: max relative deviation over (map, expert)", "approx. equal", f"{100 * f.s1_rel.abs().max():.2f} % (median {100 * f.s1_rel.abs().median():.3f} %)",
          ok=f.s1_rel.abs().max() < 0.02, approx=True)
    check("FULL §3 (side-track math note)", "real L0 k_proj base kappa0", "1.04e8", f"{M.loc['encoder.layers.0.self_attn.k_proj.weight', 'kappa0']:.3g}".replace("e+0", "e"))
    check("FULL §3 (side-track math note)", "L6 fc1 base kappa0", "14.4", f"{M.loc['encoder.layers.6.mlp.fc1.weight', 'kappa0']:.1f}")
    check("FULL §2", "layer-0 q/k_proj have kappa0 of about 1e8", "about 1e8", f"q {M.loc['encoder.layers.0.self_attn.q_proj.weight', 'kappa0']:.3g}, k {M.loc['encoder.layers.0.self_attn.k_proj.weight', 'kappa0']:.3g}",
          ok=all(5e7 <= M.loc[f'encoder.layers.0.self_attn.{x}_proj.weight', 'kappa0'] <= 2e8 for x in "qk"), approx=True)
    ea = expert_acc("b32")
    cross = ea["cross"]["dev"]
    bs = max(np.mean([cross[e][t] / ea["expert"]["dev"][t] for t in TASKS8]) for e in TASKS8)
    zsn = np.mean([ea["zeroshot"]["dev"][t] / ea["expert"]["dev"][t] for t in TASKS8])
    check("RESEARCH_LOG wave 1 / REPORT_01 §2", "best single expert (dev mean_norm 54.8) is below zero-shot (55.0)", "54.8 < 55.0", f"{100 * bs:.1f} < {100 * zsn:.1f}", ok=bs < zsn)


# --------------------------------------------------------------------------------------------
# Legends, checks, manifest, header
# --------------------------------------------------------------------------------------------
def build_legends() -> None:
    rules = sorted(sweeps()["by"], key=lambda r: (FAMILY_ORDER.index(family(r)), r))
    rows = []
    for r in rules:
        row = rule_legend_row(r)
        rows.append([family(r).split(" ", 1)[1]] + row)
    rows.append(["single expert", "single:cars", "(no score; lambda = one-hot on the expert with the best dev row of the cross matrix)", "'best_single' frozen config (experiments/make_freeze_list.py, freeze_eval.py)"])
    add_table("L1", "Legend of rule names (every rule that appears in A3-A7)",
              "Definition of each rule name used in the tables: the score c_i[p] per expert i and linear map p (lower score -> larger weight, lambda = softmax(-beta c)), quoted from src/condmerge/scores.py (verbatim lambda body) and the gloss from docs/PROTOCOL.md / scores.py comments; no numbers.",
              [REPO / "src" / "condmerge" / "scores.py", DOCS / "PROTOCOL.md", REPO / "experiments" / "rule_sweep.py"], ["family", "rule", "score definition (verbatim from scores.py)", "gloss (PROTOCOL.md / scores.py comment)"],
              rows, align="llll",
              notes=["Notation in the expressions: full(s) = statistic s of the full expert weight W_i per map (kappa2_raw = s1/sq from a float64 SVD of the stored float32 matrix; kappa2 = same but +inf when the numerical rank is below q; smax/smin = extreme singular values; kappa_q90 = s1 / s_{ceil(0.9 q)}; kappa_plus_0.001 = s1 / s_r on the r singular values above 1e-3 s1); "
                     "upd(s) = statistic of the task update D_i = W_i - W0; the *_sup kinds are spectra of M C^{1/2} (C = pooled calibration second moment, budget B); zscore = within-map standardisation across experts; `_rev` = negated score (larger statistic -> larger weight).",
                     "Layerwise rules apply the softmax independently to each of the 73 maps; `global:<rule>` replaces the per-map scores by one parameter-weighted mean per expert; temperature beta = b / scale (scale = median over maps of the across-expert s.d. of the scores) except the absolute-beta rules gd_invvar, global:gd_invvar."])
    rows = [["RULE|tuned", "frozen configuration = dev-best (b, alpha) over the rule's whole grid (experiments/make_freeze_list.py)"],
            ["RULE|alpha1", "frozen configuration = dev-best b at alpha = 1 (the literal normalised average)"],
            ["best_single", f"single:EXPERT, alpha = 1: the expert with the highest dev mean_norm in the 8x8 cross matrix ({SW['best_single_expert']}), evaluated alone"],
            ["b16xfer|RULE|tuned (or alpha1)", "the B/32-frozen config of RULE evaluated once on ViT-B/16 (configs/b16_transfer_configs.json)"],
            ["b16xfer|NULL@BASE|tuned", "shuffled-score null NULL (e.g. shuf0:kappa) evaluated at exactly the frozen (b, alpha) of BASE|tuned"],
            ["w1 / w2 / w3", f"frozen_configs_w1/w2/w3.json: freeze waves first committed in {' / '.join(SW['freeze_commits'].values())}"],
            ["mean_norm", "mean over the 8 tasks (or the subset's tasks) of accuracy divided by that task's own-expert accuracy on the same split (%)"],
            ["mean_acc", "mean over tasks of absolute top-1 accuracy (%)"],
            ["min_norm", "smallest per-task normalised accuracy (%)"],
            ["n_eff", "exp(entropy of the per-map coefficient vector), averaged over the 73 maps (8 = uniform, 1 = one expert per map)"],
            ["dev / test", "dev = 20 % of each official test split (permutation seed 0), used for all selection; test = the other 80 %, only touched by frozen configs (docs/PROTOCOL.md)"],
            ["b, alpha, beta", "beta = b / scale(rule); b = inf = per-map argmin; W = W0 + alpha sum_i lambda_i D_i; alpha = 1 is the normalised average; uniform with alpha = K s is Task Arithmetic with scale s"],
            ["pp", "percentage points (difference of two percentages)"]]
    add_table("L2", "Legend of tags, variants and metrics", "Meaning of the tag suffixes, the metric names and units used in all tables of this appendix.", [DOCS / "PROTOCOL.md", REPO / "experiments" / "make_freeze_list.py",
              REPO / "experiments" / "freeze_eval.py", CFG / "b16_transfer_configs.json"], ["term", "meaning"], rows, align="ll")


def verify_claim_tokens() -> None:
    """Soft check that the numbers typed as 'claimed' really occur in the quoted docs (guards against typos in the claims)."""
    txt = ""
    for fn in ("REPORT_01_literal_rule.md", "FULL_REPORT.md", "RESEARCH_LOG.md", "SUMMARY.md", "REPORT_02_inverse_problem.md", "PROTOCOL.md", "MATH_NOTES.md"):
        txt += "\n" + read_text(DOCS / fn)
    toks = set(re.findall(r"\d+(?:\.\d+)?", txt.replace("\\|", "|")))
    for c in CHECKS:
        cl = c["claimed"]
        if c["doc"] == "derived" or re.search(r"[A-Za-z]", re.sub(r"e[+-]?\d+", "", cl)) or not _NUM.findall(cl):
            continue
        missing = [t.lstrip("+-") for t in _NUM.findall(cl) if t.lstrip("+-") not in toks and "e" not in t.lower()]
        if missing:
            warn(f"claimed value(s) {missing} of check '{c['claim']}' ({c['doc']}) not found as numbers in the docs")
            c["note"] = (c["note"] + " " if c["note"] else "") + f"[WARNING: {missing} not found in the docs text]"


def build_checks() -> None:
    verify_claim_tokens()
    bad = [c for c in CHECKS if c["status"] == "DISCREPANCY"]
    soft = [c for c in CHECKS if c["status"] in ("rounding", "approx")]
    rows = [[str(i + 1), c["doc"], c["claim"], c["claimed"], c["computed"], c["status"], c["note"]] for i, c in enumerate(bad + soft)]
    add_table("X0", f"Discrepancies between the docs and the raw results ({len(bad)} substantive, {len(soft)} rounding-level or approximate, of {len(CHECKS)} checks)",
              "Claims quoted from docs/REPORT_01_literal_rule.md, docs/FULL_REPORT.md, docs/RESEARCH_LOG.md and docs/SUMMARY.md whose value recomputed from the raw files by this script does not match exactly: substantive discrepancies first, then rounding-level (<= 1 unit in the last quoted digit) and approximate-claim entries; units as in the claim (mostly % and pp).",
              [DOCS / "REPORT_01_literal_rule.md", DOCS / "FULL_REPORT.md", DOCS / "RESEARCH_LOG.md", DOCS / "SUMMARY.md"], ["#", "doc / section", "claim", "claimed", "recomputed", "status", "note"], rows, align="lllllll",
              notes=["Status: DISCREPANCY = the quoted value or statement is not supported by the data; rounding = every number differs from the quoted one by at most one unit of its last digit; approx = the claim itself is approximate ('about ...') and the data are consistent with it within the stated tolerance (in the note)."])
    rows = [[str(i + 1), c["doc"], c["claim"], c["claimed"], c["computed"], c["status"], c["note"]] for i, c in enumerate(CHECKS)]
    add_table("X1", f"All {len(CHECKS)} cross-checks (docs vs raw results, plus internal consistency checks)",
              "Every comparison made by this script: numbers quoted in the docs (docs/REPORT_01_literal_rule.md, docs/FULL_REPORT.md, docs/RESEARCH_LOG.md, docs/SUMMARY.md) against the values recomputed from the raw files, and 'derived' rows that check stored summaries against recomputation; units as in the claim.",
              [DOCS / "REPORT_01_literal_rule.md", DOCS / "FULL_REPORT.md", DOCS / "RESEARCH_LOG.md", DOCS / "SUMMARY.md", "all inputs listed in F1"],
              ["#", "doc / section", "claim", "claimed", "recomputed", "status", "note"], rows, align="lllllll")


def build_manifest() -> None:
    import fnmatch
    rows = []
    for path, v in sorted(USED.items()):
        cited = []
        for t in TABLES:
            for src in t["sources"]:
                pat = re.sub(r"\s*\(.*\)$", "", src)
                pat = pat.replace("<key>", "*")
                if fnmatch.fnmatch(path, pat) or path == pat:
                    cited.append(t["tid"])
                    break
        rows.append([path, str(v["bytes"]), str(v["lines"]), v["md5"][:12], ", ".join(sorted(set(cited), key=tid_key)) or "-"])
    if NPZ_USED:
        comb = hashlib.md5("".join(f"{k}:{NPZ_USED[k]}" for k in sorted(NPZ_USED)).encode()).hexdigest()
        rows.append([f"artifacts/runs/freeze_correct/<key>.npz ({len(NPZ_USED)} files read: per-example correctness of the frozen test evaluations)", "-", "-", comb[:12] + " (md5 of the sorted per-file md5s)", "A1a-A1c (n test), A4d, A5d"])
    add_table("F1", f"Input files read by make_tables.py ({len(USED)} files + the per-example arrays)",
              "Every raw file parsed to build the tables, with size, line count and the first 12 hex digits of the MD5 of the exact bytes read, so that a later change of any input can be detected; no results.",
              ["(repository /data/home/azpitarte/condmerge, read-only)"], ["path (relative to repo root)", "bytes", "newlines", "md5[:12]", "cited as source by"], rows, align="lrrll")


def build_header() -> str:
    head = git_head()
    lines = ["# Appendix A: weight-conditioning experiments on full fine-tunes (CLIP ViT-B/32 and ViT-B/16, 8 tasks)\n",
             f"Generated by `make_tables.py` from the raw result files of `/data/home/azpitarte/condmerge` (git HEAD at generation: {head}). Every number below was computed by the script from those files "
             "(list and MD5s in F1). Hand-written content is limited to captions (which state design parameters such as grid sizes, split fractions and counts transcribed from the code and configs; several are asserted against the data inside the script) "
             "and to the quotations of documentation claims in X0/X1, which are the objects being checked.\n",
             "**Conventions.** Accuracy metrics are in percent with 2 decimals; differences are in percentage points (pp). `mean_norm` = normalised accuracy (each task's accuracy divided by that task's own-expert accuracy on the same split, averaged over the tasks); "
             "`mean_acc` = mean absolute accuracy; `min_norm` = worst task's normalised accuracy. dev = 20 % of each official test set (selection only), test = the other 80 % (frozen configurations only). "
             "Row labels (rule names, tags, parameter names) are copied from the files; the pipe in a tag such as `uniform|tuned` is escaped as `\\|` inside Markdown tables. "
             "The reference for all test differences is `uniform|tuned` (B/32) and `b16xfer|uniform|tuned` (B/16).\n",
             "**Contents.**\n"]
    for t in sorted(TABLES, key=lambda t: tid_key(t["tid"])):
        lines.append(f"- {t['tid']}: {t['title']} ({len(t['rows'])} rows)")
    lines.append("")
    # scope coverage
    scope_pats = ["results/experts/*", "results/sanity/*", "results/spectra/*", "results/sweeps/*", "results/freeze/*", "results/pairs/*", "results/b16/**/*", "configs/b16_*", "configs/subsets.json", "results/figures/pilot_rule_vs_b.png"]
    scope = sorted({str(f.relative_to(REPO)) for pat in scope_pats for f in REPO.glob(pat) if f.is_file()})
    unread = [f for f in scope if f not in USED]
    lines.append(f"**Coverage of the requested scope.** {len(scope)} files match the requested patterns (results/experts, sanity, spectra, sweeps, freeze, pairs, b16/**, configs/b16_*, configs/subsets.json, results/figures/pilot_rule_vs_b.png); "
                 f"{len(scope) - len(unread)} were parsed by the script. Not parsed: " + (", ".join(f"`{f}`" for f in unread) if unread else "none") +
                 (" (image: the data behind its panels are tabulated in A3i, A3j and A3o)" if any(f.endswith(".png") for f in unread) else "") + ".\n")
    if WARN:
        lines.append("**Script warnings:** " + "; ".join(WARN) + "\n")
    return "\n".join(lines) + "\n"


def build_A0() -> None:
    sums, res32, res16 = SW["sums"], SW["A4"]["res"], SW["A5"]["res"]
    ci = lambda v: f"[{pp(v['ci95'][0])}, {pp(v['ci95'][1])}]"  # noqa: E731
    rows = []
    for rule in ["uniform", "kappa", "kappa_rev", "logkappa", "logkappa_rev", "kappa_strict", "kappa_z", "global:kappa_z", "metagpt", "dnorm", "logkappa_q90_rev", "global:logkappa"]:
        s_ = sums[rule]
        t, t1 = res32[f"{rule}|tuned"], res32[f"{rule}|alpha1"]
        b16 = res16.get(f"b16xfer|{rule}|tuned")
        bm = s_["best"]["cfg"]
        rows.append([rule, pct(s_["best1"]["metrics"]["mean_norm"]), pct(s_["best"]["metrics"]["mean_norm"]), f"({bdisp(rule, s_['best'])}, {afmt(bm['alpha'])})",
                     pct(t["test_mean_norm"]), pct(t["test_mean_acc"]), f"{pp(t['diff_vs_ref'])} {ci(t)}", pct(t1["test_mean_norm"]),
                     "-" if b16 is None else pct(b16["test_mean_norm"]), "-" if b16 is None else f"{pp(b16['diff_vs_ref'])} {ci(b16)}"])
    nd = SW["NULLS"]["kappa"]["rules"]
    def rng(vals, f=pct):
        return f"{f(np.mean(vals))} ({f(min(vals))} .. {f(max(vals))})"
    sh_t = [res32[f"{n}|tuned"] for n in nd]
    sh_1 = [res32[f"{n}|alpha1"] for n in nd]
    b16n = [res16[f"b16xfer|shuf{k}:kappa@kappa|tuned"] for k in range(5)]
    rows.append(["shuf0..4:kappa (null, mean and range)", rng(SW["NULLS"]["kappa"]["a1"]), rng(SW["NULLS"]["kappa"]["tuned"]), "-", rng([v["test_mean_norm"] for v in sh_t]),
                 rng([v["test_mean_acc"] for v in sh_t]), rng([v["diff_vs_ref"] for v in sh_t], pp), rng([v["test_mean_norm"] for v in sh_1]),
                 rng([v["test_mean_norm"] for v in b16n]), rng([v["diff_vs_ref"] for v in b16n], pp)])
    bs = res32["best_single"]
    rows.append(["best_single (cars)", pct(bs["dev_metric"]), pct(bs["dev_metric"]), "(single:cars, 1)", pct(bs["test_mean_norm"]), pct(bs["test_mean_acc"]), f"{pp(bs['diff_vs_ref'])} {ci(bs)}", "-", "-", "-"])
    add_table("A0", "Headline comparison: key rules on dev, frozen ViT-B/32 test and ViT-B/16 transfer",
              "For the key rules, dev mean_norm at alpha = 1 and tuned (best over the shared grid, with its (b, alpha)), the frozen ViT-B/32 TEST mean_norm / mean_acc and difference to uniform|tuned with its 95 % CI, the alpha = 1 test mean_norm, and the ViT-B/16 transfer test mean_norm and difference to uniform|tuned; "
              "mean_norm and mean_acc in %, differences in pp; selection on dev (20 %), test on the other 80 %.",
              [RES / "sweeps" / "rule_sweep_dev*.jsonl (9 shard files, listed in F1)", RES / "freeze" / "test_comparison.json", RES / "b16" / "freeze" / "transfer_comparison.json"],
              ["rule", "dev mean_norm, alpha=1 (%)", "dev mean_norm, tuned (%)", "(b, alpha) tuned on dev", "B/32 test mean_norm, tuned (%)", "B/32 test mean_acc, tuned (%)", "B/32 test diff vs uniform|tuned (pp) [CI95]",
               "B/32 test mean_norm, alpha=1 (%)", "B/16 test mean_norm, tuned (%)", "B/16 test diff vs uniform|tuned (pp) [CI95]"], rows, align="lrrlrrlrrl",
              notes=["Every entry is also in A3a/A3b (dev), A4a (B/32 test) and A5a (B/16 test). '-' = configuration not run (B/16 transfer covered only the rules listed in A5a). Null row: mean over the 5 shuffled-score nulls with min .. max in parentheses. "
                     "B/16 numbers use the (b, alpha) frozen on B/32 dev, not B/16-tuned values."])


def build_A3n() -> None:
    """Per-map concentration of the literal rule, reconstructed from stats.csv with the sweep's own beta = b / scale."""
    S = spectra("b32")
    full = S["full"]
    piv = full.pivot(index="param", columns="expert", values="kappa2_raw")[TASKS8].loc[S["M"].index]
    grp = S["M"].module.map(mod_group)
    by = sweeps()["by"]["kappa"]
    BS = sweeps()["bs"]

    def neff(v: np.ndarray, beta: float) -> float:
        if math.isinf(beta):
            return 1.0
        z = -beta * v
        z = z - z.max()
        w = np.exp(z)
        w /= w.sum()
        pw = w[w > 0]
        return float(np.exp(-(pw * np.log(pw)).sum()))
    rows, dm, dmed = [], 0.0, 0.0
    for b in BS:
        rec = by[(b, 1.0)]
        beta = INF if math.isinf(b) else b / rec["cfg"]["scale"]
        ne = pd.Series({p: neff(piv.loc[p].values, beta) for p in piv.index})
        dm = max(dm, abs(ne.mean() - rec["coeff_summary"]["n_eff_mean"]))
        dmed = max(dmed, abs(ne.median() - rec["coeff_summary"]["n_eff_median"]))
        mlp, att, pat = ne[grp == "MLP (fc1, fc2)"], ne[grp == "attention (q, k, v, out)"], ne[grp == "patch_embedding"]
        rows.append([bfmt(b), fnum(mlp.mean(), 3), fnum(mlp.min(), 3), fnum(att.mean(), 2), fnum(att.median(), 2), f"{int((att < 2).sum())}/{len(att)}", f"{int((att < 1.1).sum())}/{len(att)}",
                     fnum(pat.mean(), 3), fnum(ne.mean(), 4), fnum(rec["coeff_summary"]["n_eff_mean"], 4)])
    check("derived", "per-map n_eff of `kappa` reconstructed from stats.csv reproduces the recorded n_eff_mean / n_eff_median (max abs deviation over b)", "< 1e-6", f"{max(dm, dmed):.1e}", ok=max(dm, dmed) < 1e-6)
    last = rows[-2]  # b = 128
    check("REPORT_01 §2 / FULL §5", "'The literal rule concentrates only attention maps': at b = 128 the 24 MLP maps stay uniform (min n_eff over MLP maps)", "~8", last[2], ok=float(last[2]) > 7.9,
          note=f"attention maps at b=128: mean n_eff {last[3]}, {last[5]} maps with n_eff < 2")
    add_table("A3n", "Literal rule `kappa`: which maps get concentrated (per-map n_eff reconstructed from the weights)",
              "For each b, the per-map effective number of experts n_eff of lambda = softmax(-beta kappa2_raw), beta = b/scale as in the sweep, reconstructed from results/spectra/stats.csv and summarised by MLP / attention / patch-embedding maps, next to the recorded sweep n_eff_mean; weights only, dimensionless (8 = uniform, 1 = one expert).",
              [SPEC_DIR["b32"] / "stats.csv", RES / "sweeps" / "rule_sweep_dev_literal.jsonl"],
              ["b", "MLP maps (24): mean n_eff", "MLP maps: min n_eff", "attention maps (48): mean n_eff", "attention: median n_eff", "# attention maps with n_eff < 2", "# attention maps with n_eff < 1.1",
               "patch-embedding map: n_eff", "all 73 maps: reconstructed mean n_eff", "recorded n_eff_mean (sweep)"], rows, align="lrrrrrrrrr",
              notes=["The reconstruction reproduces the recorded n_eff_mean and n_eff_median to < 1e-6 for every b (check in X1)."])


# @@INSERT_HERE@@


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------
def git_head() -> str:
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            f = REPO / ".git" / ref
            if f.exists():
                return f.read_text().strip()[:7] + f" ({ref})"
            for ln in (REPO / ".git" / "packed-refs").read_text().splitlines():
                if ln.endswith(" " + ref):
                    return ln.split()[0][:7] + f" ({ref})"
        return head[:7]
    except Exception:
        return "unknown"


def tid_key(tid: str):
    sec = {"L": 0, "A": 1, "X": 2, "F": 3}.get(tid[0], 4)
    toks = re.findall(r"\d+|[a-z]+|'", tid[1:])
    return (sec, [(0, int(x)) if x.isdigit() else (1, x) for x in toks])


def render() -> str:
    out = []
    for t in sorted(TABLES, key=lambda t: tid_key(t["tid"])):
        out.append(f"### {t['tid']}. {t['title']}\n")
        out.append(f"**Varied / split / metric / units:** {t['what']}\n")
        out.append("**Source:** " + "; ".join(f"`{s}`" for s in t["sources"]) + f"  \n**Rows:** {len(t['rows'])}\n")
        out.append(md_table(t["headers"], t["rows"], t["align"]) + "\n")
        for n in t["notes"]:
            out.append(f"*Note:* {n}\n")
    return "\n".join(out)


def main() -> None:
    BUILDERS = [build_A1, build_A2, build_A2_sigma, build_A3, build_A3n, build_A4, build_A5, build_A6, build_A7, build_A0, build_spectra_checks, build_legends, build_checks, build_manifest]
    for fn in BUILDERS:
        fn()
    body = build_header() + render()
    OUT_MD.write_text(body)
    print(f"wrote {OUT_MD} ({len(body.splitlines())} lines, {len(TABLES)} tables)")
    for t in TABLES:
        print(f"  {t['tid']:6s} rows={len(t['rows']):4d}  {t['title']}")


if __name__ == "__main__":
    main()
