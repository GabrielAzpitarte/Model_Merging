#!/usr/bin/env python
"""Appendix tables B1-B7 (diagnostics of the aa/side track) for the condition-number merging report.

Every number in tables.md is computed here from the raw result files under
/data/home/azpitarte/condmerge_aa/results/aa (read-only). Nothing is typed in by hand except
descriptive text; setup descriptions that mention numbers are asserted against the producing scripts.

Run:  OMP_NUM_THREADS=4 /data/home/azpitarte/lora_merging_ica/.venv/bin/python make_tables.py
Output: tables.md next to this file (+ csv dumps in ./csv, scratch in ./_work).

Metric conventions (all files in this scope): accuracy = fraction correct on a task's split, the
"avg_acc" of a file = unweighted mean over the 8 tasks of the ABSOLUTE accuracy (NOT normalised by the
expert's own accuracy). Normalised accuracy only appears in master's files used as external references.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

WT = Path("/data/home/azpitarte/condmerge_aa")
OUT_DIR = Path("/tmp/claude-1061/-data-home-azpitarte/3fa82598-2657-4652-97a2-9c24f2860353/scratchpad/report_build/B_diagnostics")
WORK = OUT_DIR / "_work"
CSV_DIR = OUT_DIR / "csv"
AA = WT / "results" / "aa"
TASKS = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
BGRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf]

BODY: list[str] = []          # markdown body, in order
INDEX: list[tuple] = []       # (id, title, nrows)
V: dict = {}                  # computed values reused by the cross-check section
PARSE_NOTES: list[str] = []   # files / items that could not be parsed or verified
CHECKS: list[dict] = []       # cross-check rows (filled by section X)


# ------------------------------------------------------------------ helpers
def jload(p) -> dict:
    return json.loads(Path(p).read_text())


def jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def rel(p) -> str:
    return str(Path(p).relative_to(WT))


def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def f(x, d=3, sign=False) -> str:
    """Plain float formatting; inf/nan handled."""
    if x is None or isnan(x):
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def pct(x, d=2, sign=False) -> str:
    if x is None or isnan(x):
        return "–"
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def g(x, d=3) -> str:
    """General (significant-digit) formatting for wide-range numbers."""
    if x is None or isnan(x):
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{d}g}"


def fb(b) -> str:
    """Temperature multiplier label."""
    if b is None or isnan(b):
        return "–"
    if isinstance(b, float) and math.isinf(b):
        return "inf"
    return f"{b:g}"


def kfmt(x) -> str:
    """Condition-number style formatting: plain digits below 1e6, 3 significant digits in scientific notation above."""
    if x is None or isnan(x):
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if abs(x) < 1e6:
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}" if abs(x) >= 10 else f"{x:.3f}"
    return f"{x:.3g}"


def lst(xs, d=4) -> str:
    return "[" + ", ".join(g(x, d) for x in xs) + "]"


def pm(x, p, d=3, dp=3) -> str:
    """'+0.045 (0.307)' as in score_vs_targets.md."""
    return f"{x:+.{d}f} ({p:.{dp}f})"


def md_table(headers, rows, align=None) -> str:
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")

    n = len(headers)
    if align is None:
        align = "l" + "r" * (n - 1)
    assert len(align) == n, (len(align), n, headers)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        assert len(r) == n, (len(r), n, r)
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def add_table(tid, title, what, sources, headers, rows, align=None, legend=None, notes=None, level=3):
    """Emit one table with ID heading, what/split/metric sentence, sources, legend, notes."""
    INDEX.append((tid, title, len(rows)))
    B = BODY
    B.append(f"{'#' * level} {tid}. {title}")
    B.append("")
    B.append(f"**What varies / split / metric / units.** {what}")
    B.append("")
    B.append("**Source.** " + "; ".join(f"`{s}`" for s in sources))
    B.append("")
    B.append(md_table(headers, rows, align))
    B.append("")
    if legend:
        B.append("**Legend.**")
        for l in legend:
            B.append(f"- {l}")
        B.append("")
    if notes:
        B.append("**Notes.**")
        for n in notes:
            B.append(f"- {n}")
        B.append("")


def add_text(lines, level=None, title=None):
    if title:
        BODY.append(f"{'#' * level} {title}")
        BODY.append("")
    for l in lines:
        BODY.append(l)
    BODY.append("")


def mtype(name: str) -> str:
    for k in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"):
        if k in name:
            return k
    return "other"


def mgroup(name: str) -> str:
    if "self_attn" in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "patch_embedding"


def short(name: str) -> str:
    return name.replace("encoder.layers.", "L").replace(".self_attn", "").replace(".mlp", "").replace(".weight", "")


def rk(m: np.ndarray) -> np.ndarray:
    return m.argsort(1).argsort(1).astype(np.float64)


def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


def within_exp(ra: np.ndarray, rb: np.ndarray) -> float:
    ea, eb = ra - ra.mean(0, keepdims=True), rb - rb.mean(0, keepdims=True)
    return float((ea * eb).sum() / np.sqrt((ea ** 2).sum() * (eb ** 2).sum()))


def neff_rows(lam: np.ndarray) -> np.ndarray:
    """exp(entropy) per map, same floor (1e-12) as scripts/aa_analyze_oracle.py."""
    return np.exp(-np.sum(lam * np.log(np.maximum(lam, 1e-12)), axis=1))


def load_lambda(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    order = list(dict.fromkeys(df["param"]))
    exp = list(dict.fromkeys(df["expert"]))
    assert exp == TASKS, exp
    return df.pivot(index="param", columns="expert", values="lambda").loc[order, TASKS]



def _perm_counts() -> dict:
    sc = WT / "scripts"
    g1 = (sc / "aa_analyze_grad_kappa.py").read_text()
    g2 = (sc / "aa_decompose_pref.py").read_text()
    g3 = (sc / "aa_score_vs_targets.py").read_text()
    g4 = (sc / "aa_analyze_oracle.py").read_text()
    out = dict(
        grad_map=int(re.search(r'"--perm", type=int, default=(\d+)', g1).group(1)),
        dec_exp=int(re.search(r"for _ in range\((\d+)\):\s+Sp = S\.copy\(\)", g2).group(1)),
        svt_map=int(re.search(r"N_PERM = (\d+)", g3).group(1)),
        svt_exp=int(re.search(r"for _ in range\(N_PERM // (\d+)\)", g3).group(1)),
        orc_map=int(re.search(r"for _ in range\((\d+)\):\s+perm = np\.argsort", g4).group(1)),
        orc_exp=int(re.search(r"for _ in range\((\d+)\):\s+Sp = S\.copy\(\)", g4).group(1)),
    )
    out["svt_exp"] = out["svt_map"] // out["svt_exp"]
    return out


PERMS = _perm_counts()

# ==================================================================== B1
def build_B1():
    src_path = WT / "scripts" / "aa_synthetic_checks.py"
    src = src_path.read_text()
    stored = jload(AA / "synthetic" / "synthetic_checks.json")
    # ---- re-run a COPY of the script in the scratchpad (seed 0, CPU, ~3 s) and compare with the stored JSON
    WORK.mkdir(parents=True, exist_ok=True)
    cp = WORK / "aa_synthetic_checks_copy.py"
    cp.write_text(src)
    rerun_json = WORK / "synthetic_rerun.json"
    res = subprocess.run([sys.executable, str(cp), str(rerun_json)], capture_output=True, text=True, cwd=str(WORK),
                         env={**os.environ, "OMP_NUM_THREADS": "4"})
    (WORK / "synthetic_rerun.log").write_text(res.stdout + res.stderr)
    assert res.returncode == 0, res.stderr
    rer = jload(rerun_json)

    def maxrel(x, y):
        if isinstance(x, dict):
            return max([maxrel(x[k], y[k]) for k in x] + [0.0])
        if isinstance(x, list):
            return max([maxrel(a, b) for a, b in zip(x, y)] + [0.0]) if len(x) == len(y) else math.inf
        if isinstance(x, (int, float)):
            return abs(x - y) / abs(x) if x != 0 else abs(y)
        return 0.0 if x == y else math.inf

    keys_cmp = [k for k in rer if k in stored]
    max_dev = max(maxrel(stored[k], rer[k]) for k in keys_cmp)
    V["synthetic_rerun_maxrel"] = max_dev
    log_lines = [l for l in res.stdout.splitlines() if l.startswith("[")]
    parsed = []
    for l in log_lines:
        m = re.match(r"\[(PASS|FAIL)\] ((C\d+) .*)$", l)
        assert m, l
        parsed.append((m.group(3), m.group(1), m.group(2)))
    by_id = defaultdict(list)
    for cid, st, txt in parsed:
        by_id[cid].append((st, txt))
    final_ok = "ALL CHECKS PASS" in res.stdout
    assert stored["fails"] == [] and final_ok

    def detail(prefix):
        for cid, st, txt in parsed:
            if txt.startswith(prefix):
                return txt[len(prefix):].strip()
        raise KeyError(prefix)

    # ---- claims from the script docstring
    doc = re.search(r'"""(.*?)"""', src, re.S).group(1)
    claims, cur = OrderedDict(), None
    for line in doc.splitlines():
        m = re.match(r"^ (C\d+) \[(.*?)\]\s+(.*)$", line)
        m2 = re.match(r"^ (C\d+) (\[new bridge\])\s+(.*)$", line)
        if m or m2:
            mm = m or m2
            cur = mm.group(1)
            claims[cur] = [mm.group(2).strip("[]"), mm.group(3).strip()]
        elif cur and line.startswith("      "):
            claims[cur][1] += " " + line.strip()
    assert list(claims) == [f"C{i}" for i in range(1, 12)], list(claims)

    # ---- setup texts (numbers asserted against the script source)
    def need(*snips):
        for s in snips:
            assert s in src, f"snippet not in source: {s}"

    J = stored
    eps = re.search(r"eps = ([0-9.e-]+)", src).group(1)
    need("d, n_mc = 50, 20000", "v = np.array([0.5, 1.0, 2.0, 4.0])", "< 0.03", "atol=1e-6")
    c1 = (f"4 unbiased estimates of one d=50 target with independent Gaussian errors of total variance v={lst(J['C1']['v'])}; "
          f"QP over the simplex vs closed form inverse variance; Monte Carlo risk with 20000 draws (opt and uniform weights)")
    need("d, nu2 = 8, 1.0", "kap = np.array([1.0, 2.0, 4.0, 8.0, 16.0])", "for _ in range(200)",
         "np.geomspace(1e-3, 10, 400)", "np.linspace(0, 4, 401)")
    c2 = (f"LS designs Z_i = diag(1,..,1,1/kappa_i), d=8, nu^2=1, kappa={lst(J['C2']['kappa'])} so v_i=nu^2(d-1+kappa_i^2)={lst(J['C2']['v'])}; "
          f"risk = sum lambda^2 v; compares inverse-variance oracle, uniform, best exp(-beta*kappa) (400 betas in [1e-3,10]), "
          f"best kappa^-p (401 p in [0,4]) and the reversed rule; eq.21 bounds on 200 random 20xd designs; two counterexamples")
    need("for rho in (0.0, 0.5, 0.9)", "C = np.array([[1.0, 1.2], [1.2, 2.0]])", "softmax(np.zeros(3))")
    c3 = ("2x2 covariance C=[[1,1.2],[1.2,2]] (GLS weights, negativity, simplex clipping); Var((e1+e2)/2)=(1+rho)/2 at rho in {0,0.5,0.9}; "
          "a duplicated expert under uniform softmax over 3 items")
    need(f"eps = {eps}", "np.diag([1, eps]), np.diag([eps, 1])", "kappa(7 * W1)")
    c4 = (f"W1=diag(1,eps), W2=diag(eps,1) with eps={eps}: equal kappa, kappa of the average, the cancelling pair W2=-W1, scale invariance kappa(7*W1)")
    need("np.diag([1.0, 1e-4]), np.eye(2)", "np.linspace(0, 1, 101)", "-1e-3 * np.array")
    c5 = ("two 'tasks' W1=diag(1,1e-4) (kappa 1e4) and W2=I (kappa 1) with isotropic inputs; loss(lam)=0.5(‖W-W1‖²+‖W-W2‖²), W=lam*W1+(1-lam)*W2 "
          "on a 101-point grid; the rule softmax(-1e-3*kappa) puts weight lam on the high-kappa expert")
    need("t1, t2 = np.array([1.0, 1.0]), np.array([2.0, 2.0])", "np.diag([100.0, 1.0]), np.diag([1.0, 100.0])")
    c6 = ("two quadratics with targets t1=(1,1), t2=(2,2) and Hessians H1=diag(100,1), H2=diag(1,100) (kappa 100 each); "
          "curvature-weighted optimum vs best point of span{t1,t2} with free scale")
    need("tau = 0.7", "c = rng.normal(size=5) * 3", "q = rng.dirichlet(np.ones(5))")
    c7 = "5 random costs c~3*N(0,1), random prior q~Dirichlet(1), tau=0.7; SLSQP minimiser of lam.c + tau*KL(lam, q) on the simplex vs prior-weighted softmax"
    need("dout, din, k = 6, 10, 3", "pis = np.array([0.2, 0.3, 0.5])", "rho = 1e-1", "np.array([1, 2, 5, 10, 50])")
    c8 = ("regression merge B G^-1 with d_out=6, d_in=10, k=3 experts (200 samples each, lognormal column scales, pi=(0.2,0.3,0.5)); vs stacked least squares, "
          "identical C_i, first-order perturbation bound (1e-6 perturbations), ridge rho=0.1, and kappa(Z^T Z)=kappa(Z)^2 on a 30x5 matrix")
    need("dout, din, r = 64, 48, 8", "np.geomspace(1, 100, r)", "rtol=1e-10", "< 1e-12")
    c9 = ("LoRA factors B (64x8), A (8x48), random invertible gauge R = G1 diag(geomspace(1,100,8)) G2 with Gaussian G1, G2: product invariance, factor kappas, "
          "QR-core spectrum vs product spectrum, kappa2 of the rank-8 product, averaging factors vs products, concatenated adapter")
    need("n = 6", "np.geomspace(1, 1000, n)", "> 0.5")
    c10 = ("6x6 random W, W*, input second moment C=LL^T+0.1I and reparameterisation R=diag(geomspace(1,1000,6))Q: "
           "functional error tr((W-W*)C(W-W*)ᵀ) vs kappa(W) before/after (W->W R^-1, C->R C R^T)")
    need("n = 128", "np.geomspace(1.0, 1e-6, n)", "np.geomspace(1e-9, 1e-1, 17)", "for _ in range(20)", "np.geomspace(1.0, 0.5, n)",
         "(1e-5, 1e-4, 1e-3, 3e-3, 1e-2)", "for _ in range(5)")
    n_levels = len(J["C11"]["near_singular"])
    assert n_levels == 17
    c11 = ("n=128, W*=U diag(sigma) V^T with sigma=geomspace(1,1e-6) (near-singular) or geomspace(1,0.5) (well-conditioned), Gaussian noise of per-entry std s "
           f"in {n_levels} levels geomspace(1e-9,1e-1), 20 draws per level, median kappa; plus 5 real CLIP base maps x 5 noise levels (5 draws)")

    ns = J["C11"]["near_singular"]
    s_lo = next(r for r in ns if abs(r["s"] - 1e-6) < 1e-12)
    s_hi = next(r for r in ns if abs(r["s"] - 1e-2) < 1e-12)
    wc = J["C11"]["well_conditioned"]
    wc_lo = next(r for r in wc if abs(r["s"] - 1e-3) < 1e-12)
    wc_hi = wc[-1]
    rb = J["C11_real_base"]
    k0 = rb["encoder.layers.0.self_attn.k_proj.weight"]
    V["synth_c11"] = dict(k_s1e6=s_lo["kappa_median"], k_s1e2=s_hi["kappa_median"],
                          l0k_k0=k0["kappa_W0"], l0k_s1e2=next(r for r in k0["noise"] if abs(r["s"] - 1e-2) < 1e-12)["kappa_median"],
                          l6fc1_k0=rb["encoder.layers.6.mlp.fc1.weight"]["kappa_W0"],
                          l6fc1_s1e2=next(r for r in rb["encoder.layers.6.mlp.fc1.weight"]["noise"] if abs(r["s"] - 1e-2) < 1e-12)["kappa_median"])

    measured = {
        "C1": f"lam_opt={[round(x, 4) for x in J['C1']['lam_opt']]}; analytic risk opt {J['C1']['risk_opt']:.4f} vs uniform {J['C1']['risk_uniform']:.4f}; "
              f"Monte Carlo opt {detail('C1 MC risk matches analytic (opt)')}, uniform {detail('C1 MC risk matches analytic (uniform)')} (MC vs analytic)",
        "C2": f"risk: oracle {J['C2']['risk_oracle']:.4f}, uniform {J['C2']['risk_uniform']:.4f}, best exp(-beta*kappa) {J['C2']['risk_exp_best']:.4f} (beta={J['C2']['beta_best']:.4f}), "
              f"best kappa^-p {J['C2']['risk_pow_best']:.4f} (p={J['C2']['p_best']}), best reversed {J['C2']['risk_reversed_best']:.4f}",
        "C3": f"GLS weights {[round(x, 4) for x in J['C3']['gls']]}; simplex-QP {[round(x, 4) for x in J['C3']['simplex']]}",
        "C4": f"analytic: kappa(W1)=kappa(W2)=1/eps={1 / float(eps):g}, kappa((W1+W2)/2)=1, (W1-W1)/2=0 (no numbers stored in the JSON)",
        "C5": f"lam(high-kappa expert) under softmax(-1e-3*kappa) = {J['C5']['lam_rule_beta1e-3']:.3g}; excess loss over equal weights = {J['C5']['excess']:.4f}",
        "C6": f"theta*={[round(x, 4) for x in J['C6']['opt']]} (loss {J['C6']['loss_opt']:.4f}); best loss in span{{t1,t2}} = {J['C6']['loss_best_span']:.4f}",
        "C7": f"max abs difference SLSQP vs closed form = {detail('C7 entropy-regularised optimum = prior-weighted softmax').replace('max err', '').strip()} (not in JSON; from re-run)",
        "C8": f"first-order perturbation, norm of change <= bound: {detail('C8 eq.34 first-order perturbation bound')} (not in JSON; from re-run)",
        "C9": f"{detail('C9 factor kappas change under gauge')}; sigma_min/sigma_max of the rank-8 product: {detail('C9 ordinary kappa2 of a rank-r product is numerically infinite').split('=')[-1]} (not in JSON; from re-run)",
        "C10": f"kappa(W) {detail('C10 kappa changes under the same reparameterisation')} (not in JSON; from re-run)",
        "C11": (f"near-singular: median kappa {g(s_lo['kappa_median'])} at s=1e-6 -> {g(s_hi['kappa_median'])} at s=1e-2 (monotone decrease in [1e-6,1e-2]); "
                f"well-conditioned: {g(wc_lo['kappa_median'])} at s=1e-3 -> {g(wc_hi['kappa_median'])} at s=0.1; "
                f"real CLIP L0 k_proj: kappa0 {g(k0['kappa_W0'])} -> {g(V['synth_c11']['l0k_s1e2'])} at s=1e-2"),
    }
    crit = {
        "C1": "QP optimum == inverse variance (atol 1e-6); MC risk within 3% of analytic, opt and uniform",
        "C2": "eq.22 variance formula; eq.21 bounds on 200 designs; best decreasing rules < uniform; reversed >= uniform; counterexamples exact",
        "C3": "variance (1+rho)/2 exact; GLS has a negative weight; simplex QP clips to [1,0]; duplicate doubles group mass",
        "C4": "equal kappa; complementary average kappa=1; cancelling pair averages to 0; kappa scale invariant",
        "C5": "excess == (lam-1/2)^2(1-1e-4)^2 (atol 1e-12); rule moves weight off the high-kappa expert and loses",
        "C6": "optimum == (102/101, 201/101); best-span loss > optimum + 1e-3; both kappa(H)=100",
        "C7": "optimum == prior-weighted softmax (atol 1e-5); shift invariance and scale sensitivity",
        "C8": "all 5 sub-checks (see B1b), tolerances 1e-8..1e-9, bound x 1.01",
        "C9": "all 6 sub-checks (see B1b); factor kappa changes > 10%",
        "C10": "functional error invariant (rel 1e-8); kappa changes by > 50%",
        "C11": "near-singular: kappa decreases with s on s in [1e-6,1e-2]; well-conditioned: increases for s >= 1e-3",
    }
    rows = []
    for cid in claims:
        st = [s for s, _ in by_id[cid]]
        npass = sum(s == "PASS" for s in st)
        status = ("PASS" if all(s == "PASS" for s in st) else "FAIL") + f" ({npass}/{len(st)} sub-checks)"
        rows.append([cid, claims[cid][0], claims[cid][1], {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7, "C8": c8, "C9": c9,
                                                         "C10": c10, "C11": c11}[cid], measured[cid], crit[cid], status])
    V["synth_status"] = {r[0]: r[-1] for r in rows}
    add_table(
        "B1a", "Synthetic checks C1-C11 (math of the review on small linear systems)",
        "Not a data experiment: each check compares a numerical value with an analytic one in a small synthetic system (seed 0, CPU); the quantity measured is shown per check; no split, no accuracy; "
        "'result' = PASS/FAIL of the script's own asserts.",
        [rel(AA / "synthetic" / "synthetic_checks.json"), rel(src_path)],
        ["ID", "review ref", "claim tested (script docstring)", "setup", "measured quantity", "pass criterion", "result"], rows, align="lllllll",
        legend=["review ref = section/equation of the review the check tests (from the script docstring).",
                "The stored JSON records numbers for C1, C2, C3, C5, C6, C11 and C11_real_base and `fails: []` (no failing check). "
                "C4 and C7-C10 store no numbers; their numbers and every individual PASS line come from re-running a COPY of the script in the scratchpad "
                f"(seed 0): the re-run JSON equals the stored JSON on all shared keys (max relative deviation {max_dev:.1e}); "
                "C11_real_base cannot be re-run (needs the base-weight npz passed as argv[2]) and is taken from the stored JSON only.",
                f"Stored `fails` = {stored['fails']}; re-run console ends with 'ALL CHECKS PASS'."])
    # ---- B1b sub-check log
    rows_b = [[cid, st, txt] for cid, st, txt in parsed]
    add_table("B1b", "All individual sub-checks (console lines of the script)",
              "One row per `check()` call executed; PASS/FAIL against the script's analytic value; no split or accuracy involved.",
              [rel(src_path)],
              ["check", "status", "text printed by the script (name + detail)"], rows_b, align="lll",
              legend=["Lines come from re-running a copy of the script (seed 0) in the scratchpad (log: report_build/B_diagnostics/_work/synthetic_rerun.log); its JSON equals the stored synthetic_checks.json (B1a legend)."])
    # ---- B1c C11 near-singular curve
    rows_c = []
    for r, w in zip(J["C11"]["near_singular"], J["C11"]["well_conditioned"]):
        assert abs(r["s"] - w["s"]) < 1e-15
        rows_c.append([g(r["s"], 3), g(r["kappa_median"], 4), g(r["kappa_q10"], 4), g(r["kappa_q90"], 4), g(w["kappa_median"], 5)])
    add_table("B1c", "C11: condition number of W*+E vs weight-noise level s (synthetic 128x128)",
              "Varied: Gaussian noise std s per entry (17 levels); metric: kappa_2 of W*+E (median, 10% and 90% quantile over 20 draws), dimensionless; no split.",
              [rel(AA / "synthetic" / "synthetic_checks.json")],
              ["s", "near-singular W*: kappa median", "q10", "q90", "well-conditioned W*: kappa median"], rows_c,
              legend=["near-singular W*: singular values geomspace(1, 1e-6), kappa(W*) = 1e6; well-conditioned: geomspace(1, 0.5), kappa(W*) = 2."])
    # ---- B1d real base maps
    rows_d = []
    for name, d in J["C11_real_base"].items():
        row = [short(name), g(d["kappa_W0"], 4), g(d["sigma_min_W0"], 3)]
        for r in d["noise"]:
            row.append(g(r["kappa_median"], 4))
        rows_d.append(row)
    sl = [g(r["s"], 2) for r in next(iter(J["C11_real_base"].values()))["noise"]]
    add_table("B1d", "C11: condition number of real CLIP ViT-B/32 base maps under added weight noise",
              "Varied: Gaussian noise std s added to the base weight (5 levels, 5 draws, median); metric: kappa_2, dimensionless; base-model maps, no split.",
              [rel(AA / "synthetic" / "synthetic_checks.json")],
              ["map", "kappa(W0)", "sigma_min(W0)"] + [f"kappa median, s={s}" for s in sl], rows_d,
              legend=["map names are shortened: L<layer>.<module> (e.g. L0.k_proj = encoder.layers.0.self_attn.k_proj.weight)."])


# ==================================================================== B2
def build_B2():
    gp = AA / "grad_pref"
    summ = jload(gp / "summary.json")
    ana = jload(gp / "analysis_grad_vs_kappa.json")
    dec = jload(gp / "decompose_pref.json")
    svt = jload(AA / "score_vs_targets.json")
    V["gp_summary"], V["gp_ana"], V["gp_dec"] = summ, ana, dec
    # ---------- B2a context
    rows = []
    for t in TASKS:
        rows.append([t, pct(summ["1.0"]["task_acc"][t]), pct(summ["2.4"]["task_acc"][t]), f(summ["1.0"]["task_ce"][t], 4), f(summ["2.4"]["task_ce"][t], 4)])
    for a in ("1.0", "2.4"):
        assert abs(np.mean(list(summ[a]["task_acc"].values())) - summ[a]["avg_acc"]) < 1e-12
        assert abs(np.mean(list(summ[a]["task_ce"].values())) - summ[a]["avg_ce"]) < 1e-12
    rows.append(["**mean (avg_acc / avg_ce)**", pct(summ["1.0"]["avg_acc"]), pct(summ["2.4"]["avg_acc"]), f(summ["1.0"]["avg_ce"], 4), f(summ["2.4"]["avg_ce"], 4)])
    add_table("B2a", "Operating point of the gradient-preference diagnostic: the uniform merge at alpha = 1 and 2.4",
              "Varied: global scale alpha of the uniform (lambda = 1/8) merge W0 + alpha * mean(Delta); split: dev (20% of each official test set); "
              "metric: absolute per-task accuracy (%) and mean dev cross-entropy at CLIP logit scale 100 (nats); budget D (diagnostic).",
              [rel(gp / "summary.json")],
              ["task", "acc (%) alpha=1", "acc (%) alpha=2.4", "CE alpha=1", "CE alpha=2.4"], rows,
              legend=["Gradients d(mean dev CE)/d(lambda_i^p) are taken at these two uniform merges (script scripts/aa_grad_preference.py); "
                      "the objective is the mean over the 8 tasks of each task's mean dev CE.",
                      "avg_acc is the unweighted mean of the 8 absolute accuracies (not normalised by the expert's own accuracy)."])

    # ---------- B2b within-map correlation with the gradient preference
    order = ["kappa", "log_kappa", "kappa_layer_std", "fro_D", "smin_W", "stable_rank_D", "entropy_rank_D"]
    rows = []
    for a in ("1", "2.4"):
        for s in order:
            d = ana[a][s]
            top = d["top_maps_by_abs_dL"][0]
            rows.append([a, s, f(d["mean_spearman"], 4, True), pct(d["frac_negative"], 1), f(d["p_mean_spearman_le"], 4), f(d["p_mean_spearman_ge"], 4),
                         g(d["first_order_dL"], 4), f(d["p_dL_le"], 4), pct(d["frac_dL_from_top5"], 1), f"{short(top[0])}: {g(top[1], 3)}"])
    add_table("B2b", "Within-map Spearman between gradient preference and each score (73 linear maps, 8 experts), alpha = 1 and 2.4",
              "Varied: score (7), alpha (1, 2.4); split: dev; metric: mean over the 73 maps of the across-expert Spearman rho(pref, score) (dimensionless, +1..-1) with one-sided permutation p-values "
              f"({PERMS['grad_map']} shuffles of the scores within each map), first-order effect on dev CE of moving towards the rule, budget D target / budget A scores.",
              [rel(gp / "analysis_grad_vs_kappa.json")],
              ["alpha", "score", "mean within-map rho(pref, score)", "% maps with rho<0", "p (null <= obs)", "p (null >= obs)",
               "first-order dL", "p (null dL <= obs)", "share of total abs(dL) from top-5 maps", "largest-abs(dL) map: dL"], rows, align="llrrrrrrrl",
              legend=["pref = -(dL/dlambda centred across experts), using the tied weight+bias derivative; positive pref = moving mass to that expert lowers dev CE. "
                      "The literal rule predicts a NEGATIVE correlation with kappa.",
                      "p (null <= obs): fraction of permutation draws with mean rho <= the observed value (small = evidence in the direction the rule predicts); "
                      "p (null >= obs): the reverse direction.",
                      "first-order dL = -sum_p sum_i g_i^p (s_i^p - mean_i s^p) (g = tied gradient of the mean dev CE), the derivative of dev CE along the rule's direction up to the "
                      "positive factor beta/K; negative = the rule's direction lowers dev CE. Its unit is (CE nats) x (score unit), so only the sign, the p-value and the concentration are comparable across scores.",
                      "kappa = kappa_2(W_i); log_kappa; kappa_layer_std = within-map z-score of kappa; fro_D = ‖Δ_i‖_F; smin_W = sigma_min(W_i); stable_rank_D, entropy_rank_D = spread of Delta_i's spectrum.",
                      "map labels are shortened: L<layer>.<module>."],
              notes=["kappa, log_kappa and kappa_layer_std have identical within-map ranks, hence identical mean rho; they differ in dL and p(dL)."])

    # extras of analysis_grad_vs_kappa.json
    for k in ("spearman_kappa_vs_froD", "spearman_kappa_vs_sminW"):
        assert ana["1"][k] == ana["2.4"][k], k
    rows = []
    for a in ("1", "2.4"):
        d = ana[a]
        rows.append([a, f(d["kappa_partial_given_froD"]["mean"], 4, True), pct(d["kappa_partial_given_froD"]["frac_negative"], 1),
                     f(d["spearman_kappa_vs_froD"]["mean_all"], 4, True), f(d["spearman_kappa_vs_froD"]["mean_kappa0_gt_1e4"], 4, True),
                     f(d["spearman_kappa_vs_froD"]["mean_kappa0_le_1e4"], 4, True), str(d["spearman_kappa_vs_froD"]["n_big"]),
                     f(d["spearman_kappa_vs_sminW"], 4, True)])
    add_table("B2c", "Extra statistics of the kappa analysis: partial correlation given ‖Δ‖_F, kappa vs update norm, kappa vs sigma_min",
              "Varied: alpha (target only for the first two columns); split: dev; metric: mean (over maps) Spearman correlations, dimensionless.",
              [rel(gp / "analysis_grad_vs_kappa.json")],
              ["alpha", "partial rho(pref, kappa given ‖Δ‖_F)", "% maps negative", "rho(kappa, ‖Δ‖_F) all maps", "... maps with kappa0 > 1e4",
               "... maps with kappa0 <= 1e4", "n maps kappa0 > 1e4", "rho(kappa, sigma_min(W))"], rows, align="lrrrrrrr",
              legend=["The last 5 columns are properties of the experts' weights and do not depend on alpha (identical in both blocks of the JSON)."])
    # top-5 maps by |dL| for every score and alpha
    rows = []
    for a in ("1", "2.4"):
        for s_ in order:
            d_ = ana[a][s_]
            rows.append([a, s_, pct(d_["frac_dL_from_top5"], 1)] + [f"{short(nm)}: {g(val, 3)}" for nm, val in d_["top_maps_by_abs_dL"]])
    add_table("B2d", "Which 5 maps carry the first-order effect on dev loss, for every score",
              "Varied: score (7) and alpha (1, 2.4); split: dev; metric: the 5 maps with the largest absolute first-order dL of moving towards softmax(-beta*score) (per-map contribution, up to the factor beta/K; negative lowers dev CE) and their share of the total abs(dL) over the 73 maps; budget A rule / D target.",
              [rel(gp / "analysis_grad_vs_kappa.json")], ["alpha", "score", "share of total abs(dL) from these 5 maps (%)", "map 1: dL", "map 2: dL", "map 3: dL", "map 4: dL", "map 5: dL"], rows, align="llrlllll",
              legend=[f"For the literal score kappa these 5 maps carry {pct(ana['1']['kappa']['frac_dL_from_top5'], 2)}% (alpha=1) and {pct(ana['2.4']['kappa']['frac_dL_from_top5'], 2)}% (alpha=2.4) of the total abs(dL) over the 73 maps; "
                      "all listed kappa maps are attention q_proj/k_proj maps: " + str(all(("q_proj" in nm or "k_proj" in nm) for a in ("1", "2.4") for nm, _ in ana[a]["kappa"]["top_maps_by_abs_dL"])) + " (their kappa(W0) is in B2h / per_map_grad_vs_kappa.csv).",
                      "dL is in units of (CE nats) x (score unit), so magnitudes are comparable only within a score."])

    # ---------- B2e decomposition between-expert / within-expert
    sc_order = ["kappa_W", "fro_D", "stable_rank_D", "entropy_rank_D", "smin_W", "kplus_D_0.01"]
    rows = []
    for a in ("1", "2.4"):
        for s in sc_order:
            d = dec[a][s]
            rows.append([a, s, f(d["between_expert_corr"], 3, True), f(d["within_expert_corr"], 4, True), f(d["p_within_le"], 4), f(d["p_within_ge"], 4),
                         f(d["null_sd"], 4), f(d["mean_per_expert_spearman_over_maps"], 4, True)])
    add_table("B2e", "Expert-identity vs layer-specific parts of the score / gradient-preference association",
              "Varied: score (6), alpha (1, 2.4); split: dev; metric: Pearson correlation of within-map ranks, dimensionless: between-expert (8 points = expert identity) and within-expert "
              f"(pooled residual ranks after removing each expert's mean rank = layer-specific part) with permutation p-values ({PERMS['dec_exp']} shuffles of maps inside each expert column).",
              [rel(gp / "decompose_pref.json")],
              ["alpha", "score", "between-expert corr (identity)", "within-expert corr (layer-specific)", "p (null <= obs)", "p (null >= obs)", "null sd", "mean per-expert Spearman over maps"], rows, align="llrrrrrr",
              legend=["Ranks are taken across the 8 experts inside each map; 'between-expert' correlates each expert's mean preference rank with its mean score rank "
                      "(8 points); 'within-expert' correlates the residual ranks after removing those expert means.",
                      "p (null <= obs) / (null >= obs): one-sided, null = scores permuted over maps independently inside each expert column (keeps each expert's score distribution). "
                      f"Two-sided p-values for the same quantity from a separate {PERMS['svt_exp']}-shuffle null are in B2g.",
                      "mean per-expert Spearman over maps: for each expert Spearman(pref / max abs(pref), score) over the 73 maps, averaged over the 8 experts."],
              )

    # expert-level tables
    rows = []
    for t in TASKS:
        r = [t, f(dec["1"]["expert_mean_pref_rank"][t], 3), f(dec["2.4"]["expert_mean_pref_rank"][t], 3)]
        for s in sc_order:
            assert dec["1"][s]["expert_mean_score_rank"] == dec["2.4"][s]["expert_mean_score_rank"]
            r.append(f(dec["1"][s]["expert_mean_score_rank"][t], 3))
        rows.append(r)
    add_table("B2f", "Expert identity: mean within-map rank of each expert (gradient preference and scores)",
              "Varied: expert; split: dev; metric: mean over the 73 linear maps of the expert's rank among the 8 experts (1 = lowest value, 8 = highest), dimensionless; "
              "preference rank high = the dev loss most wants more weight on this expert.",
              [rel(gp / "decompose_pref.json")],
              ["expert", "pref rank alpha=1", "pref rank alpha=2.4"] + [f"{s} rank" for s in sc_order], rows, align="lrrrrrrrr",
              legend=["Score ranks do not depend on alpha (asserted equal in both blocks of the JSON)."])

    # ---------- B2g gradient preference vs EVERY score (score_vs_targets.json)
    scores = list(svt["gradpref_a1"].keys())
    rows = []
    for s in scores:
        r = [s]
        for tn in ("gradpref_a1", "gradpref_a2.4"):
            d = svt[tn][s]
            r += [pm(d["within_map"], d["p_two"]), pm(d["within_exp"], d["p_within_two"])]
        rows.append(r)
    add_table("B2g", "Gradient preference vs every score (A: budget A weights-only scores, B: budget B data-supported scores), alpha = 1 and 2.4",
              "Varied: score (14), alpha (1, 2.4); split: dev; metric: within-map Spearman (mean over 73 maps) and within-expert (layer-specific) rank correlation, both dimensionless, "
              f"with two-sided permutation p-values in brackets ({PERMS['svt_map']} and {PERMS['svt_exp']} shuffles, seed 0).",
              [rel(AA / "score_vs_targets.json"), rel(AA / "score_vs_targets.md")],
              ["score", "alpha=1: within-map rho (p)", "alpha=1: within-expert (p)", "alpha=2.4: within-map rho (p)", "alpha=2.4: within-expert (p)"], rows, align="lllll",
              legend=["gradpref_a1 / gradpref_a2.4 = -(centred tied gradient of mean dev CE at the uniform merge with alpha 1 / 2.4).",
                      "A: scores computed from weights only (kappa_W, smin_W, fro_D = ‖Δ‖_F, stable_rank_D, entropy_rank_D, kplus_D_0.01 = kappa_plus(Delta; tau=0.01), logkappa_rel = log kappa(W_i) - log kappa(W0)). "
                      "B: pooled unlabeled calibration inputs (second moment C): fnorm_D = tr(Delta C Delta^T), rel_fnorm_D, kappa_WC = kappa_2(W C^1/2), dlogkappa_WC, srank_DC, erank_DC, kplus_DC (see B7).",
                      "Attention/MLP split, between-expert correlations and all other targets are in B4a (same file, 56 rows)."])

    # ---------- B2h per-map summary by module type
    pmap = pd.read_csv(gp / "per_map_grad_vs_kappa.csv")
    pmap["mt"] = pmap["param"].map(mtype)
    pmap["grp"] = pmap["param"].map(mgroup)
    rows = []
    order_m = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"]
    for a in sorted(pmap["alpha"].unique()):
        sub = pmap[pmap["alpha"] == a]
        groups = [(m, sub[sub.mt == m]) for m in order_m] + [("attention (q,k,v,out)", sub[sub.grp == "attention"]), ("mlp (fc1,fc2)", sub[sub.grp == "mlp"]), ("all 73 maps", sub)]
        for name, s in groups:
            agree = int((s.argmin_kappa == s.argmax_pref).sum())
            rows.append([f(a, 1), name, len(s), f(s.rho_pref_kappa.mean(), 3, True), f(s.rho_pref_froD.mean(), 3, True), f(s.rho_kappa_froD.mean(), 3, True),
                         f"{agree}/{len(s)}", kfmt(s.kappa0.median())])
    V["per_map"] = pmap
    add_table("B2h", "Per-map summary of gradient preference vs kappa by module type (full 146-row table is the CSV)",
              "Varied: alpha (1, 2.4) and module type; split: dev; metric: mean over maps of the across-expert Spearman (dimensionless); agreement = number of maps whose argmin-kappa expert is the max-preference expert (chance = 1/8 = 12.5%).",
              [rel(gp / "per_map_grad_vs_kappa.csv")],
              ["alpha", "module", "n maps", "mean rho(pref, kappa)", "mean rho(pref, ‖Δ‖_F)", "mean rho(kappa, ‖Δ‖_F)", "argmin-kappa == argmax-pref", "median kappa(W0)"], rows, align="llrrrrrr",
              legend=["Full data: results/aa/grad_pref/per_map_grad_vs_kappa.csv (one row per alpha x map: kappa0, kappa_min, kappa_max, the three rank correlations, argmin-kappa expert, argmax-preference expert).",
                      "The overall mean rho(pref, kappa) over the 73 maps reproduces B2b (0.0453); the module-level means are not independent samples (each is over 12 maps, 1 for the patch embedding)."])

    # ---------- B2i per-expert aggregated preference + task x expert derivative matrix
    glin = {}
    for a in ("1", "2.4"):
        df = pd.read_csv(gp / f"grad_pref_alpha{float(a):g}.csv")
        glin[a] = df
    rows = []
    for t in TASKS:
        r = [t]
        for a in ("1", "2.4"):
            df = glin[a]
            lin = df[df.is_linear_map == 1]
            pv = lin.pivot(index="param", columns="expert", values="pref_tied").loc[list(dict.fromkeys(lin.param)), TASKS]
            r.append(g(pv[t].sum(), 4))
            r.append(int((pv.values.argmax(1) == TASKS.index(t)).sum()))
            # cross-check with decompose_pref mean ranks
            ranks = np.apply_along_axis(rankdata, 1, pv.values)
            assert abs(ranks.mean(0)[TASKS.index(t)] - dec[a]["expert_mean_pref_rank"][t]) < 1e-3, (t, a)
        rows.append(r)
    add_table("B2i", "Which expert the dev loss prefers, aggregated over the 73 linear maps",
              "Varied: expert and alpha; split: dev; metric: sum over the 73 linear maps of the centred preference pref_tied (CE nats per unit lambda) and number of maps in which the expert has the largest preference (of 73).",
              [rel(gp / "grad_pref_alpha1.csv"), rel(gp / "grad_pref_alpha2.4.csv")],
              ["expert", "sum of pref_tied, alpha=1", "n maps argmax-pref, alpha=1", "sum of pref_tied, alpha=2.4", "n maps argmax-pref, alpha=2.4"], rows, align="lrrrr",
              legend=["pref_tied = -(grad_tied - mean over experts of grad_tied) per linear map, grad_tied = d(mean dev CE)/d(lambda_i^p) including the tied bias. The mean pref rank recomputed from these CSVs equals decompose_pref.json (asserted to 1e-3).",
                      "Full data (1592 rows per alpha: param, layer, module, is_linear_map, expert, grad, grad_centered, pref, grad_tied, grad_tied_centered, pref_tied, delta_fro, grad_own_task, numel) is in the two CSVs; "
                      "the non-linear parameters (LayerNorm, class/position embeddings; is_linear_map = 0) have derivatives there too but get the uniform prior in the rule, so they are not used in any table."])

    z = np.load(gp / "grad_pref_pertask.npz")
    assert list(z["tasks"]) == TASKS

    def owner(n):
        if n.endswith(".bias") and "layer_norm" not in n and "layrnorm" not in n and "layernorm" not in n:
            return n[:-5] + ".weight"
        return None

    for a, tag in (("1", "a1"), ("2.4", "a2.4")):
        df = glin[a]
        lin = list(dict.fromkeys(df[df.is_linear_map == 1].param))
        comp = {p: z[f"{tag}|{p}"].copy() for p in lin}
        for k in z.files:
            if k == "tasks" or not k.startswith(tag + "|"):
                continue
            o = owner(k.split("|", 1)[1])
            if o in comp:
                comp[o] = comp[o] + z[k]
        gt = df[df.is_linear_map == 1].pivot(index="param", columns="expert", values="grad_tied").loc[lin, TASKS]
        chk = np.abs(np.stack([comp[p].mean(0) for p in lin]) - gt.values).max()
        assert chk < 1e-12, chk
        M = sum(comp[p] for p in lin)
        V[f"task_expert_M_{a}"] = M
    rows = []
    for a in ("1", "2.4"):
        M = V[f"task_expert_M_{a}"]
        for ti, t in enumerate(TASKS):
            own = M[ti, ti]
            rank_own = int((M[ti] < own).sum()) + 1
            rows.append([a, t] + [g(M[ti, i], 3) for i in range(8)] + [rank_own])
    add_table("B2j", "Task-resolved loss sensitivity: derivative of each task's dev CE when expert i's weight is raised in all 73 linear maps",
              "Varied: alpha (1, 2.4), task t (row) and expert i (column); split: dev; metric: d CE_t / d lambda_i summed over the 73 linear maps (tied biases) at the uniform merge, CE nats per unit lambda "
              "(negative = raising that expert lowers the task's CE); budget D diagnostic.",
              [rel(gp / "grad_pref_pertask.npz"), rel(gp / "grad_pref_alpha1.csv"), rel(gp / "grad_pref_alpha2.4.csv")],
              ["alpha", "task (row)"] + [f"d/dλ({t})" for t in TASKS] + ["rank of own expert (1 = most negative)"], rows, align="ll" + "r" * 9,
              legend=["Entry = sum_p alpha * <grad_W CE_t, Delta_i^p> (components stored in the npz, bias gradients added to their weight map). "
                      "Their mean over tasks reproduces grad_tied in grad_pref_alpha*.csv (asserted to 1e-12).",
                      f"Own-task entry (diagonal) is the most negative of its row in {sum(1 for r in rows if r[-1] == 1)} of {len(rows)} (alpha, task) rows (last column = rank of the own expert among the 8 derivatives, 1 = most negative)."])




# ==================================================================== B3
def ext_uniform_refs() -> dict:
    """Uniform-merge TEST accuracy is not stored in the oracle files: read it from the frozen-test files of master / this track."""
    out = {}
    rows = jsonl(WT / "results" / "freeze" / "test.jsonl")
    r = next(r for r in rows if r["cfg"]["tag"] == "uniform|tuned")
    m = r["metrics"]
    assert abs(np.mean(list(m["acc"].values())) - m["mean_acc"]) < 1e-12
    out["full-FT"] = dict(alpha=r["cfg"]["alpha"], acc=m["acc"], mean_acc=m["mean_acc"], mean_norm=m["mean_norm"],
                          dev_metric_norm=r["dev_metric"], src="results/freeze/test.jsonl (cfg.tag = uniform|tuned)")
    rows = jsonl(AA / "lora" / "test_freeze.jsonl")
    r = next(r for r in rows if r["label"] == "uniform")
    assert abs(np.mean(list(r["acc"].values())) - r["avg_acc"]) < 1e-12
    out["LoRA"] = dict(alpha=r["alpha"], acc=r["acc"], mean_acc=r["avg_acc"], mean_norm=None, src="results/aa/lora/test_freeze.jsonl (label = uniform)")
    return out


BANK_DIRS = (("full-FT", "oracle_lambda"), ("LoRA", "oracle_lambda_lora"))
VARIANTS = ("global_alpha", "layerwise_a1", "layerwise_alpha")


def build_B3():
    ext = ext_uniform_refs()
    V["ext_uniform"] = ext
    fits, grids, lam = OrderedDict(), {}, {}
    for bank, d in BANK_DIRS:
        grids[bank] = jload(AA / d / "uniform_alpha_grid.json")
        for variant in VARIANTS:
            for seed in (0, 1):
                p = AA / d / f"{variant}_seed{seed}.json"
                if p.exists():
                    fits[(bank, variant, seed)] = jload(p)
                    lam[(bank, variant, seed)] = load_lambda(AA / d / f"{variant}_seed{seed}_lambda.csv")
    uni = {}
    for bank in grids:
        a_best = max(grids[bank], key=lambda a: grids[bank][a]["avg_acc"])
        uni[bank] = dict(alpha=float(a_best), dev=grids[bank][a_best]["avg_acc"], dev_acc=grids[bank][a_best]["acc"], dev_ce=grids[bank][a_best]["avg_ce"])
        assert abs(uni[bank]["alpha"] - ext[bank]["alpha"]) < 1e-9, (bank, uni[bank]["alpha"], ext[bank]["alpha"])
    V["uni"], V["fits"], V["lam"], V["grids"] = uni, fits, lam, grids
    osrc = (WT / "scripts" / "aa_oracle_lambda.py").read_text()
    lr_alpha = float(re.search(r'"--lr-alpha", type=float, default=([0-9.]+)', osrc).group(1))
    for k_, r_ in fits.items():
        assert r_["steps"] == 400 and r_["per_task_batch"] == 128 and abs(r_["lr"] - 0.1) < 1e-12, k_
    grid_rng = {b_: f"{min(float(a) for a in grids[b_]):g}..{max(float(a) for a in grids[b_]):g}" for b_ in grids}

    # ---------------- B3a headline
    rows = []
    for bank in grids:
        u = uni[bank]
        rows.append([bank, "uniform, best alpha on dev (grid)", "–", "–", f(u["alpha"], 2), pct(u["dev"]), pct(ext[bank]["mean_acc"]) + " †", "–", "0.00", "0.00", f(u["dev_ce"], 4), "–", "–"])
        for variant in VARIANTS:
            for seed in (0, 1):
                k = (bank, variant, seed)
                if k not in fits:
                    continue
                r = fits[k]
                dev, test = r["dev"]["avg_acc"], r["test"]["avg_acc"]
                rows.append([bank, variant, seed, f(r["init_alpha"], 2), f(r["alpha"], 4), pct(dev), pct(test), f"{100 * (dev - test):+.2f}",
                             f"{100 * (dev - u['dev']):+.2f}", f"{100 * (test - ext[bank]['mean_acc']):+.2f}", f(r["dev"]["avg_ce"], 4), f(r["test"]["avg_ce"], 4),
                             f"{r['curve'][0]['loss']:.3f} -> {r['curve'][-1]['loss']:.3f}"])
    add_table("B3a", "Oracle weights fitted with dev labels: headline accuracy, full fine-tuning and LoRA banks",
              "Varied: oracle family (one lambda per expert + alpha; per-map lambda at alpha = 1; per-map lambda + alpha), seed (0, 1; LoRA seed 0 only); fit on dev labels (budget D, oracle/diagnostic only); "
              "evaluation on the full dev split and on the untouched test split (80%); metric: absolute accuracy (%, unweighted mean of the 8 per-task accuracies, NOT normalised), CE in nats at logit scale 100.",
              [rel(AA / "oracle_lambda" / "uniform_alpha_grid.json"), rel(AA / "oracle_lambda_lora" / "uniform_alpha_grid.json")]
              + [rel(AA / d / f"{v}_seed{sd}.json") for _, d in BANK_DIRS for v in VARIANTS for sd in (0, 1) if (AA / d / f"{v}_seed{sd}.json").exists()]
              + [ext["full-FT"]["src"], ext["LoRA"]["src"]],
              ["bank", "fit", "seed", "init alpha", "fitted alpha", "dev acc (%)", "test acc (%)", "dev - test (pt)", "gain vs uniform, dev (pt)", "gain vs uniform, test (pt)†", "dev CE", "test CE", "minibatch loss, step 0 -> 399"],
              rows, align="llrrrrrrrrrrr",
              legend=["global_alpha: one softmax lambda per expert shared by every linear map + a learned global alpha (8 + 1 free parameters). layerwise_a1: a per-map softmax lambda (73 maps x 8 for full-FT, 48 x 8 for LoRA) with alpha fixed at 1 "
                      "(the literal normalised-average form). layerwise_alpha: per-map lambda + learned alpha. Adam, 400 steps, lr 0.1 (alpha: lr " + f"{lr_alpha:g}" + " from the script default), 128 images per task and step, logit scale 100 (steps, lr and batch asserted equal in all JSONs); init alpha = dev-best uniform alpha when alpha is learned.",
                      "'uniform, best alpha on dev (grid)': lambda = 1/8, alpha chosen on the dev grid of uniform_alpha_grid.json (full-FT grid " + grid_rng["full-FT"] + ", LoRA grid " + grid_rng["LoRA"] + ").",
                      f"† The test accuracy of tuned uniform is not stored in the oracle files; it is read from {ext['full-FT']['src']} (alpha {ext['full-FT']['alpha']:g}; absolute {pct(ext['full-FT']['mean_acc'])}, "
                      f"normalised {pct(ext['full-FT']['mean_norm'])}) and {ext['LoRA']['src']} (alpha {ext['LoRA']['alpha']:g}; absolute {pct(ext['LoRA']['mean_acc'])}); the alphas equal the dev-best uniform alphas of the grids (asserted).",
                      "gain = fit minus uniform on the same split, in accuracy points (pt). The oracle lambdas are fitted on dev, so the dev numbers are in-sample; the test numbers are out-of-sample for the lambdas.",
                      "minibatch loss = mean-over-tasks CE of the sampled minibatch at the first and last logged step (noisy; not the dev CE)."])

    # ---------------- B3b per-task accuracy
    rows = []
    for bank in grids:
        u = uni[bank]
        rows.append([bank, "uniform (best alpha)", "–", "dev"] + [pct(u["dev_acc"][t]) for t in TASKS] + [pct(u["dev"])])
        rows.append([bank, "uniform (best alpha) †", "–", "test"] + [pct(ext[bank]["acc"][t]) for t in TASKS] + [pct(ext[bank]["mean_acc"])])
        for variant in VARIANTS:
            for seed in (0, 1):
                k = (bank, variant, seed)
                if k not in fits:
                    continue
                for split in ("dev", "test"):
                    a = fits[k][split]["acc"]
                    assert abs(np.mean(list(a.values())) - fits[k][split]["avg_acc"]) < 1e-12
                    rows.append([bank, variant, seed, split] + [pct(a[t]) for t in TASKS] + [pct(fits[k][split]["avg_acc"])])
    add_table("B3b", "Oracle weights: per-task accuracy on dev and test",
              "Varied: oracle family and seed as in B3a, per task; splits dev (in-sample for the fit) and test (80% held out); metric: absolute per-task accuracy (%); last column = unweighted mean of the 8.",
              [rel(AA / d / f"{v}_seed{sd}.json") for _, d in BANK_DIRS for v in VARIANTS for sd in (0, 1) if (AA / d / f"{v}_seed{sd}.json").exists()] + [ext["full-FT"]["src"], ext["LoRA"]["src"]],
              ["bank", "fit", "seed", "split"] + TASKS + ["mean"], rows, align="llll" + "r" * 9,
              legend=["† rows: tuned-uniform test accuracy from the frozen-test files (see B3a)."])

    # ---------------- B3c uniform alpha grids
    rows = []
    for bank in grids:
        for a, r in grids[bank].items():
            rows.append([bank, f(float(a), 2), pct(r["avg_acc"]), f(r["avg_ce"], 4)] + [pct(r["acc"][t]) for t in TASKS] + ["best" if a == str(uni[bank]["alpha"]) or float(a) == uni[bank]["alpha"] else ""])
    add_table("B3c", "Uniform-lambda merge: dev accuracy vs the global scale alpha",
              "Varied: alpha of the uniform merge W0 + alpha * mean(Delta); split: dev; metric: absolute accuracy (%), mean dev CE (nats, logit scale 100).",
              [rel(AA / "oracle_lambda" / "uniform_alpha_grid.json"), rel(AA / "oracle_lambda_lora" / "uniform_alpha_grid.json")],
              ["bank", "alpha", "avg acc (%)", "avg CE"] + TASKS + ["dev-best"], rows, align="lrrr" + "r" * 8 + "l")

    # ---------------- B3d seed agreement
    ana = jload(AA / "oracle_lambda" / "analysis_oracle_vs_kappa.json")
    V["oracle_ana"] = ana
    rows = []
    for variant in ("layerwise_alpha", "layerwise_a1"):
        a = lam[("full-FT", variant, 0)].values
        b = lam[("full-FT", variant, 1)].values
        sp_all = spearmanr(np.log(a).ravel(), np.log(b).ravel()).statistic
        ra, rb = np.apply_along_axis(rankdata, 1, np.log(a)), np.apply_along_axis(rankdata, 1, np.log(b))
        ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
        within = float(np.mean(np.sum(ra * rb, 1) / np.sqrt(np.sum(ra ** 2, 1) * np.sum(rb ** 2, 1))))
        j = ana["seed_agreement"][variant]
        assert abs(sp_all - j["spearman_loglambda_all"]) < 1e-9 and abs(within - j["mean_within_map_spearman"]) < 1e-9
        names = list(lam[("full-FT", variant, 0)].index)
        att = np.array([mgroup(n) == "attention" for n in names])
        mlp = np.array([mgroup(n) == "mlp" for n in names])
        am = a.argmax(1) == b.argmax(1)
        rows.append([variant, f(sp_all, 4), f(j["spearman_loglambda_all"], 4), f(within, 4), f(j["mean_within_map_spearman"], 4),
                     f"{am[att].sum()}/{att.sum()}", f"{am[mlp].sum()}/{mlp.sum()}", f"{am.sum()}/{len(am)}",
                     f(fits[("full-FT", variant, 0)]["alpha"], 4), f(fits[("full-FT", variant, 1)]["alpha"], 4)])
        V.setdefault("seed_agree", {})[variant] = dict(sp_all=sp_all, within=within, att=int(am[att].sum()), n_att=int(att.sum()), mlp=int(am[mlp].sum()), n_mlp=int(mlp.sum()))
    for variant in ("global_alpha",):
        a = lam[("full-FT", variant, 0)].values[0]
        b = lam[("full-FT", variant, 1)].values[0]
        rows.append([variant + " (8 expert weights)", f(spearmanr(a, b).statistic, 4), "n/a (not in JSON)", "n/a", "n/a", "–", "–", f"argmax {TASKS[a.argmax()]} vs {TASKS[b.argmax()]}",
                     f(fits[("full-FT", variant, 0)]["alpha"], 4), f(fits[("full-FT", variant, 1)]["alpha"], 4)])
    add_table("B3d", "Oracle seed agreement (full fine-tuning; two independent fits per family)",
              "Varied: seed (0 vs 1) of the Adam fit for each family; split: dev (fit); metric: Spearman correlation (dimensionless) of the fitted log lambda over all (map, expert) pairs, mean within-map Spearman, and fraction of maps with the same top expert.",
              [rel(AA / "oracle_lambda" / "analysis_oracle_vs_kappa.json"), rel(AA / "oracle_lambda" / "layerwise_alpha_seed0_lambda.csv"), "... (all *_lambda.csv of oracle_lambda)"],
              ["family", "Spearman log-lambda, all (map,expert) [recomputed from CSV]", "same, from analysis JSON", "mean within-map Spearman [recomputed]", "same, from analysis JSON",
               "same argmax expert: attention maps", "same argmax: MLP maps", "same argmax: all 73 maps", "alpha seed 0", "alpha seed 1"], rows, align="lrrrrrrrrr",
              legend=["The recomputation from the CSVs equals the values in analysis_oracle_vs_kappa.json (asserted to 1e-9). The JSON has no entry for global_alpha (see B3h legend for the file times): for global_alpha the first column is the Spearman correlation of the 8 expert weights of seed 0 vs seed 1.",
                      "LoRA oracles were fitted with seed 0 only."])

    # ---------------- B3e concentration (n_eff) by map type
    cols = ["q_proj", "k_proj", "v_proj", "out_proj", "attn", "fc1", "fc2", "mlp", "patch", "all"]
    names_ff = list(lam[("full-FT", "layerwise_alpha", 0)].index)
    cnt_ = {m_: sum(1 for n_ in names_ff if mtype(n_) == m_) for m_ in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding")}
    assert len({v_ for k_, v_ in cnt_.items() if k_ != "patch_embedding"}) == 1 and cnt_["patch_embedding"] == 1
    n_per_type = cnt_["q_proj"]
    n_attn_mlp = len(names_ff) - 1
    rows, rows2 = [], []
    V["neff"] = {}
    for k, L in lam.items():
        names = list(L.index)
        ne = neff_rows(L.values)
        mx = L.values.max(1)
        mt = np.array([mtype(n) for n in names])
        gp_ = np.array([mgroup(n) for n in names])

        def mean_of(mask):
            return float(ne[mask].mean()) if mask.any() else float("nan")

        d = {c: mean_of(mt == c) for c in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")}
        d.update(attn=mean_of(gp_ == "attention"), mlp=mean_of(gp_ == "mlp"), patch=mean_of(gp_ == "patch_embedding"), all=float(ne.mean()))
        V["neff"][k] = dict(d, attn_min=float(ne[gp_ == "attention"].min()), attn_max=float(ne[gp_ == "attention"].max()),
                            mlp_min=float(ne[gp_ == "mlp"].min()) if (gp_ == "mlp").any() else float("nan"),
                            mlp_max=float(ne[gp_ == "mlp"].max()) if (gp_ == "mlp").any() else float("nan"),
                            maxlam_attn=float(mx[gp_ == "attention"].mean()), maxlam_mlp=float(mx[gp_ == "mlp"].mean()) if (gp_ == "mlp").any() else float("nan"), maxlam_all=float(mx.mean()))
        if k[0] == "full-FT" and f"{k[1]}_seed{k[2]}" in ana:
            j = ana[f"{k[1]}_seed{k[2]}"]
            assert abs(j["n_eff_mean"] - d["all"]) < 1e-9 and abs(j["max_lambda_mean"] - V["neff"][k]["maxlam_all"]) < 1e-9
        rows.append([k[0], k[1], k[2]] + [f(d[c], 2) for c in cols])
        e = V["neff"][k]
        rows2.append([k[0], k[1], k[2], f"{f(e['attn_min'], 2)} .. {f(e['attn_max'], 2)}", f"{f(e['mlp_min'], 2)} .. {f(e['mlp_max'], 2)}" if (gp_ == "mlp").any() else "–",
                      f(e["maxlam_attn"], 3), f(e["maxlam_mlp"], 3) if (gp_ == "mlp").any() else "–", f(e["maxlam_all"], 3),
                      f"{int((ne[gp_ == 'attention'] < 1.5).sum())}/{int((gp_ == 'attention').sum())}"])
    srcs = [rel(AA / d / f"{v}_seed{sd}_lambda.csv") for _, d in BANK_DIRS for v in VARIANTS for sd in (0, 1) if (AA / d / f"{v}_seed{sd}_lambda.csv").exists()]
    add_table("B3e", "Oracle concentration: effective number of experts n_eff = exp(entropy of lambda) per map, by map type",
              "Varied: oracle family, seed and map type; split: dev (fit); metric: mean over the maps of n_eff (1 = one expert owns the map, 8 = uniform), dimensionless; computed here from the *_lambda.csv files.",
              srcs, ["bank", "fit", "seed", "q_proj", "k_proj", "v_proj", "out_proj", "attention (all)", "fc1", "fc2", "MLP (all)", "patch emb. (1 map)", "all maps"], rows, align="llr" + "r" * 10,
              legend=["n_eff = exp(-sum_i lambda_i log lambda_i) with lambda floored at 1e-12 per map. The 'all maps' mean equals n_eff_mean and max_lambda_mean of analysis_oracle_vs_kappa.json (asserted to 1e-9 for the 4 layerwise full-FT fits).",
                      f"full-FT: {n_per_type} maps per type ({n_attn_mlp} attention/MLP weights) + the patch embedding; LoRA: only the 48 adapted attention maps (no MLP/patch entries). global_alpha has the same lambda on every map, so its n_eff is the n_eff of the 8 expert weights."])
    add_table("B3f", "Oracle concentration: spread of n_eff and size of the largest weight",
              "Varied: oracle family, seed, attention vs MLP maps; split: dev (fit); metric: min..max of n_eff over maps, mean over maps of max_i lambda_i (dimensionless), count of attention maps with n_eff < 1.5.",
              srcs, ["bank", "fit", "seed", "n_eff range, attention maps", "n_eff range, MLP maps", "mean max lambda, attention", "mean max lambda, MLP", "mean max lambda, all maps", "attention maps with n_eff < 1.5"],
              rows2, align="llr" + "r" * 6)

    # ---------------- B3g expert-level: mean lambda and ownership
    rows, rows_o = [], []
    for k, L in lam.items():
        names = list(L.index)
        gp_ = np.array([mgroup(n) for n in names])
        rows.append([k[0], k[1], k[2]] + [f(L[t].mean(), 4) for t in TASKS])
        if k[0] == "full-FT" and f"{k[1]}_seed{k[2]}" in ana:
            assert [round(float(L[t].mean()), 4) for t in TASKS] == [ana[f"{k[1]}_seed{k[2]}"]["expert_mean_lambda"][t] for t in TASKS], k
        if k[1] == "global_alpha":
            continue
        for grp in ("attention", "mlp"):
            m = gp_ == grp
            if not m.any():
                continue
            own = np.bincount(L.values[m].argmax(1), minlength=8)
            rows_o.append([k[0], k[1], k[2], grp] + [int(x) for x in own] + [int(m.sum())])
    add_table("B3g", "Oracle weights per expert: mean lambda over maps, and how many maps each expert 'owns' (largest lambda)",
              "Varied: oracle family, seed, expert; split: dev (fit); metric: mean over the maps of lambda_i (rows sum to 1) -- first table; number of maps in which expert i has the largest lambda -- second table.",
              srcs, ["bank", "fit", "seed"] + TASKS, rows, align="llr" + "r" * 8,
              legend=["mean lambda for layerwise fits equals expert_mean_lambda of analysis_oracle_vs_kappa.json (4 decimals) where that file has the fit; global_alpha rows are the 8 fitted expert weights themselves."])
    add_table("B3g2", "Oracle ownership: number of attention / MLP maps where each expert has the largest lambda",
              "Varied: oracle family, seed, map group, expert; split: dev (fit); metric: count of maps (attention: 48, MLP: 24).",
              srcs, ["bank", "fit", "seed", "maps"] + TASKS + ["n maps"], rows_o, align="llrl" + "r" * 9)

    # ---------------- B3h oracle vs scores (full-FT, from analysis JSON)
    sc = ["kappa_W", "smin_W", "fro_D", "stable_rank_D", "entropy_rank_D"]
    import time as _time
    mt_ = {n: _time.strftime("%H:%M", _time.localtime((AA / "oracle_lambda" / n).stat().st_mtime))
           for n in ("analysis_oracle_vs_kappa.json", "global_alpha_seed0.json", "global_alpha_seed1.json")}
    mtime_note = (f"file modification times (local, same day): analysis_oracle_vs_kappa.json {mt_['analysis_oracle_vs_kappa.json']}, "
                  f"global_alpha_seed0.json {mt_['global_alpha_seed0.json']}, global_alpha_seed1.json {mt_['global_alpha_seed1.json']}.")
    rows = []
    for tag in ("layerwise_alpha_seed0", "layerwise_alpha_seed1", "layerwise_a1_seed0", "layerwise_a1_seed1"):
        j = ana[tag]
        for s_ in sc:
            d = j[s_]
            rows.append([tag, s_, f(d["mean_within_map_spearman"], 4, True), f(d["p_le"], 4), f(d["p_ge"], 4), f(d["attn_mean"], 3, True), f(d["mlp_mean"], 3, True),
                         f(d["between_expert"], 3, True), f(d["within_expert"], 4, True), f(d["p_within_le"], 4), f(d["p_within_ge"], 4)])
    add_table("B3h", "Oracle lambda vs kappa and other scores, full fine-tuning (4 layerwise fits)",
              "Varied: oracle fit (family, seed) and score (5); split: dev (fit); metric: Spearman across the 8 experts inside each map, averaged over the 73 maps (attention / MLP split too), between-expert and within-expert (layer-specific) rank correlation; "
              f"dimensionless; one-sided permutation p-values ({PERMS['orc_map']} shuffles within maps; {PERMS['orc_exp']} for the within-expert null).",
              [rel(AA / "oracle_lambda" / "analysis_oracle_vs_kappa.json")],
              ["oracle fit", "score", "within-map rho(lambda, score)", "p (null <= obs)", "p (null >= obs)", "attention maps", "MLP maps", "between-expert", "within-expert", "p (null <= obs)", "p (null >= obs)"],
              rows, align="ll" + "r" * 9,
              legend=["The literal rule predicts a NEGATIVE correlation between lambda and kappa_W (higher kappa -> lower weight).",
                      "The JSON contains only the four layerwise fits (no global_alpha entry); " + mtime_note + " The global fits are summarised in B3g and B3d."])

    # ---------------- B3i LoRA oracle vs scores
    lj = jload(AA / "lora" / "oracle_vs_scores.json")
    V["lora_oracle_ana"] = lj
    rows = []
    for tag, j in lj.items():
        for s_, d in j.items():
            if not isinstance(d, dict) or "within_map" not in d:
                continue
            rows.append([tag, s_, pm(d["within_map"], d["p"]), f(d["between"], 3, True), pm(d["within_exp"], d["p_within"])])
        assert abs(j["n_eff"] - float(neff_rows(lam[("LoRA", tag.rsplit("_seed", 1)[0], int(tag[-1]))].values).mean())) < 1e-9
    add_table("B3i", "Oracle lambda vs kappa and other scores, LoRA bank (48 adapted attention maps)",
              "Varied: oracle fit (layerwise_a1 / layerwise_alpha, seed 0) and score (7); split: dev (fit); metric: within-map Spearman (mean over 48 maps), between-expert (8 points) and within-expert rank correlation, dimensionless, two-sided permutation p-values in brackets.",
              [rel(AA / "lora" / "oracle_vs_scores.json"), rel(AA / "lora" / "oracle_vs_scores.md"), rel(AA / "oracle_lambda_lora" / "layerwise_a1_seed0.json")],
              ["oracle fit", "score", "within-map rho (p)", "between-expert", "within-expert (p)"], rows, align="llrrr",
              legend=["This file lives in results/aa/lora/ (outside the oracle_lambda_lora directory) but is the analysis of the oracle_lambda_lora fits; its n_eff equals the one recomputed from the CSV (asserted).",
                      "kappa_A(gauge-dep) = condition number of the LoRA factor A, gauge dependent and therefore not a valid score; kappa_W = kappa_2(W0 + B A); kplus_D_0.01 = kappa_plus(Delta; tau = 0.01).",
                      "p-values here are two-sided (|null| >= |obs|), unlike the one-sided p-values of B3h."])


# ==================================================================== B4
def build_B4():
    svt = jload(AA / "score_vs_targets.json")
    V["svt"] = svt
    rows, lines = [], []
    for tn, d in svt.items():
        for sn, v in d.items():
            rows.append([tn, sn, f(v["within_map"], 4, True), f(v["p_two"], 3), f(v["attn"], 3, True), f(v["mlp"], 3, True), f(v["between"], 3, True),
                         f(v["within_exp"], 4, True), f(v["p_within_two"], 3)])
            lines.append(f"| {tn} | {sn} | {v['within_map']:+.3f} ({v['p_two']:.3f}) | {v['attn']:+.3f} | {v['mlp']:+.3f} | "
                         f"{v['between']:+.2f} | {v['within_exp']:+.3f} ({v['p_within_two']:.3f}) |")
    md_lines = (AA / "score_vs_targets.md").read_text().splitlines()[2:]
    V["svt_md_ok"] = (md_lines == lines)
    add_table("B4a", "Score vs targets: every row of score_vs_targets.json (4 targets x 14 scores)",
              "Varied: target (gradient preference at alpha 1 and 2.4; oracle layerwise lambda with learned alpha / with alpha = 1, seed 0) and score (14); split: dev; metric: rank correlations across the 8 experts (dimensionless), "
              f"two-sided permutation p-values ({PERMS['svt_map']} shuffles within maps for the within-map mean; {PERMS['svt_exp']} shuffles inside expert columns for the within-expert correlation, seed 0).",
              [rel(AA / "score_vs_targets.json"), rel(AA / "score_vs_targets.md")],
              ["target", "score", "within-map rho", "p (two-sided)", "attention maps", "MLP maps", "between-expert (8 pts)", "within-expert", "p (two-sided)"], rows, align="ll" + "r" * 7,
              legend=["Targets (budget D, diagnostics): gradpref_a1 / gradpref_a2.4 = -(centred tied dev-CE gradient at the uniform merge, alpha 1 / 2.4); oracle_layerwise_alpha / oracle_layerwise_a1 = fitted per-map lambda, seed 0 (B3).",
                      "Scores A:* use weights only (budget A); B:* use the pooled unlabeled calibration second moment C (budget B): fnorm_D = tr(Delta C Delta^T); rel_fnorm_D = fnorm_D / tr(W0 C W0^T); kappa_WC = kappa_2(W C^1/2); "
                      "dlogkappa_WC = log kappa_WC(W_i) - log kappa_WC(W0); srank_DC, erank_DC = stable / entropy rank of Delta C^1/2; kplus_DC = kappa_plus(Delta C^1/2; 0.01). A:logkappa_rel = log kappa(W_i) - log kappa(W0).",
                      "within-map rho = mean over the 73 maps of the across-expert Spearman; between-expert = correlation over experts of the mean within-map ranks; within-expert = correlation of ranks after removing each expert's mean rank (layer-specific part).",
                      f"The .md file reproduces this JSON exactly in the script's own format (row-by-row string comparison: {V['svt_md_ok']}). Rows with a score that is a monotone function of another within a map (kappa_W / logkappa_rel; fnorm_D / rel_fnorm_D; kappa_WC / dlogkappa_WC) differ only through the permutation draws."])


N_MAP_B4B, N_EXP_B4B = 20000, 5000   # own permutation draws of B4b (seed 0)


def perm_corr_tests(T, S, rng, n_map=N_MAP_B4B, n_exp=N_EXP_B4B):
    """Within-map mean Spearman and within-expert correlation of target T with score S, two-sided permutation p-values (same nulls as aa_score_vs_targets.py)."""
    rt, rs = rk(T), rk(S)
    wm = float(rowcorr(rt, rs).mean())
    null = np.array([rowcorr(rt, np.take_along_axis(rs, np.argsort(rng.random(rs.shape), 1), 1)).mean() for _ in range(n_map)])
    we = within_exp(rt, rs)
    nullw = np.array([within_exp(rt, rk(np.stack([S[rng.permutation(S.shape[0]), i] for i in range(S.shape[1])], 1))) for _ in range(n_exp)])
    return dict(wm=wm, p_wm=float(np.mean(np.abs(null) >= abs(wm))), we=we, p_we=float(np.mean(np.abs(nullw - nullw.mean()) >= abs(we - nullw.mean()))))


def build_B4x():
    """Re-derivation of the report's section 2b (master's tail-robust ratio kappa_q90) from master's spectral statistics and the in-scope targets."""
    st = pd.read_csv(WT / "results" / "spectra" / "stats.csv")
    full = st[st.kind == "full"]
    g1 = pd.read_csv(AA / "grad_pref" / "grad_pref_alpha1.csv")
    names = list(dict.fromkeys(g1[g1.is_linear_map == 1].param))
    att = np.array([mgroup(n) == "attention" for n in names])
    scores = {"log kappa_q90": np.log(full.pivot(index="param", columns="expert", values="kappa_q90").loc[names, TASKS].values),
              "log kappa_q50": np.log(full.pivot(index="param", columns="expert", values="kappa_q50").loc[names, TASKS].values)}
    targets = OrderedDict()
    for a in ("1", "2.4"):
        df = pd.read_csv(AA / "grad_pref" / f"grad_pref_alpha{float(a):g}.csv")
        gm = df[df.is_linear_map == 1].pivot(index="param", columns="expert", values="grad_tied").loc[names, TASKS].values
        targets[f"gradpref_a{a}"] = -(gm - gm.mean(1, keepdims=True))
    for tag in ("layerwise_alpha_seed0", "layerwise_alpha_seed1", "layerwise_a1_seed0", "layerwise_a1_seed1"):
        rec = jload(AA / "oracle_lambda" / f"{tag}.json")
        targets[f"oracle_{tag}"] = np.array([rec["lambda"][n] for n in names])
    rng = np.random.default_rng(0)
    rows, res = [], {}
    for sn, S in scores.items():
        for tn, T in targets.items():
            r = perm_corr_tests(T, S, rng)
            rt, rs = rk(T), rk(S)
            r["attn"] = float(rowcorr(rt[att], rs[att]).mean())
            r["mlp"] = float(rowcorr(rt[~att], rs[~att]).mean())
            res[(sn, tn)] = r
            rows.append([sn, tn, pm(r["wm"], r["p_wm"]), f(r["attn"], 3, True), f(r["mlp"], 3, True), pm(r["we"], r["p_we"])])
    V["q90"] = res
    add_table("B4b", "Re-derivation of report section 2b: master's tail-robust ratio kappa_q90 (and kappa_q50) against the same targets",
              "Varied: score (log kappa_q90, log kappa_q50 of the FULL expert weight; positive rho = a higher ratio coincides with a higher preference, i.e. supports the reversed rule logkappa_q90_rev) and target (6); split: dev; "
              "metric: within-map Spearman (mean over 73 maps), attention / MLP split, within-expert correlation, dimensionless, two-sided permutation p-values from " + f"{N_MAP_B4B} / {N_EXP_B4B}" + " new shuffles (seed 0).",
              ["results/spectra/stats.csv (master's per-map spectra, kind = full; OUTSIDE the scope list, used only for this check)", rel(AA / "grad_pref" / "grad_pref_alpha1.csv"), rel(AA / "grad_pref" / "grad_pref_alpha2.4.csv"),
               rel(AA / "oracle_lambda" / "layerwise_alpha_seed0.json"), "... seed1 and layerwise_a1 seeds 0, 1"],
              ["score", "target", "within-map rho (p)", "attention maps", "MLP maps", "within-expert (p)"], rows, align="llrrrr",
              legend=["Not stored in any result file of the aa track: the report's section 2b table has no producing script in scripts/aa_*.py, so it is re-derived here. The p-values come from this script's own permutation draws and can differ from the report's in the 2nd-3rd decimal.",
                      "Compare with report 2b: gradient pref alpha 1 +0.076 (0.09) / within-expert +0.070 (0.11); alpha 2.4 +0.076 (0.10), attention +0.18, MLP -0.12, within-expert +0.059 (0.19); oracle layerwise + alpha (2 seeds) +0.09 (0.046), within-expert +0.075; oracle alpha = 1 +0.03 (0.48) -- compared in section X (cross-check)."])


# ==================================================================== B5
SOUP_FILES = OrderedDict([
    ("partA_controlled_noise", "A, first run (mild noise)"),
    ("partA_controlled_noise_strong", "A-strong"),
    ("partA_controlled_noise_strong_normmatched", "A-normmatched (A-nm)"),
    ("partA_controlled_noise_strong_v2", "A-strong-v2"),
    ("partB_lora_variants", "B, first run (interrupted)"),
    ("partB_lora_variants_v2", "B-v2 (real LoRA variants)"),
])
GRID_RULES = ["kappa", "logkappa", "kappa_rev", "global:logkappa", "dsrank", "dsrank_rev", "dnorm", "dnorm_rev", "dmedsv", "dnoiseshare"]
FIXED_B = (0.25, 1.0, 4.0, 16.0, math.inf)


def soup_acc(rows):
    """dict (rule, b|None) -> {task: acc}, order of first appearance."""
    out = OrderedDict()
    for r in rows:
        b = None if isnan(float(r["b"])) else float(r["b"])
        out.setdefault((r["rule"], b), OrderedDict())[r["task"]] = r["acc"]
    return out


def soup_lam(rows):
    out = OrderedDict()
    for r in rows:
        b = None if isnan(float(r["b"])) else float(r["b"])
        out.setdefault((r["rule"], b), OrderedDict())[r["task"]] = np.array(r["mean_lambda"])
    return out


def summary_like(acc, tasks):
    """Re-implementation of scripts/aa_summarize_soups.py column logic for one file -> {task: {column: acc}}."""
    orc = sorted({k[0] for k in acc if k[0].startswith(("oracle", "cleanest", "noisiest"))})
    table = {}
    for t in tasks:
        row = {}
        for (rule, b), d in acc.items():
            if t not in d:
                continue
            if rule == "uniform":
                row["uniform"] = d[t]
            if rule == "kappa" and b is not None and any((b == x) or (math.isinf(b) and math.isinf(x)) for x in FIXED_B):
                for x in FIXED_B:
                    if b == x or (math.isinf(b) and math.isinf(x)):
                        row[f"kappa b={x:g}"] = d[t]
            if rule in orc or rule == "mp_invvar":
                row[rule] = d[t]
        for rule in ("kappa", "logkappa", "kappa_rev", "global:logkappa", "dsrank", "dsrank_rev", "dnorm", "dnorm_rev"):
            vals = [d[t] for (r2, b), d in acc.items() if r2 == rule and t in d]
            if vals:
                row[f"best {rule}"] = max(vals)
        table[t] = row
    return table


def wins_of(vals: dict, uni: dict) -> int:
    return sum(1 for t in vals if t in uni and vals[t] > uni[t] + 1e-12)


def build_B5():
    sd = AA / "soup"
    soup_src = (WT / "scripts" / "aa_soup_shared_target.py").read_text()
    levels_default = [float(x) for x in re.search(r'"--levels", default="([^"]+)"', soup_src).group(1).split(",")]
    summ = jload(sd / "summary.json")
    data = OrderedDict()
    for fn in SOUP_FILES:
        rows = jsonl(sd / f"{fn}.jsonl")
        data[fn] = dict(rows=rows, acc=soup_acc(rows), lam=soup_lam(rows), tasks=list(dict.fromkeys(r["task"] for r in rows)))
    V["soup"] = data
    # ---- check against summary.json (re-implemented aa_summarize_soups logic)
    n_cmp, max_d = 0, 0.0
    for fn, d in data.items():
        tab = summary_like(d["acc"], d["tasks"])
        assert set(tab) == set(summ[fn]), fn
        for t in tab:
            assert set(tab[t]) == set(summ[fn][t]), (fn, t, set(tab[t]) ^ set(summ[fn][t]))
            for c, v in tab[t].items():
                max_d = max(max_d, abs(v - summ[fn][t][c]))
                n_cmp += 1
    V["soup_summary_check"] = (n_cmp, max_d)
    assert max_d < 1e-12

    # ---- meta files
    meta_files = OrderedDict([("partA_controlled_noise", "partA_meta.json"), ("partA_controlled_noise_strong", "partA_meta_strong.json"),
                              ("partA_controlled_noise_strong_normmatched", "partA_meta_strong_normmatched.json"), ("partA_controlled_noise_strong_v2", "partA_meta_strong_v2.json")])
    metas = {}
    for fn, mf in meta_files.items():
        m = jload(sd / mf)
        if "per_expert" in m:
            metas[fn] = dict(levels=m["levels"], per=m["per_expert"])
        else:
            metas[fn] = dict(levels=None, per=m)
    V["soup_meta"] = metas
    metaB = jload(sd / "partB_meta_v2.json")
    V["soup_metaB"] = metaB

    # ---- B5a setup
    rows = []
    for fn, label in SOUP_FILES.items():
        d = data[fn]
        K = len(d["rows"][0]["mean_lambda"])
        per_task = defaultdict(int)
        for r in d["rows"]:
            per_task[r["task"]] += 1
        rules = list(dict.fromkeys(r["rule"] for r in d["rows"]))
        if fn.startswith("partA"):
            if metas[fn]["levels"] is None:
                lam0 = d["lam"][("oracle_invvar", None)][d["tasks"][0]]
                rel_lv = np.sqrt(lam0[0] / lam0)
                lv = "x".join([]) or f"{lst(levels_default, 3)} (script default; not stored in the meta file; relative levels {lst(rel_lv, 3)} recovered from the oracle_invvar lambda)"
                assert np.allclose(rel_lv, np.array(levels_default) / levels_default[0])
            else:
                lv = lst(metas[fn]["levels"], 3)
        else:
            lv = "n/a"
        kinds = {"partA_controlled_noise": "noisy copies of each FusionBench expert",
                 "partA_controlled_noise_strong": "noisy copies of each FusionBench expert",
                 "partA_controlled_noise_strong_normmatched": "noisy copies, each update rescaled to the clean update's Frobenius norm per map",
                 "partA_controlled_noise_strong_v2": "noisy copies of each FusionBench expert (same realisations as A-strong)",
                 "partB_lora_variants": "real same-task LoRA variants", "partB_lora_variants_v2": "real same-task LoRA variants"}[fn]
        ntask = len(d["tasks"])
        rowcount = ", ".join(f"{t}:{per_task[t]}" for t in d["tasks"]) if len(set(per_task.values())) > 1 else f"{ntask} x {next(iter(per_task.values()))}"
        rows.append([fn, label, kinds, lv, K, ntask, len(d["rows"]), rowcount, len(rules)])
    K_A = len(data["partA_controlled_noise_strong"]["rows"][0]["mean_lambda"])
    # overlaps
    def common_maxdiff(fa, fb_):
        a, b = data[fa]["acc"], data[fb_]["acc"]
        keys = set(a) & set(b)
        md, n = 0.0, 0
        for k in keys:
            for t in set(a[k]) & set(b[k]):
                md = max(md, abs(a[k][t] - b[k][t]))
                n += 1
        return n, md
    ov1 = common_maxdiff("partA_controlled_noise_strong", "partA_controlled_noise_strong_v2")
    ov2 = common_maxdiff("partB_lora_variants", "partB_lora_variants_v2")
    V["soup_overlap"] = (ov1, ov2)
    dn = [r for r in data["partA_controlled_noise_strong_normmatched"]["rows"] if r["rule"] == "dnorm"]
    dn_dev = max(float(np.abs(np.array(r["mean_lambda"]) - 1 / len(r["mean_lambda"])).max()) for r in dn)
    dn0 = [r for r in data["partA_controlled_noise_strong_v2"]["rows"] if r["rule"] == "dnorm"]
    dn0_dev = max(float(np.abs(np.array(r["mean_lambda"]) - 1 / len(r["mean_lambda"])).max()) for r in dn0)
    V["soup_dnorm_dev"] = (dn_dev, dn0_dev)
    add_table("B5a", "Shared-target soups: the pools (setup)",
              "Varied: pool (A: noisy copies of one FusionBench expert; A-nm: same with norm matching; B: real same-task LoRA variants), where each soup merges the copies/variants of ONE task and is evaluated only on that task's dev split "
              "(split: dev; metric: absolute own-task accuracy, %; budget A for the rules, D for the oracles).",
              [rel(sd / f"{fn}.jsonl") for fn in SOUP_FILES] + [rel(sd / mf) for mf in meta_files.values()] + [rel(sd / "partB_meta_v2.json"), rel(WT / "scripts" / "aa_soup_shared_target.py")],
              ["file (.jsonl)", "report name", "what the copies are", "noise std per copy (x per-matrix rms of the clean update)", "copies / variants per task K", "tasks", "rows", "rows per task", "distinct rules"],
              rows, align="lllllrrlr",
              legend=[f"Part A: for each of the 8 experts, K={K_A} copies W_e + r_j * rms(Delta_e^p) * G_j on every linear map p (G_j iid Gaussian, seeded by the expert index), non-linear parameters from the clean expert; alpha = 1; "
                      "oracle_invvar = inverse-variance weights lambda_j ~ 1/r_j^2; cleanest_only / noisiest_only = one-hot on the first / last copy.",
                      "Part B: the 13 LoRA variants per task (seeds, logit scales, label smoothing, step budgets; Alexandre's banks) merged with alpha = 1; the oracles use each variant's own dev accuracy: oracle_best_single = one-hot on the best variant, "
                      "oracle_quality_softmax{10,30,100} = softmax(q * accuracy).",
                      f"A-strong-v2 reproduces A-strong exactly on the {ov1[0]} common (task, rule, b) accuracies (max abs difference {ov1[1]:.1e}) and adds the rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar; "
                      f"B-v2 reproduces the interrupted first B run exactly on the {ov2[0]} common accuracies (max abs difference {ov2[1]:.1e}); the first B run covers only sun397 ({sum(1 for r in data['partB_lora_variants']['rows'] if r['task'] == 'sun397')} rows) and cars (partial).",
                      f"Norm matching: A-nm rescales every noisy update to the clean update's Frobenius norm per map (script flag --norm-matched), so the dnorm scores of the copies of a map should agree up to round-off. "
                      f"Consistent with this, the mean lambda of the dnorm rule in A-nm stays within {dn_dev:.3f} of uniform over all tasks and temperatures (A-strong: up to {dn0_dev:.3f}), but it is not exactly uniform; "
                      "the accuracy of dnorm in A-nm (B5d3) is therefore best read as a pseudo-random choice driven by round-off (inference, not measured separately).",
                      f"The summary.json / summary.md of the folder are reproduced by this script's re-implementation of scripts/aa_summarize_soups.py: {V['soup_summary_check'][0]} values compared across the six files, max abs difference {V['soup_summary_check'][1]:.1e}."])

    # ---- B5b meta A
    rows = []
    for fn, m in metas.items():
        vals = {"k": [], "f": [], "s": []}
        for e in TASKS:
            d = m["per"][e]
            rows.append([fn, e, f(d["mean_rank_corr_kappa_vs_noise"], 3, True), pct(d["frac_kappa_decreasing_in_noise"], 1), f(d["mean_rank_corr_srank_vs_noise"], 3, True)])
            vals["k"].append(d["mean_rank_corr_kappa_vs_noise"]); vals["f"].append(d["frac_kappa_decreasing_in_noise"]); vals["s"].append(d["mean_rank_corr_srank_vs_noise"])
        rows.append([fn, "**mean over experts**", f(np.mean(vals["k"]), 3, True), pct(np.mean(vals["f"]), 1), f(np.mean(vals["s"]), 3, True)])
        V.setdefault("soup_metaA_mean", {})[fn] = dict(kappa=float(np.mean(vals["k"])), frac=float(np.mean(vals["f"])), srank=float(np.mean(vals["s"])),
                                                        frac_min=float(np.min(vals["f"])), frac_max=float(np.max(vals["f"])))
    assert metas["partA_controlled_noise_strong_v2"] == metas["partA_controlled_noise_strong"], "strong and strong_v2 meta differ"
    for e in TASKS:
        assert metas["partA_controlled_noise_strong_normmatched"]["per"][e]["mean_rank_corr_srank_vs_noise"] == metas["partA_controlled_noise_strong"]["per"][e]["mean_rank_corr_srank_vs_noise"]
    add_table("B5b", "Part A: how kappa and the stable rank of the update move with the injected noise",
              "Varied: pool and expert; metric: mean over the 73 linear maps of the rank correlation (dimensionless) between the copy's noise level and its kappa_2(W) or the stable rank of its update; "
              "'% of maps with kappa decreasing' = share of maps whose kappa correlation with noise is negative; no accuracy involved.",
              [rel(sd / mf) for mf in meta_files.values()],
              ["file", "expert", "mean rank corr(kappa, noise)", "% maps where kappa decreases with noise", "mean rank corr(stable rank of Delta, noise)"], rows, align="llrrr",
              legend=["Negative correlation with noise = more noise gives a better-conditioned (lower kappa) weight, the smoothed-analysis effect (math note 3(b)): the literal rule exp(-beta*kappa) then favours the noisiest copy.",
                      "partA_meta_strong_v2.json is identical to partA_meta_strong.json (same noise realisations); the norm-matched pool changes kappa's correlation with noise but not the stable rank's."])

    # ---- B5c meta B
    rows = []
    variants0 = metaB[TASKS[0]]["variants"]
    for t in TASKS:
        m = metaB[t]
        assert m["variants"] == variants0
        q = np.array(m["quality"])
        rows.append([t, len(m["variants"]), pct(q.min()), pct(q.mean()), pct(q.max()), f(m["spearman_quality_vs_global_logkappa"], 3, True), f(m["spearman_quality_vs_global_srank"], 3, True)])
    ql = [metaB[t]["spearman_quality_vs_global_logkappa"] for t in TASKS]
    qs = [metaB[t]["spearman_quality_vs_global_srank"] for t in TASKS]
    rows.append(["**mean over tasks**", len(variants0), "", "", "", f(np.mean(ql), 3, True), f(np.mean(qs), 3, True)])
    V["soup_B_corr"] = dict(logk_min=min(ql), logk_max=max(ql), srank_mean=float(np.mean(qs)), logk_mean=float(np.mean(ql)))
    add_table("B5c", "Part B: the real LoRA variants -- own-task quality and its relation to kappa / update spread",
              "Varied: task; split: dev; metric: standalone dev accuracy of each of the 13 variants (%, min / mean / max) and Spearman correlation across the 13 variants between quality and the variant's mean log kappa_2 / mean stable rank of its update.",
              [rel(sd / "partB_meta_v2.json")],
              ["task", "variants", "quality min (%)", "quality mean (%)", "quality max (%)", "Spearman(quality, mean log kappa)", "Spearman(quality, mean stable rank)"], rows, align="lrrrrrr",
              legend=["Variants (same 13 for every task): " + ", ".join(variants0) + ".",
                      "Each variant's dense update is gamma * B A with gamma = 1 on every map that has A_/B_ factors in the variant's npz (script comment: all banks r = 16, alpha = 16, per the banks' README); the variants differ by seed, logit scale, label smoothing and step budget (names)."])

    # ---- B5d grid tables (rule x temperature)
    def grid_rows(acc, tasks):
        uni = acc[("uniform", 0.0)]
        rows_ = []
        # uniform
        rows_.append(["uniform", f"{pct(np.mean([uni[t] for t in tasks]))} (–)"] + [""] * len(BGRID))
        for rule in GRID_RULES:
            if (rule, 0.25) not in acc:
                continue
            r = [rule, ""]
            for b in BGRID:
                v = acc[(rule, b)]
                r.append(f"{pct(np.mean([v[t] for t in tasks]))} ({wins_of(v, uni)})")
            rows_.append(r)
        for (rule, b), v in acc.items():
            if rule in GRID_RULES or rule == "uniform":
                continue
            rows_.append([rule + ("" if b is None else f" (b={fb(b)})"), f"{pct(np.mean([v[t] for t in tasks]))} ({wins_of(v, uni)})"] + [""] * len(BGRID))
        return rows_

    hdr = ["rule", "single config: mean (n tasks > uniform)"] + [f"b={fb(b)}" for b in BGRID]
    gid_of = {"partA_controlled_noise": "B5d1", "partA_controlled_noise_strong": "B5d2", "partA_controlled_noise_strong_normmatched": "B5d3",
              "partA_controlled_noise_strong_v2": "B5d4", "partB_lora_variants": "B5d5", "partB_lora_variants_v2": "B5d6"}
    for fn, gid in gid_of.items():
        d = data[fn]
        tasks_ = d["tasks"] if fn != "partB_lora_variants" else ["sun397"]
        extra = {"partA_controlled_noise_strong": " (rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar not run; values identical to the same rows of A-strong-v2)",
                 "partA_controlled_noise_strong_v2": " (superset of A-strong, identical values on the common rows)",
                 "partB_lora_variants": " (interrupted run: only sun397 is complete, the cars rows are partial)",
                 "partB_lora_variants_v2": " (superset of the interrupted first run, identical values on the common rows)"}.get(fn, "")
        add_table(gid, f"Rule x temperature grid, pool {SOUP_FILES[fn]}{extra}",
                  f"Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; "
                  f"metric: absolute own-task accuracy (%) {'averaged over the ' + str(len(tasks_)) + ' tasks' if len(tasks_) > 1 else 'on ' + tasks_[0] + ' only'}, with the number of tasks beating uniform by more than 1e-12 in brackets.",
                  [rel(sd / f"{fn}.jsonl"), rel(sd / "summary.json"), rel(sd / "summary.md")],
                  hdr, grid_rows(d["acc"], tasks_), align="l" + "r" * (len(hdr) - 1),
                  legend=["kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; "
                          "dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.",
                          "mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.",
                          "A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns."]
                  + (["Rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar were not run in this file."] if fn in ("partA_controlled_noise", "partA_controlled_noise_strong") else []))
    # full CSV dump (all rules x b x tasks)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for fn, d in data.items():
        uni = d["acc"].get(("uniform", 0.0), {})
        for (rule, b), v in d["acc"].items():
            r = OrderedDict(file=fn, rule=rule, b="" if b is None else fb(b))
            for t in TASKS:
                r[t] = v.get(t, "")
            vv = [v[t] for t in TASKS if t in v]
            r["mean_acc_over_tasks_present"] = float(np.mean(vv))
            r["n_tasks_beating_uniform"] = wins_of(v, uni)
            recs.append(r)
    pd.DataFrame(recs).to_csv(CSV_DIR / "soup_full_grid.csv", index=False)

    # ---- B5e per-task tables
    def per_task(acc, tasks):
        uni = acc[("uniform", 0.0)]
        rows_ = []

        def add(label, d):
            vals = [d[t] for t in tasks]
            rows_.append([label] + [pct(d[t]) for t in tasks] + [pct(float(np.mean(vals))), "–" if label == "uniform" else str(wins_of(d, uni))])

        add("uniform", uni)
        for b in FIXED_B:
            add(f"kappa b={fb(b)}", acc[("kappa", b)])
        for rule in ("logkappa", "global:logkappa", "kappa_rev", "dsrank", "dnoiseshare", "dmedsv", "dnorm"):
            if (rule, 4.0) in acc:
                add(f"{rule} b=4", acc[(rule, 4.0)])
        for rule in GRID_RULES:
            if (rule, 0.25) in acc:
                add(f"best {rule} (per-task max over the b grid)", {t: max(acc[(rule, b)][t] for b in BGRID) for t in tasks})
        for (rule, b), v in acc.items():
            if rule in GRID_RULES or rule == "uniform":
                continue
            add(rule, v)
        return rows_

    for fn, tid in (("partA_controlled_noise", "B5e1"), ("partA_controlled_noise_strong_v2", "B5e2"), ("partA_controlled_noise_strong_normmatched", "B5e3"), ("partB_lora_variants_v2", "B5e4")):
        d = data[fn]
        add_table(tid, f"Per-task accuracy, pool {SOUP_FILES[fn]}",
                  "Varied: rule (rows) and task (columns); split: dev; metric: absolute own-task accuracy (%), last two columns = mean over the 8 tasks and number of tasks beating uniform.",
                  [rel(sd / f"{fn}.jsonl"), rel(sd / "summary.md"), rel(sd / "summary.json")],
                  ["rule"] + TASKS + ["mean", "tasks > uniform"], per_task(d["acc"], d["tasks"]), align="l" + "r" * 10,
                  legend=[f"'best <rule>' = per task maximum over the {len(BGRID)} temperatures (an optimistic oracle over b, as in summary.md); 'rule b=4' rows are single configurations. Rows are a superset of summary.md's columns (which are reproduced exactly, see B5a legend)."])

    # ---- B5f mean lambda placed on each noise level (A pools)
    def lam_rows(fn):
        lam = data[fn]["lam"]
        K = 4
        sel = [("uniform", 0.0)] + [("kappa", b) for b in (1.0, 4.0, 16.0, math.inf)] + [("logkappa", b) for b in (1.0, 4.0, 16.0, math.inf)] \
              + [("global:logkappa", b) for b in (1.0, 4.0, 16.0, math.inf)] + [("dsrank", 4.0), ("dnorm", 4.0), ("dmedsv", 4.0), ("mp_invvar", 1.0), ("global:mp_invvar", 1.0), ("oracle_invvar", None)]
        rows_ = []
        for key in sel:
            if key not in lam:
                continue
            m = np.stack([lam[key][t] for t in TASKS])
            rows_.append([key[0] + ("" if key[1] is None or key[0] in ("mp_invvar", "global:mp_invvar", "uniform") else f" b={fb(key[1])}")]
                         + [f(x, 3) for x in m.mean(0)] + [f"{f(m[:, -1].min(), 3)} .. {f(m[:, -1].max(), 3)}"])
        return rows_

    for fn, tid in (("partA_controlled_noise_strong_v2", "B5f1"), ("partA_controlled_noise_strong_normmatched", "B5f2")):
        lv = metas[fn]["levels"]
        add_table(tid, f"Weight each rule puts on the copies, pool {SOUP_FILES[fn]}",
                  "Varied: rule and temperature; split: dev (weights do not depend on labels for budget-A rules); metric: mean over maps (then over the 8 tasks) of the weight lambda_j on the copy with noise level " + lst(lv, 3) + " x rms; dimensionless, rows sum to 1.",
                  [rel(sd / f"{fn}.jsonl")],
                  ["rule", f"copy 1 (noise {lv[0]:g})", f"copy 2 ({lv[1]:g})", f"copy 3 ({lv[2]:g})", f"copy 4 ({lv[3]:g}, noisiest)", "weight on noisiest copy: min .. max over the 8 tasks"], lam_rows(fn), align="lrrrrr",
                  legend=["oracle_invvar puts most weight on the cleanest copy. 'Anti-inverse-variance' (shorthand used in the report) = a rule whose weight on the noisiest copy exceeds its weight on the cleanest copy.",
                          "mean over the 8 tasks of the per-task map-averaged weight; the last column gives the spread over tasks of the weight on the noisiest copy."])

    # ---- B5g frozen test
    tf = jsonl(sd / "test_freeze.jsonl")
    tfs = jload(sd / "test_freeze_summary.json")
    pools = list(dict.fromkeys(r["pool"] for r in tf))
    tacc = OrderedDict()
    for r in tf:
        b = None if isnan(float(r["b"])) else float(r["b"])
        lab = f"{r['rule']} b={b:g}" if r["rule"] in ("kappa", "dsrank") else r["rule"]
        tacc.setdefault(r["pool"], OrderedDict()).setdefault(lab, OrderedDict())[r["task"]] = r["acc"]
    n_cmp, max_d, bad = 0, 0.0, []
    for pool in pools:
        uni = tacc[pool]["uniform"]
        assert set(tacc[pool]) == set(tfs[pool]), (pool, set(tacc[pool]) ^ set(tfs[pool]))
        for lab, v in tacc[pool].items():
            mean = float(np.mean([v[t] for t in TASKS]))
            w = 0 if lab == "uniform" else wins_of(v, uni)
            max_d = max(max_d, abs(mean - tfs[pool][lab]["mean"]))
            n_cmp += 1
            if w != tfs[pool][lab]["beats_uniform"] or tfs[pool][lab]["n"] != len(v):
                bad.append((pool, lab, w, tfs[pool][lab]["beats_uniform"]))
    V["test_freeze_check"] = (n_cmp, max_d, bad)
    V["soup_test"] = tacc
    mx_lam = 0.0
    dev_file0 = {"A_strong": "partA_controlled_noise_strong_v2", "A_strong_normmatched": "partA_controlled_noise_strong_normmatched", "B": "partB_lora_variants_v2"}
    for r in tf:
        b = None if isnan(float(r["b"])) else float(r["b"])
        if r["rule"] in ("oracle_invvar", "cleanest_only", "noisiest_only", "oracle_best_single_by_dev"):
            continue
        L = data[dev_file0[r["pool"]]]["lam"].get((r["rule"], b))
        assert L is not None, (r["pool"], r["rule"], b)
        mx_lam = max(mx_lam, float(np.abs(np.array(r["mean_lambda"]) - L[r["task"]]).max()))
    V["test_lambda_vs_dev_maxdiff"] = mx_lam
    rows = []
    pool_name = {"A_strong": "A-strong", "A_strong_normmatched": "A-nm", "B": "B-v2"}
    dev_file = {"A_strong": "partA_controlled_noise_strong_v2", "A_strong_normmatched": "partA_controlled_noise_strong_normmatched", "B": "partB_lora_variants_v2"}
    for pool in pools:
        uni = tacc[pool]["uniform"]
        dacc = data[dev_file[pool]]["acc"]
        duni = dacc[("uniform", 0.0)]
        for lab, v in tacc[pool].items():
            # dev counterpart of the same rule/config
            key = {"uniform": ("uniform", 0.0), "kappa b=1": ("kappa", 1.0), "kappa b=4": ("kappa", 4.0), "dsrank b=4": ("dsrank", 4.0), "mp_invvar": ("mp_invvar", 1.0),
                   "global:mp_invvar": ("global:mp_invvar", 1.0), "oracle_invvar": ("oracle_invvar", None), "cleanest_only": ("cleanest_only", None), "noisiest_only": ("noisiest_only", None),
                   "oracle_best_single_by_dev": ("oracle_best_single", None)}[lab]
            dv = dacc[key]
            rows.append([pool_name[pool], lab] + [pct(v[t]) for t in TASKS] + [pct(float(np.mean([v[t] for t in TASKS]))), "–" if lab == "uniform" else wins_of(v, uni),
                         pct(float(np.mean([dv[t] for t in TASKS]))), "–" if lab == "uniform" else wins_of(dv, duni)])
    V["test_rows"] = rows
    add_table("B5g", "Frozen TEST evaluation of the shared-target soups (configurations fixed before any test look)",
              "Varied: pool (A-strong, A-nm, B-v2) and rule; split: TEST (80% of each official test set; each soup scored on its own task); metric: absolute own-task accuracy (%), mean over the 8 tasks and number of tasks beating uniform; "
              "the dev mean / wins of the same configuration from B5d/B5e are added for comparison.",
              [rel(sd / "test_freeze.jsonl"), rel(sd / "test_freeze_summary.json"), rel(sd / "partA_controlled_noise_strong_v2.jsonl"), rel(sd / "partA_controlled_noise_strong_normmatched.jsonl"), rel(sd / "partB_lora_variants_v2.jsonl")],
              ["pool", "rule"] + TASKS + ["TEST mean", "TEST tasks > uniform", "dev mean", "dev tasks > uniform"], rows, align="ll" + "r" * 12,
              legend=["Frozen rules: uniform; kappa (literal) at b=1 and b=4; dsrank (update stable rank) at b=4; mp_invvar (tuning-free lambda ~ median-sv(Delta)^-2) and global:mp_invvar; references oracle_invvar / cleanest_only / noisiest_only (A pools) and "
                      "oracle_best_single_by_dev (B: best variant chosen by DEV accuracy, evaluated on test).",
                      f"Recomputed means and win counts equal test_freeze_summary.json ({V['test_freeze_check'][0]} configurations; max abs difference of the means {V['test_freeze_check'][1]:.1e}; win-count mismatches: {len(V['test_freeze_check'][2])}). "
                      "No script in scripts/ writes test_freeze_summary.json; it is reproduced here from test_freeze.jsonl.",
                      "Noise realisations are seeded exactly as in the dev runs (script docstring); consistent with that, the mean-lambda vectors stored in test_freeze.jsonl equal those of the dev runs for every frozen configuration compared (max abs difference " + f"{mx_lam:.1e}" + "). A pools use noise levels 1, 4, 8, 16."])


# ==================================================================== B6
def build_B6():
    n_dev = jload(WT / "results" / "sanity" / "expert_matrix_dev.json")["n"]
    ts = OrderedDict()
    for ex in ("svhn", "sun397"):
        ts[ex] = jload(AA / f"tail_stress_{ex}.json")
        assert ts[ex]["expert"] == ex
    V["tail"] = ts
    src_ts = [rel(AA / f"tail_stress_{ex}.json") for ex in ts]
    # ---- B6a edit size
    rows = []
    for ex, d in ts.items():
        rows.append([ex, f(d["floor"], 0), d["n_maps_edited"], pct(d["rel_edit_median"], 3), pct(d["rel_edit_max"], 2)])
    add_table("B6a", "Tail-stress edit: size of the function-null edit",
              "Varied: the edited expert (SVHN or SUN397); the edit floors the singular values of the expert's full weight at floor x sigma_min in every map with kappa(W0) > 1e3 (full fine-tuning, 8 FusionBench experts); "
              "metric: ‖W' - W‖_F / ‖Δ_j‖_F per edited map (%), median and max over the edited maps; no split involved.",
              src_ts + [rel(WT / "scripts" / "aa_tail_stress.py")],
              ["edited expert", "floor (x sigma_min)", "maps edited (of 73)", "median ‖edit‖/‖Δ_j‖ (%)", "max ‖edit‖/‖Δ_j‖ (%)"], rows, align="lrrrr",
              legend=["W' = U diag(max(s, floor * s_min)) V^T per edited map: kappa of the edited expert falls by about the floor factor while the weight changes by a tiny fraction of its fine-tuning update."])
    # ---- B6b expert alone before/after
    rows = []
    for ex, d in ts.items():
        b, a = d["expert_alone_before"], d["expert_alone_after"]
        rows.append([ex, "before"] + [pct(b[t]) for t in TASKS])
        rows.append([ex, "after"] + [pct(a[t]) for t in TASKS])
        rows.append([ex, "after - before (pt)"] + [f"{100 * (a[t] - b[t]):+.3f}" for t in TASKS])
        V.setdefault("tail_alone_maxabs", {})[ex] = max(abs(100 * (a[t] - b[t])) for t in TASKS)
    add_table("B6b", "Tail-stress edit: the edited expert alone, accuracy on all 8 tasks before and after the edit",
              "Varied: before / after the edit; split: dev; metric: absolute accuracy (%) of the single expert (lambda = e_j, alpha = 1) on each task, and the change in points.",
              src_ts,
              ["edited expert", "version"] + TASKS, rows, align="ll" + "r" * 8,
              legend=[f"Largest absolute change on any task: SVHN edit {V['tail_alone_maxabs']['svhn']:.3f} pt, SUN397 edit {V['tail_alone_maxabs']['sun397']:.3f} pt (dev sets of {min(n_dev.values())}..{max(n_dev.values())} images, so one image = {100 / max(n_dev.values()):.2f}..{100 / min(n_dev.values()):.2f} pt on a task)."])
    # ---- B6c every rule row
    rows = []
    for ex, d in ts.items():
        for r in d["rule"]:
            assert abs(np.mean(list(r["acc"].values())) - r["avg_acc"]) < 1e-12
            rows.append([ex, fb(float(r["b"])), f(r["alpha"], 1), r["version"], pct(r["avg_acc"])] + [pct(r["acc"][t]) for t in TASKS] + [f(r["mean_lambda_j"], 4), f(r["mean_lambda_j_attn"], 4)])
    b0 = next(r["mean_lambda_j"] for r in ts["svhn"]["rule"] if math.isinf(float(r["b"])) and r["version"] == "original")
    for r1 in ts["svhn"]["rule"]:
        if r1["version"] != "original":
            continue
        r2 = next(r for r in ts["sun397"]["rule"] if r["version"] == "original" and r["b"] == r1["b"] and r["alpha"] == r1["alpha"])
        assert r1["acc"] == r2["acc"] and r1["avg_acc"] == r2["avg_acc"]
    add_table("B6c", "Tail-stress: every row of both JSON files (literal rule on the original and on the tail-edited expert)",
              "Varied: edited expert, temperature b (rule scale: beta = b / median across-expert std of kappa), alpha, version (original / tail-edited); split: dev (8 tasks); metric: absolute merged accuracy (%) per task and mean, "
              "mean over maps of the weight lambda_j given to the edited expert j (all 73 maps and the 48 attention maps).",
              src_ts,
              ["edited expert", "b", "alpha", "version", "avg acc (%)"] + TASKS + ["mean lambda_j (all maps)", "mean lambda_j (attention maps)"], rows, align="lrrl" + "r" * 11,
              legend=["Rule: lambda_i^p = softmax_i(-beta * kappa_2(W_i^p)) per linear map p, merge W0 + alpha * sum lambda_i Delta_i; b = inf is argmin selection (the expert with the smallest kappa in each map gets weight 1, so mean lambda_j = share of maps it wins: " + f"{round(b0 * 73)}/73 = {b0:.4f}" + " for SVHN before the edit).",
                      "The 'original' rows of the two files are identical except for lambda_j: the merge before the edit does not depend on which expert is later edited (same accuracies in both files)."])
    # ---- B6d before/after summary
    rows = []
    V["tail_pairs"] = []
    for ex, d in ts.items():
        rr = d["rule"]
        for o in [r for r in rr if r["version"] == "original"]:
            e = next(r for r in rr if r["version"] == "tail-edited" and r["b"] == o["b"] and r["alpha"] == o["alpha"])
            own = ex
            d_own = 100 * (e["acc"][own] - o["acc"][own])
            others = [t for t in TASKS if t != own]
            d_oth = float(np.mean([100 * (e["acc"][t] - o["acc"][t]) for t in others]))
            worst_t = min(TASKS, key=lambda t: e["acc"][t] - o["acc"][t])
            d_all = 100 * (e["avg_acc"] - o["avg_acc"])
            rows.append([ex, fb(float(o["b"])), f(o["alpha"], 1), f"{o['mean_lambda_j']:.3f} -> {e['mean_lambda_j']:.3f}", f"{o['mean_lambda_j_attn']:.3f} -> {e['mean_lambda_j_attn']:.3f}",
                         f"{pct(o['avg_acc'])} -> {pct(e['avg_acc'])}", f"{d_all:+.2f}", f"{d_own:+.2f}", f"{d_oth:+.2f}", f"{worst_t} ({100 * (e['acc'][worst_t] - o['acc'][worst_t]):+.2f})"])
            V["tail_pairs"].append(dict(expert=ex, b=float(o["b"]), alpha=o["alpha"], lam0=o["mean_lambda_j"], lam1=e["mean_lambda_j"], lam0a=o["mean_lambda_j_attn"], lam1a=e["mean_lambda_j_attn"],
                                        acc0=o["avg_acc"], acc1=e["avg_acc"], d_all=d_all, d_own=d_own, d_oth=d_oth))
    add_table("B6d", "Tail-stress summary: weight on the edited expert and merged accuracy before -> after the edit",
              "Varied: edited expert, b, alpha; split: dev; metric: mean lambda on the edited expert (all maps; attention maps), absolute merged accuracy (%) and its change (pt): overall, on the edited expert's own task, "
              "mean over the other 7 tasks, and the task with the largest drop (pt).",
              src_ts,
              ["edited expert", "b", "alpha", "mean lambda_j, all maps", "mean lambda_j, attention maps", "avg acc (%)", "change, 8-task mean (pt)", "change on own task (pt)", "mean change on the other 7 tasks (pt)", "largest drop: task (pt)"],
              rows, align="lrrrrrrrrl",
              legend=["Reading: a function-null edit (B6a, B6b) raises the literal rule's weight on the edited expert and moves the merge towards that expert: its own task gains, the other tasks lose.",
                      "alpha = 1.0 rows (b = 4 only) are in the JSON files but not in report section 5, which lists alpha = 1.6 only."])


# ==================================================================== B7
def build_B7():
    d = pd.read_csv(AA / "data_supported" / "data_supported_stats.csv")
    assert d.shape == (584, 10)
    src = (WT / "scripts" / "aa_data_supported_scores.py").read_text()
    doc = re.search(r'"""(.*?)"""', src, re.S).group(1)
    defs = {}
    for line in doc.splitlines():
        m = re.match(r"^  (\w+)\s+(\S.*)$", line)
        if m and m.group(1) in ("fnorm_D", "rel_fnorm_D", "kappa_WC", "dlogkappa_WC", "srank_DC", "erank_DC", "kplus_DC"):
            parts = re.split(r"\s{2,}", m.group(2).strip(), maxsplit=1)
            defs[m.group(1)] = parts[0] + (f" -- {parts[1]}" if len(parts) > 1 else "")
    stats = ["fnorm_D", "rel_fnorm_D", "kappa_WC", "dlogkappa_WC", "srank_DC", "erank_DC", "kplus_DC", "kappa_W0C"]
    for s_ in stats[:-1]:
        assert s_ in defs, s_
    d["mt"] = d["param"].map(mtype)
    d["grp"] = d["param"].map(mgroup)
    names = list(dict.fromkeys(d["param"]))
    assert len(names) == 73 and list(dict.fromkeys(d["expert"])) == TASKS
    V["ds"] = d
    # ---- B7a distribution
    rows = []
    for s_ in stats:
        x = d[s_].values
        q = np.quantile(x, [0, 0.1, 0.5, 0.9, 1.0])
        rows.append([s_, defs.get(s_, "kappa_2(W0 C^{1/2}) of the BASE map (repeated for the 8 experts)"), len(x)] + [g(v, 3) for v in q])
    add_table("B7a", "Data-supported scores (budget B): distribution over the 584 (map, expert) rows",
              "Varied: statistic; no split of the data is evaluated (second moments C come from the pooled unlabeled calibration inputs of the 8 tasks, no task IDs, no labels); metric: min / 10% / median / 90% / max over 73 maps x 8 experts; units as in the legend.",
              [rel(AA / "data_supported" / "data_supported_stats.csv"), rel(WT / "scripts" / "aa_data_supported_scores.py")],
              ["statistic", "definition (script docstring)", "n rows", "min", "q10", "median", "q90", "max"], rows, align="llrrrrrr",
              legend=["C_p = E[x x^T] of the map's input (base-model activations on the union of the 8 calibration splits); C^{1/2} by eigh in float64 with eigenvalues clipped at 0; the patch embedding uses the 3072-dim patch Gram.",
                      "kappa_WC and kappa_W0C are condition numbers (dimensionless); fnorm_D is an expected squared output change (tr(Delta C Delta^T)); rel_fnorm_D is that divided by tr(W0 C W0^T); srank_DC, erank_DC, kplus_DC are dimensionless spread measures of the update's data-weighted spectrum.",
                      "The full 584-row table is the CSV (columns: param, expert, the 8 statistics); 73 maps x 8 experts in the order of fullft_stats.npz."])
    # ---- B7b by module type (median)
    order_m = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"]
    rows = []
    for m in order_m:
        sub = d[d.mt == m]
        rows.append([m, sub.param.nunique(), len(sub)] + [g(sub[s_].median(), 3) for s_ in stats])
    sub = d
    rows.append(["all", sub.param.nunique(), len(sub)] + [g(sub[s_].median(), 3) for s_ in stats])
    add_table("B7b", "Data-supported scores by module type (median over maps x experts)",
              "Varied: module type; no split; metric: median of each statistic over the rows of that module type (12 maps x 8 experts; patch embedding: 1 map x 8).",
              [rel(AA / "data_supported" / "data_supported_stats.csv")],
              ["module", "maps", "rows"] + stats, rows, align="lrr" + "r" * 8)
    # ---- B7c by expert
    rows = []
    for t in TASKS:
        sub = d[d.expert == t]
        rows.append([t] + [g(sub[s_].median(), 3) for s_ in stats[:-1]])
    add_table("B7c", "Data-supported scores by expert (median over the 73 maps)",
              "Varied: expert; no split; metric: median over the 73 maps of each statistic for that expert.",
              [rel(AA / "data_supported" / "data_supported_stats.csv")],
              ["expert"] + stats[:-1], rows, align="l" + "r" * 7,
              legend=["Compare with the expert-identity structure of B2f: the update-energy and update-spread statistics differ systematically between experts, which is why they correlate with the expert-level preference in B4a."])
    # ---- B7d within-map spread across the 8 experts
    rows = []
    for s_ in stats[:-1]:
        piv = d.pivot(index="param", columns="expert", values=s_).loc[names, TASKS].values
        if s_ == "dlogkappa_WC":
            spread = piv.max(1) - piv.min(1)
            lab = "max - min (log units)"
        else:
            spread = piv.max(1) / piv.min(1)
            lab = "max / min"
        sp = np.quantile(spread, [0.1, 0.5, 0.9])
        # fraction of maps where the 8 values are not all distinct
        rows.append([s_, lab] + [g(v, 3) for v in sp])
    add_table("B7d", "Spread of each data-supported statistic across the 8 experts inside a map",
              "Varied: statistic; no split; metric: per map, max/min (or max-min for the signed dlogkappa_WC) over the 8 experts; quantiles over the 73 maps give the dynamic range the softmax rule has to work with.",
              [rel(AA / "data_supported" / "data_supported_stats.csv")],
              ["statistic", "spread measure", "q10 over maps", "median over maps", "q90 over maps"], rows, align="llrrr")
    # ---- B7e kappa_W0C vs base kappa, numerically singular rows
    pmap = V.get("per_map")
    if pmap is None:
        pmap = pd.read_csv(AA / "grad_pref" / "per_map_grad_vs_kappa.csv")
    k0 = pmap[pmap.alpha == pmap.alpha.min()].set_index("param")["kappa0"]
    base = d.groupby("param", sort=False)["kappa_W0C"].agg(["first", "nunique"])
    assert (base["nunique"] == 1).all()
    rows = []
    for m in order_m + ["all"]:
        ps = [n for n in names if (mtype(n) == m or m == "all")]
        a = np.array([base.loc[n, "first"] for n in ps])
        b = np.array([k0.loc[n] for n in ps])
        sp = spearmanr(a, b).statistic if len(ps) > 2 else float("nan")
        rows.append([m, len(ps), kfmt(float(np.median(b))), kfmt(float(np.median(a))), g(float(np.median(a / b)), 3), f(sp, 3, True) if not isnan(sp) else "–",
                     int((a > 1e12).sum())])
    big = d[d.kappa_WC > 1e12]
    add_table("B7e", "Data-weighted vs plain condition number of the BASE map, and numerically singular rows",
              "Varied: module type; no split; metric: median kappa_2(W0) (from per_map_grad_vs_kappa.csv) vs median kappa_2(W0 C^{1/2}), median ratio, Spearman correlation across maps, and the number of maps whose data-weighted base kappa exceeds 1e12.",
              [rel(AA / "data_supported" / "data_supported_stats.csv"), rel(AA / "grad_pref" / "per_map_grad_vs_kappa.csv")],
              ["module", "maps", "median kappa(W0)", "median kappa(W0 C^1/2)", "median ratio", "Spearman across maps", "maps with kappa(W0 C^1/2) > 1e12"], rows, align="lrrrrrr",
              legend=[f"Rows with kappa_WC > 1e12: {len(big)} of 584, in {big.param.nunique()} maps ({', '.join(short(n) + ' x' + str(int((big.param == n).sum())) for n in big.param.unique())}); "
                      f"values of order 1e15 and above are at the float64 round-off floor (1/eps = {1 / np.finfo(float).eps:.2g}), i.e. not meaningful condition numbers; the cause is not recorded in these files "
                      "(plausibly a numerically rank-deficient pooled input Gram C for the layer-0 inputs). Statistics derived from those rows (kappa_WC, dlogkappa_WC) are round-off in B4a/B4b.",
                      "median kappa(W0): the base-map condition number from fp64 SVD of the stored fp32 matrix."])
    # ---- B7f within-map rank agreement between the statistics
    mats = {s_: d.pivot(index="param", columns="expert", values=s_).loc[names, TASKS].values for s_ in stats[:-1]}
    rows = []
    for a_ in stats[:-1]:
        row = [a_]
        for b_ in stats[:-1]:
            row.append(f(float(rowcorr(rk(mats[a_]), rk(mats[b_])).mean()), 2, True))
        rows.append(row)
    add_table("B7f", "Agreement between data-supported statistics: mean within-map Spearman across the 8 experts",
              "Varied: pair of statistics; no split; metric: Spearman correlation across the 8 experts inside each map, averaged over the 73 maps (dimensionless).",
              [rel(AA / "data_supported" / "data_supported_stats.csv")],
              ["statistic"] + stats[:-1], rows, align="l" + "r" * 7,
              legend=["fnorm_D and rel_fnorm_D have the same within-map ranks (the divisor tr(W0 C W0^T) is constant inside a map); kappa_WC and dlogkappa_WC likewise (the subtracted log kappa_WC(W0) is constant inside a map)."])


# ==================================================================== X cross-check against the report
def parse_md_tables(text: str):
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[i:j]]
            out.append(dict(line=i + 1, rows=rows))
            i = j
        else:
            i += 1
    return out


def nums(cell: str) -> list[float]:
    c = cell.replace("−", "-").replace("**", "")
    return [float(x) for x in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?", c)]


def status_num(rep: float, comp: float, d: int) -> str:
    r = float(f"{comp:.{d}f}")
    unit = 10.0 ** (-d)
    if abs(r - rep) < unit * 1e-6:
        return "OK"
    if abs(comp - rep) <= unit * 1.0001:
        return "ROUNDING"
    return "MISMATCH"


def build_X():
    rep_path = WT / "docs" / "aa" / "AA_SIDE_REPORT.md"
    math_path = WT / "docs" / "aa" / "MATH_kappa_of_finetuned_weights.md"
    R, M = rep_path.read_text(), math_path.read_text()
    tabs = parse_md_tables(R)
    rows_out = []     # (where, claim, rep_txt, comp_txt, status, source/note)
    UNVER = []        # claims that cannot be verified from the files in scope
    ncount = [0]

    def line_of(text, pos):
        return text[:pos].count("\n") + 1

    def add(where, claim, rep_txt, comp_txt, status, src, note=""):
        ncount[0] += 1
        rows_out.append([ncount[0], where, claim, rep_txt, comp_txt, status, (src + (" -- " + note if note else ""))])

    def num_check(where, claim, rep, comp, d, src, note="", rep_txt=None, others=None):
        st = status_num(rep, comp, d)
        extra = ""
        if st != "OK" and others:
            ok = [k for k, v in others.items() if status_num(rep, v, d) == "OK"]
            if ok:
                st = f"{st} (OK for {', '.join(ok)})"
            extra = "; other candidates: " + ", ".join(f"{k} {v:.{d + 2}f}" for k, v in others.items())
        add(where, claim, rep_txt if rep_txt is not None else f"{rep:.{d}f}", f"{comp:.{d + 2}f}", st, src, note + extra)
        return st

    def bound_check(where, claim, bound, comp, d, src, note="", rep_txt=None, upper=True):
        ok = comp <= bound + 1e-9 if upper else comp >= bound - 1e-9
        add(where, claim, rep_txt or (("<= " if upper else ">= ") + f"{bound:g}"), f"{comp:.{d + 2}f}", "OK" if ok else "MISMATCH", src, note)

    fits, uni, ext = V["fits"], V["uni"], V["ext_uniform"]
    S = V["soup"]
    POOL = {"A": "partA_controlled_noise_strong_v2", "A-nm": "partA_controlled_noise_strong_normmatched", "B": "partB_lora_variants_v2"}

    def smean(fn, rule, b):
        v = S[fn]["acc"][(rule, b)]
        return 100 * float(np.mean([v[t] for t in TASKS]))

    def swins(fn, rule, b):
        v = S[fn]["acc"][(rule, b)]
        return wins_of(v, S[fn]["acc"][("uniform", 0.0)])

    # ---------- section 3 table (oracle)
    t3 = next(t for t in tabs if t["rows"][0][0] == "bank" and "global oracle" in t["rows"][0][2])
    for ri, row in enumerate(t3["rows"][2:], start=2):
        bank = "full-FT" if row[0].startswith("full-FT") else "LoRA"
        ln = t3["line"] + ri
        cols = [("uniform (best alpha)", None), ("global oracle", "global_alpha"), ("layerwise oracle, alpha = 1", "layerwise_a1"), ("layerwise oracle + alpha", "layerwise_alpha")]
        for ci, (lab, variant) in enumerate(cols, start=1):
            rn = nums(row[ci])
            if variant is None:
                comps = [100 * uni[bank]["dev"]] + ([100 * ext[bank]["mean_acc"]] if len(rn) > 1 else [])
                names_ = ["dev", "test"]
                for k, (rv, cv) in enumerate(zip(rn, comps)):
                    src = "uniform_alpha_grid.json" if k == 0 else ext[bank]["src"]
                    num_check(f"§3 table l.{ln}, {bank}, {lab}, {names_[k]}", row[ci], rv, cv, 1, src, rep_txt=f"{rv:.1f}")
            else:
                for k, sp in enumerate(("dev", "test")):
                    cands = {f"seed {sd}": 100 * fits[(bank, variant, sd)][sp]["avg_acc"] for sd in (0, 1) if (bank, variant, sd) in fits}
                    if len([1 for sd in (0, 1) if (bank, variant, sd) in fits]) == 2:
                        cands["2-seed mean"] = float(np.mean([cands["seed 0"], cands["seed 1"]]))
                    c0 = cands.pop("seed 0")
                    num_check(f"§3 table l.{ln}, {bank}, {lab}, {sp}", row[ci], rn[k], c0, 1, f"{variant}_seed0.json ({sp})", rep_txt=f"{rn[k]:.1f}", others=cands)
    # ---------- section 4 table (soups, dev)
    t4 = next(t for t in tabs if t["rows"][0][0] == "rule" and t["rows"][0][1] == "A" and t["rows"][0][2] == "A-nm" and "real variants" in t["rows"][0][3])
    cols4 = {1: "A", 2: "A-nm", 3: "B"}
    for ri, row in enumerate(t4["rows"][2:], start=2):
        ln = t4["line"] + ri
        lab = row[0]
        for ci, pool in cols4.items():
            fn = POOL[pool]
            rn = nums(row[ci])
            w = f"§4 table l.{ln}, {pool}, '{lab[:40]}'"
            if lab == "uniform":
                num_check(w, row[ci], rn[0], smean(fn, "uniform", 0.0), 1, f"{fn}.jsonl")
            elif lab.startswith("oracle"):
                key = ("oracle_quality_softmax100", None) if pool == "B" else ("oracle_invvar", None)
                num_check(w, row[ci], rn[0], smean(fn, *key), 1, f"{fn}.jsonl, rule {key[0]}")
            elif lab.startswith("literal"):
                for rv, b in zip(rn, (1.0, 16.0, math.inf)):
                    num_check(w + f", b={fb(b)}", row[ci], rv, smean(fn, "kappa", b), 1, f"{fn}.jsonl, rule kappa")
            elif lab.startswith("tuning-free"):
                num_check(w, row[ci], rn[0], smean(fn, "mp_invvar", 1.0), 1, f"{fn}.jsonl, rule mp_invvar")
            elif lab.startswith("update shape"):
                shape = ("dsrank", "dnoiseshare", "dmedsv")
                if pool == "A":
                    for r_ in shape:
                        num_check(w + f", {r_} b=4", row[ci], rn[0], smean(fn, r_, 4.0), 1, f"{fn}.jsonl")
                elif pool == "A-nm":
                    for r_ in shape:
                        c = smean(fn, r_, 4.0)
                        okv = any(abs(float(f"{c:.1f}") - x) < 1e-6 for x in rn)
                        add(w + f", {r_} b=4", row[ci], "/".join(f"{x:.1f}" for x in rn), f"{c:.3f}", "OK" if okv else "MISMATCH", f"{fn}.jsonl", "reported as a range; each of the three rules must round to one end")
                else:
                    best = max(max(smean(fn, r_, b) for b in BGRID) for r_ in shape)
                    atb4 = ", ".join(f"{r_} {smean(fn, r_, 4.0):.2f}" for r_ in shape)
                    bound_check(w + ", best b over the 3 shape rules", row[ci], rn[0], best, 1, f"{fn}.jsonl", note=f"the cell is an upper bound on the best b; at b=4 itself the three rules give: {atb4}")
            elif lab.startswith("update norm"):
                num_check(w + ", dnorm b=4", row[ci], rn[0], smean(fn, "dnorm", 4.0), 1, f"{fn}.jsonl", note=("B cell is stated as '≈ 91.0'" if pool == "B" else ""))
    # ---------- frozen TEST table
    t4b = next(t for t in tabs if t["rows"][0][0] == "rule" and t["rows"][0][1] == "A" and t["rows"][0][2] == "A-nm" and t["rows"][0][3] == "B")
    TP = {1: "A_strong", 2: "A_strong_normmatched", 3: "B"}
    TN = {"A_strong": "A", "A_strong_normmatched": "A-nm", "B": "B"}
    tt = V["soup_test"]

    def tmean(pool, lab):
        return 100 * float(np.mean([tt[pool][lab][t] for t in TASKS]))

    def twins(pool, lab):
        return wins_of(tt[pool][lab], tt[pool]["uniform"])

    for ri, row in enumerate(t4b["rows"][2:], start=2):
        ln = t4b["line"] + ri
        lab = row[0]
        for ci, pool in TP.items():
            rn = nums(row[ci])
            w = f"§4 frozen-TEST table l.{ln}, {TN[pool]}, '{lab[:34]}'"
            src = "test_freeze.jsonl / test_freeze_summary.json"
            if lab == "uniform":
                num_check(w, row[ci], rn[0], tmean(pool, "uniform"), 1, src)
            elif lab.startswith("literal"):
                for rv, l2 in zip(rn[:2], ("kappa b=1", "kappa b=4")):
                    num_check(w + f", {l2}", row[ci], rv, tmean(pool, l2), 1, src)
                if len(rn) >= 4:
                    for l2 in ("kappa b=1", "kappa b=4"):
                        add(w + f", {l2}, tasks above uniform", row[ci], f"{int(rn[2])}/8", f"{twins(pool, l2)}/8", "OK" if twins(pool, l2) == int(rn[2]) else "MISMATCH", src)
            elif lab.startswith("update stable rank"):
                num_check(w + ", dsrank b=4", row[ci], rn[0], tmean(pool, "dsrank b=4"), 1, src)
                add(w + ", tasks above uniform", row[ci], f"{int(rn[1])}/8", f"{twins(pool, 'dsrank b=4')}/8", "OK" if twins(pool, "dsrank b=4") == int(rn[1]) else "MISMATCH", src)
            elif lab.startswith("GD/MP"):
                num_check(w + ", mp_invvar", row[ci], rn[0], tmean(pool, "mp_invvar"), 1, src)
                if len(rn) >= 3:
                    add(w + ", tasks above uniform", row[ci], f"{int(rn[1])}/8", f"{twins(pool, 'mp_invvar')}/8", "OK" if twins(pool, "mp_invvar") == int(rn[1]) else "MISMATCH", src)
            elif lab == "oracle":
                l2 = "oracle_best_single_by_dev" if pool == "B" else "oracle_invvar"
                num_check(w + f", {l2}", row[ci], rn[0], tmean(pool, l2), 1, src)
    # ---------- section 5 table (tail stress)
    t5 = next(t for t in tabs if t["rows"][0][0] == "literal rule")
    pairs = V["tail_pairs"]

    def pair(ex, b, a=1.6):
        return next(p for p in pairs if p["expert"] == ex and (p["b"] == b or (math.isinf(b) and math.isinf(p["b"]))) and abs(p["alpha"] - a) < 1e-9)

    for ri, row in enumerate(t5["rows"][2:], start=2):
        ln = t5["line"] + ri
        lab = row[0]
        l_n, a_n = nums(row[1]), nums(row[2])
        src = "tail_stress_*.json"
        if lab.startswith("SVHN edited"):
            p = pair("svhn", 4.0)
            vals = [p["lam0"], p["lam1"], p["lam0a"], p["lam1a"]]
            for rv, cv, nm in zip(l_n, vals, ("lambda before", "lambda after", "attn before", "attn after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[1], rv, cv, 2, src)
            for rv, cv, nm in zip(a_n, (100 * p["acc0"], 100 * p["acc1"]), ("acc before", "acc after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[2], rv, cv, 1, src)
        elif lab.startswith("SVHN, b = 16"):
            p = pair("svhn", 16.0)
            for rv, cv, nm in zip(l_n, (p["lam0"], p["lam1"], p["lam1a"]), ("lambda before", "lambda after", "attn after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[1], rv, cv, 2, src)
            for rv, cv, nm in zip(a_n, (100 * p["acc0"], 100 * p["acc1"]), ("acc before", "acc after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[2], rv, cv, 1, src)
        elif lab.startswith("SVHN, b = ∞"):
            p = pair("svhn", math.inf)
            for rv, cv, nm in zip(l_n, (p["lam0"], p["lam1"]), ("lambda before", "lambda after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[1], rv, cv, 2, src, note=(f"= {p['lam1'] * 73:.0f}/73" if nm == "lambda after" else ""))
            for rv, cv, nm in zip(a_n, (100 * p["acc0"], 100 * p["acc1"]), ("acc before", "acc after")):
                num_check(f"§5 table l.{ln}, {lab}, {nm}", row[2], rv, cv, 1, src)
        elif lab.startswith("SUN397"):
            ps = [pair("sun397", b) for b in (4.0, 16.0, math.inf)]
            num_check(f"§5 table l.{ln}, {lab}, lambda before (b=4)", row[1], l_n[0], ps[0]["lam0"], 2, src)
            for rv, p, b in zip(l_n[1:], ps, ("4", "16", "inf")):
                num_check(f"§5 table l.{ln}, {lab}, lambda after b={b}", row[1], rv, p["lam1"], 2, src)
            accs = [x for p in ps for x in (100 * p["acc0"], 100 * p["acc1"])]
            for k, (rv, cv) in enumerate(zip(a_n, accs)):
                num_check(f"§5 table l.{ln}, {lab}, acc {'before' if k % 2 == 0 else 'after'} b={('4', '16', 'inf')[k // 2]}", row[2], rv, cv, 1, src)
    # ---------- section 2 table
    t2 = next(t for t in tabs if t["rows"][0][0] == "score" and "gradient pref" in t["rows"][0][1])
    svt = V["svt"]
    g1, g24, oa, o1 = svt["gradpref_a1"], svt["gradpref_a2.4"], svt["oracle_layerwise_alpha"], svt["oracle_layerwise_a1"]
    for ri, row in enumerate(t2["rows"][2:], start=2):
        ln = t2["line"] + ri
        lab = row[0]
        n1, n2, n3 = nums(row[1]), nums(row[2]), nums(row[3])
        w = f"§2 table l.{ln}, '{lab[:30]}'"
        if lab.startswith("κ₂(W)"):
            num_check(w + ", ρ vs gradient pref a=1", row[1], n1[0], g1["A:kappa_W"]["within_map"], 3, "score_vs_targets.json gradpref_a1", rep_txt=row[1].replace("−", "-"))
            num_check(w + ", ρ vs oracle λ", row[2], n2[0], oa["A:kappa_W"]["within_map"], 3, "score_vs_targets.json oracle_layerwise_alpha", rep_txt=row[2].replace("−", "-"))
            num_check(w + ", within-expert vs gradient", row[3], n3[0], g1["A:kappa_W"]["within_exp"], 3, "score_vs_targets.json gradpref_a1")
            num_check(w + ", its p", row[3], n3[1], g1["A:kappa_W"]["p_within_two"], 2, "score_vs_targets.json (two-sided); one-sided p(null>=obs) in decompose_pref.json = " + f"{V['gp_dec']['1']['kappa_W']['p_within_ge']:.4f}")
            add(w + ", within-expert vs oracle", row[3], "≈ 0", f"{oa['A:kappa_W']['within_exp']:+.4f} (oracle alpha) / {o1['A:kappa_W']['within_exp']:+.4f} (oracle alpha=1)",
                "OK" if max(abs(oa["A:kappa_W"]["within_exp"]), abs(o1["A:kappa_W"]["within_exp"])) < 0.05 else "MISMATCH", "score_vs_targets.json oracle_layerwise_alpha / oracle_layerwise_a1")
        elif lab.startswith("κ(W C"):
            num_check(w + ", ρ vs gradient pref a=1", row[1], n1[0], g1["B:kappa_WC"]["within_map"], 3, "score_vs_targets.json", rep_txt=row[1])
            num_check(w + ", ρ vs oracle λ", row[2], n2[0], oa["B:kappa_WC"]["within_map"], 3, "score_vs_targets.json", rep_txt=row[2].replace("−", "-"))
            ps = [svt[t]["B:kappa_WC"]["p_within_two"] for t in svt]
            bound_check(w + ", within-expert p over the 4 targets ('p >= 0.14')", row[3], 0.14, min(ps), 2, "score_vs_targets.json (min over 4 targets)", rep_txt="p >= 0.14", upper=False)
        elif lab.startswith("update spread"):
            num_check(w + ", stable rank vs gradient pref a=1", row[1], n1[0], g1["A:stable_rank_D"]["within_map"], 2, "score_vs_targets.json", rep_txt=f"{n1[0]:.2f}")
            num_check(w + ", entropy rank vs gradient pref a=1", row[1], n1[1], g1["A:entropy_rank_D"]["within_map"], 2, "score_vs_targets.json (analysis_grad_vs_kappa.json: " + f"{V['gp_ana']['1']['entropy_rank_D']['mean_spearman']:.4f})", rep_txt=f"{n1[1]:.2f}")
            num_check(w + ", stable rank vs oracle λ", row[2], n2[0], oa["A:stable_rank_D"]["within_map"], 2, "score_vs_targets.json oracle_layerwise_alpha", rep_txt=f"{n2[0]:.2f}")
            num_check(w + ", entropy rank vs oracle λ", row[2], n2[0], oa["A:entropy_rank_D"]["within_map"], 2, "score_vs_targets.json oracle_layerwise_alpha", rep_txt=f"{n2[0]:.2f}")
            btw = [abs(svt[t][s_]["between"]) for t in svt for s_ in ("A:stable_rank_D", "A:entropy_rank_D")]
            btw_by = {t: (min(abs(svt[t][s_]["between"]) for s_ in ("A:stable_rank_D", "A:entropy_rank_D")), max(abs(svt[t][s_]["between"]) for s_ in ("A:stable_rank_D", "A:entropy_rank_D"))) for t in svt}
            add(w + ", between-expert", row[3], "±0.8", f"abs(between-expert) from {min(btw):.2f} to {max(btw):.2f} over 4 targets x 2 scores", "IMPRECISE" if min(btw) < 0.7 else "OK", "score_vs_targets.json",
                "per target (min..max over the two scores): " + "; ".join(f"{t} {lo:.2f}..{hi:.2f}" for t, (lo, hi) in btw_by.items()))
            we = [svt[t][s_]["within_exp"] for t in svt for s_ in ("A:stable_rank_D", "A:entropy_rank_D")]
            add(w + ", within-expert", row[3], "≈ 0", f"max abs(within-expert) = {max(abs(x) for x in we):.3f}", "OK" if max(abs(x) for x in we) < 0.05 else "MISMATCH", "score_vs_targets.json")
        elif lab.startswith("functional energy"):
            num_check(w + ", ρ vs gradient pref a=1", row[1], n1[0], g1["B:fnorm_D"]["within_map"], 2, "score_vs_targets.json", rep_txt=row[1])
            num_check(w + ", ρ vs oracle λ", row[2], n2[0], oa["B:fnorm_D"]["within_map"], 2, "score_vs_targets.json oracle_layerwise_alpha", rep_txt=row[2])
            vals = {t: (svt[t]["B:fnorm_D"]["within_exp"], svt[t]["B:fnorm_D"]["p_within_two"]) for t in svt}
            lohi = [v[0] for v in vals.values()]
            inrange = [t for t, v in vals.items() if 0.115 <= v[0] <= 0.165 and v[1] <= 0.0075]
            outr = [t for t in vals if t not in inrange]
            add(w + ", within-expert", row[3], "+0.12 to +0.16 (p <= 0.007)", "; ".join(f"{t}: {v[0]:+.3f} (p {v[1]:.3f})" for t, v in vals.items()),
                ("PARTIAL: holds for " + ", ".join(inrange) + "; not for " + ", ".join(outr)) if outr else "OK", "score_vs_targets.json B:fnorm_D",
                "the stated range/p-bound is checked per target")
            sig = []
            for t in ("gradpref_a1", "gradpref_a2.4"):
                for s_ in ("A:fro_D", "A:smin_W"):
                    if svt[t][s_]["p_within_two"] < 0.05:
                        sig.append(f"{t} {s_} {svt[t][s_]['within_exp']:+.3f} (p {svt[t][s_]['p_within_two']:.3f})")
            sig_or = [f"{t} {s_}" for t in ("oracle_layerwise_alpha", "oracle_layerwise_a1") for s_ in svt[t] if svt[t][s_]["p_within_two"] < 0.05]
            add(w + ", 'the only per-layer predictor'", row[3], "only functional energy", "also p<0.05 vs gradient pref: " + "; ".join(sig) + " | significant within-expert vs oracle targets: " + ", ".join(sig_or),
                "NOT UNIQUE (budget-A scores also significant)" if sig else "OK", "score_vs_targets.json",
                "A:fro_D (update norm, weights only) and A:smin_W (negative sign) are significant within-expert for the gradient-preference targets")

    # ---------- section 2b table (master's kappa_q90) vs the re-derivation in B4x
    t2b = next((t for t in tabs if t["rows"][0][0] == "target" and "within-map ρ (p)" in t["rows"][0][1]), None)
    if t2b and "q90" in V:
        q = V["q90"]

        def pcheck(where, claim, rep_p, comp_ps, src, note=""):
            ok = min(abs(rep_p - c) for c in comp_ps) <= 0.03
            add(where, claim, f"{rep_p:g}", ", ".join(f"{c:.3f}" for c in comp_ps), "OK (permutation noise, abs(diff) <= 0.03)" if ok else "DIFFERS (> 0.03)", src, note)

        for ri, row in enumerate(t2b["rows"][2:], start=2):
            ln = t2b["line"] + ri
            lab = row[0]
            n1, n2 = nums(row[1]), nums(row[2])
            w = f"§2b table l.{ln}, '{lab}'"
            src = "B4b (re-derived from results/spectra/stats.csv + in-scope targets)"
            if lab.startswith("gradient pref, α = 1"):
                r_ = q[("log kappa_q90", "gradpref_a1")]
                num_check(w + ", within-map ρ", row[1], n1[0], r_["wm"], 3, src)
                pcheck(w + ", its p", row[1], n1[1], [r_["p_wm"]], src)
                num_check(w + ", within-expert", row[2], n2[0], r_["we"], 3, src)
                pcheck(w + ", its p", row[2], n2[1], [r_["p_we"]], src)
            elif lab.startswith("gradient pref, α = 2.4"):
                r_ = q[("log kappa_q90", "gradpref_a2.4")]
                num_check(w + ", within-map ρ", row[1], n1[0], r_["wm"], 3, src)
                pcheck(w + ", its p", row[1], n1[1], [r_["p_wm"]], src)
                num_check(w + ", attention maps", row[1], n1[2], r_["attn"], 2, src)
                num_check(w + ", MLP maps", row[1], n1[3], r_["mlp"], 2, src)
                num_check(w + ", within-expert", row[2], n2[0], r_["we"], 3, src)
                pcheck(w + ", its p", row[2], n2[1], [r_["p_we"]], src)
            elif lab.startswith("oracle layerwise + α"):
                rs_ = [q[("log kappa_q90", f"oracle_layerwise_alpha_seed{sd}")] for sd in (0, 1)]
                num_check(w + ", within-map ρ (mean of the 2 seeds)", row[1], n1[0], float(np.mean([r["wm"] for r in rs_])), 2, src, others={"seed 0": rs_[0]["wm"], "seed 1": rs_[1]["wm"]})
                pcheck(w + ", its p (per seed)", row[1], n1[1], [r["p_wm"] for r in rs_], src)
                num_check(w + ", within-expert (mean of the 2 seeds)", row[2], n2[0], float(np.mean([r["we"] for r in rs_])), 3, src, others={"seed 0": rs_[0]["we"], "seed 1": rs_[1]["we"]})
                pcheck(w + ", its p (per seed)", row[2], n2[1], [r["p_we"] for r in rs_], src)
            elif lab.startswith("oracle, α = 1"):
                r_ = q[("log kappa_q90", "oracle_layerwise_a1_seed0")]
                r1_ = q[("log kappa_q90", "oracle_layerwise_a1_seed1")]
                num_check(w + ", within-map ρ (seed 0)", row[1], n1[0], r_["wm"], 2, src, others={"seed 1": r1_["wm"]})
                pcheck(w + ", its p (seed 0)", row[1], n1[1], [r_["p_wm"]], src, note=f"seed 1: rho {r1_['wm']:+.3f}, p {r1_['p_wm']:.3f}")
    UNVER.append(("§2b", "log kappa_q90 rows (+0.076 (0.09), ...)", "no producing script/result file in the aa track; re-derived in B4b from master's results/spectra/stats.csv (see the §2b rows of this table)"))
    # ---------- section 1 and other text claims
    def rx(text, pattern, label, where_prefix, cvt=lambda s: float(s.replace("−", "-"))):
        m = re.search(pattern, text)
        if not m:
            add(where_prefix, label, "–", "–", "CLAIM TEXT NOT FOUND", "report text changed?")
            return None, None
        return [cvt(x) for x in m.groups()], line_of(text, m.start())

    # §1: gradient pref and oracle correlation
    vals, ln = rx(R, r"Within-map ρ\(gradient preference, κ\) = ([+−]?\d\.\d+); oracle λ vs κ ([−+]\d\.\d+)…([−+]\d\.\d+)", "Within-map ρ(gradient pref, κ), oracle λ vs κ", "§1")
    if vals:
        num_check(f"§1 l.{ln}", "Within-map ρ(gradient preference, κ) = +0.045", vals[0], V["gp_ana"]["1"]["kappa"]["mean_spearman"], 3, "analysis_grad_vs_kappa.json (alpha 1)")
        ov = [V["oracle_ana"][t]["kappa_W"]["mean_within_map_spearman"] for t in ("layerwise_a1_seed0", "layerwise_a1_seed1", "layerwise_alpha_seed0", "layerwise_alpha_seed1")]
        num_check(f"§1 l.{ln}", "oracle λ vs κ: lower end −0.02", vals[1], min(ov), 2, "analysis_oracle_vs_kappa.json (4 layerwise fits)")
        num_check(f"§1 l.{ln}", "oracle λ vs κ: upper end +0.02", vals[2], max(ov), 2, "analysis_oracle_vs_kappa.json (4 layerwise fits)")
    vals, ln = rx(R, r"Within-layer ρ\(κ, σ_min\) = ([−+]?\d\.\d+) \(full-FT\), ([−+]?\d\.\d+) \(LoRA\)", "Within-layer ρ(κ, σ_min)", "§1")
    if vals:
        num_check(f"§1 l.{ln}", "Within-layer ρ(κ, σ_min) = −0.91 (full-FT)", vals[0], V["gp_ana"]["1"]["spearman_kappa_vs_sminW"], 2, "analysis_grad_vs_kappa.json")
        ls = pd.read_csv(AA / "lora" / "lora_stats.csv")
        rr = []
        for _, gg in ls.groupby("param"):
            gg = gg.set_index("expert").loc[TASKS]
            rr.append(spearmanr(gg["kappa2_W"], gg["smin_W"]).statistic)
        num_check(f"§1 l.{ln}", "Within-layer ρ(κ, σ_min) = −1.00 (LoRA)", vals[1], float(np.mean(rr)), 2, "results/aa/lora/lora_stats.csv (outside the scope list; mean over 48 maps of Spearman(kappa2_W, smin_W))")
    # §1 tail stress summary
    vals, ln = rx(R, r"moves λ_expert from ([\d.]+) to ([\d.]+)–([\d.]+) and costs up to (\d+) pt", "λ_expert from 0.13 to 0.45–0.69, up to 8 pt", "§1")
    if vals:
        p4, pinf = pair("svhn", 4.0), pair("svhn", math.inf)
        num_check(f"§1 l.{ln}", "λ_expert before (SVHN, b=4)", vals[0], p4["lam0"], 2, "tail_stress_svhn.json")
        num_check(f"§1 l.{ln}", "λ_expert after, lower end (SVHN b=4)", vals[1], p4["lam1"], 2, "tail_stress_svhn.json")
        allafter = [p["lam1"] for p in pairs if abs(p["alpha"] - 1.6) < 1e-9 and p["expert"] == "svhn"]
        num_check(f"§1 l.{ln}", "λ_expert after, upper end (SVHN b=inf)", vals[2], max(allafter), 2, "tail_stress_svhn.json", note="0.6849 = 50/73")
        drops = [-p["d_all"] for p in pairs if abs(p["alpha"] - 1.6) < 1e-9]
        num_check(f"§1 l.{ln}", "costs up to 8 pt", vals[3], max(drops), 0, "tail_stress_*.json (largest drop of the 8-task mean, alpha 1.6)")
    vals, ln = rx(R, r"Oracle \(D\): full-FT \+(\d+) pt dev \(test ([\d.]+)\); LoRA \+(\d+) pt \(test ([\d.]+)\); global per-expert weights ≤ \+([\d.]+) pt", "Oracle headroom", "§1")
    if vals:
        fd = 100 * (fits[("full-FT", "layerwise_alpha", 0)]["dev"]["avg_acc"] - uni["full-FT"]["dev"])
        num_check(f"§1 l.{ln}", "full-FT oracle gain +8 pt dev", vals[0], fd, 0, "layerwise_alpha_seed0 vs uniform grid")
        num_check(f"§1 l.{ln}", "full-FT oracle test 77.2", vals[1], 100 * fits[("full-FT", "layerwise_alpha", 0)]["test"]["avg_acc"], 1, "layerwise_alpha_seed0.json",
                  others={"seed 1": 100 * fits[("full-FT", "layerwise_alpha", 1)]["test"]["avg_acc"]})
        ld = 100 * (fits[("LoRA", "layerwise_alpha", 0)]["dev"]["avg_acc"] - uni["LoRA"]["dev"])
        lt = 100 * (fits[("LoRA", "layerwise_alpha", 0)]["test"]["avg_acc"] - ext["LoRA"]["mean_acc"])
        num_check(f"§1 l.{ln}", "LoRA oracle gain +12 pt (dev)", vals[2], ld, 0, "layerwise_alpha_seed0 vs uniform grid", note=f"test gain is {lt:+.2f} pt")
        num_check(f"§1 l.{ln}", "LoRA oracle test 64.1", vals[3], 100 * fits[("LoRA", "layerwise_alpha", 0)]["test"]["avg_acc"], 1, "layerwise_alpha_seed0.json")
        gd = [100 * (fits[(b_, "global_alpha", 0)]["dev"]["avg_acc"] - uni[b_]["dev"]) for b_ in ("full-FT", "LoRA")]
        gt = [100 * (fits[(b_, "global_alpha", sd)]["test"]["avg_acc"] - ext[b_]["mean_acc"]) for b_ in ("full-FT", "LoRA") for sd in (0, 1) if (b_, "global_alpha", sd) in fits]
        bound_check(f"§1 l.{ln}", "global per-expert weights ≤ +0.4 pt (DEV gain, both banks, seed 0)", vals[4], max(gd), 1, "global_alpha_seed0.json vs uniform grid", note=f"dev gains full-FT {gd[0]:+.2f}, LoRA {gd[1]:+.2f}")
        bound_check(f"§1 l.{ln}", "same claim on TEST (global_alpha, seeds, both banks)", vals[4], max(gt), 1, "global_alpha_seed*.json vs frozen-test uniform", note="gains on test: full-FT " + ", ".join(f"{x:+.2f}" for x in gt[:2]) + "; LoRA " + f"{gt[2]:+.2f}; the claim does not name a split: true on dev, false on test")
    vals, ln = rx(R, r"dev best ([\d.]+) = shuffled ([\d.]+) ± ([\d.]+), uniform ([\d.]+)\. Frozen TEST: κ ([\d.]+) vs uniform ([\d.]+) \(normalized\)", "Master dev/test numbers", "§1")
    if vals:
        fz = {r["cfg"]["tag"]: r for r in jsonl(WT / "results" / "freeze" / "test.jsonl")}
        num_check(f"§1 l.{ln}", "master: dev best 77.74 (kappa|tuned)", vals[0], 100 * fz["kappa|tuned"]["dev_metric"], 2, "results/freeze/test.jsonl dev_metric (master's file, outside scope)")
        num_check(f"§1 l.{ln}", "master: uniform dev 77.65", vals[3], 100 * fz["uniform|tuned"]["dev_metric"], 2, "results/freeze/test.jsonl dev_metric")
        num_check(f"§1 l.{ln}", "master: frozen TEST kappa 76.61 (normalized)", vals[4], 100 * fz["kappa|tuned"]["metrics"]["mean_norm"], 2, "results/freeze/test.jsonl mean_norm")
        num_check(f"§1 l.{ln}", "master: frozen TEST uniform 76.68 (normalized)", vals[5], 100 * fz["uniform|tuned"]["metrics"]["mean_norm"], 2, "results/freeze/test.jsonl mean_norm")
        UNVER.append(("§1", "shuffled-score null 77.74 ± 0.19", "produced by master's shuffle sweeps (results/sweeps/*), outside the scope of this table set"))
    # §1: update shape +9
    vals, ln = rx(R, r"even with norms matched \(\+(\d+) pt\)", "+9 pt with norms matched", "§1")
    if vals:
        fn = POOL["A-nm"]
        num_check(f"§1 l.{ln}", "update-spread rule reaches the oracle with norms matched (+9 pt over uniform)", vals[0], smean(fn, "dsrank", 4.0) - smean(fn, "uniform", 0.0), 0, "A-nm: dsrank b=4 minus uniform")
    # §2: seeds
    vals, ln = rx(R, r"log-λ Spearman ([\d.]+) between seeds", "log-λ Spearman 0.98 between seeds", "§2")
    if vals:
        for v_ in ("layerwise_alpha", "layerwise_a1"):
            num_check(f"§2 l.{ln}", f"log-λ Spearman between seeds ({v_})", vals[0], V["seed_agree"][v_]["sp_all"], 2, "oracle_lambda/*_lambda.csv (= analysis_oracle_vs_kappa.json)")
    # §2: independent kappa implementations
    UNVER.append(("§2", "Two independent κ implementations agree to 1e-13 median", "no result file or script in the aa results; not checked"))
    # §3 bullets
    vals, ln = rx(R, r"\(n_eff (\d)–([\d.]+)\)", "attention maps nearly one expert per map (n_eff 1–2.5)", "§3")
    if vals:
        nf = [V["neff"][("full-FT", v_, sd)] for v_ in ("layerwise_a1", "layerwise_alpha") for sd in (0, 1)]
        lo, hi = min(e["attn_min"] for e in nf), max(e["attn_max"] for e in nf)
        tm = [e[c] for e in nf for c in ("q_proj", "k_proj", "v_proj", "out_proj")]
        add(f"§3 l.{ln}", "attention maps nearly one expert per map (n_eff 1–2.5)", f"{vals[0]:g}–{vals[1]:g}",
            f"per-map n_eff in attention maps: {lo:.2f} to {hi:.2f} (max per fit " + ", ".join(f"{e['attn_max']:.2f}" for e in nf) + f"); type means {min(tm):.2f}-{max(tm):.2f}",
            "IMPRECISE" if hi > vals[1] + 0.05 else "OK", "oracle_lambda/*_lambda.csv (B3e, B3f)", "the type means are within 1-2.5; the largest single-map n_eff is above 2.5")
        sa = V["seed_agree"]
        add(f"§3 l.{ln}", "'with the same assignment across seeds' (attention)", "same assignment", f"same top expert in {sa['layerwise_alpha']['att']}/{sa['layerwise_alpha']['n_att']} (layerwise_alpha) and {sa['layerwise_a1']['att']}/{sa['layerwise_a1']['n_att']} (layerwise_a1) attention maps; MLP {sa['layerwise_alpha']['mlp']}/{sa['layerwise_alpha']['n_mlp']} and {sa['layerwise_a1']['mlp']}/{sa['layerwise_a1']['n_mlp']}",
            f"OK (approximately: {100 * min(sa[v]['att'] / sa[v]['n_att'] for v in sa):.0f}-{100 * max(sa[v]['att'] / sa[v]['n_att'] for v in sa):.0f}% of attention maps)", "B3d")
    vals, ln = rx(R, r"\(LoRA within-map ρ ([+−]?\d\.\d+)\)", "LoRA within-map ρ +0.03", "§3")
    if vals:
        for t in ("layerwise_a1_seed0", "layerwise_alpha_seed0"):
            num_check(f"§3 l.{ln}", f"LoRA within-map ρ(kappa, oracle) {t}", vals[0], V["lora_oracle_ana"][t]["kappa_W"]["within_map"], 2, "results/aa/lora/oracle_vs_scores.json")
    vals, ln = rx(R, r"reaches (\d+\.\d+) absolute on test vs (\d+\.\d+) for\s+tuned\s+uniform\. So no global rule can gain more than about \+(\d+\.\d+)", "global oracle 69.8 vs tuned uniform 69.3; gain bound +0.5", "§3")
    if vals:
        num_check(f"§3 l.{ln}", "global oracle test 69.8", vals[0], 100 * fits[("full-FT", "global_alpha", 0)]["test"]["avg_acc"], 1, "global_alpha_seed0.json", others={"seed 1": 100 * fits[("full-FT", "global_alpha", 1)]["test"]["avg_acc"]})
        num_check(f"§3 l.{ln}", "tuned uniform test 69.3", vals[1], 100 * ext["full-FT"]["mean_acc"], 1, ext["full-FT"]["src"])
        num_check(f"§3 l.{ln}", "no global rule can gain more than about +0.5", vals[2], 100 * (fits[("full-FT", "global_alpha", 0)]["test"]["avg_acc"] - ext["full-FT"]["mean_acc"]), 1, "global_alpha_seed0.json vs frozen-test uniform",
                  others={"seed 1": 100 * (fits[("full-FT", "global_alpha", 1)]["test"]["avg_acc"] - ext["full-FT"]["mean_acc"])})
    # §4 text
    vals, ln = rx(R, r"κ falls with noise in about (\d+)% of maps", "κ falls with noise in about 70% of maps for all experts", "§4")
    if vals:
        mm = V["soup_metaA_mean"]["partA_controlled_noise_strong_v2"]
        add(f"§4 l.{ln}", "κ falls with noise in about 70% of maps for all experts", f"about {vals[0]:g}%", f"mean {100 * mm['frac']:.1f}%, range over experts {100 * mm['frac_min']:.1f}-{100 * mm['frac_max']:.1f}% (A-strong); first run (mild noise) mean {100 * V['soup_metaA_mean']['partA_controlled_noise']['frac']:.1f}%",
            f"OK (mean {100 * mm['frac']:.1f}%; individual experts {100 * mm['frac_min']:.0f}-{100 * mm['frac_max']:.0f}%)", "B5b")
    m = re.search(r"and its argmin selects the noisiest copy", R)
    if m:
        lamA = V["soup"][POOL["A"]]["lam"][("kappa", math.inf)]
        wn = np.array([lamA[t][-1] for t in TASKS])
        wc = np.array([lamA[t][0] for t in TASKS])
        add(f"§4 l.{line_of(R, m.start())}", "the literal rule's argmin selects the noisiest copy", "the noisiest copy", f"A-strong, kappa b=inf: mean share of maps won by the noisiest copy {wn.mean():.3f} (per task {wn.min():.3f}-{wn.max():.3f}), by the cleanest {wc.mean():.3f}; chance 0.250",
            ("OK (majority of maps: the noisiest copy wins " + f"{100 * wn.min():.0f}-{100 * wn.max():.0f}% of the maps per task vs 25% by chance)") if wn.min() > 0.5 else "IMPRECISE (not a majority in every task)", "partA_controlled_noise_strong_v2.jsonl (B5f1)")
    m = re.search(r"It is too flat when scales differ \(A-nm\)", R)
    if m:
        lm = V["soup"][POOL["A-nm"]]["lam"]
        mpl = np.mean([lm[("mp_invvar", 1.0)][t] for t in TASKS], axis=0)
        orl = np.mean([lm[("oracle_invvar", None)][t] for t in TASKS], axis=0)
        add(f"§4 l.{line_of(R, m.start())}", "tuning-free GD/MP rule is too flat when signal scales differ (A-nm)", "too flat", f"A-nm weights on the 4 copies (cleanest to noisiest): mp_invvar {', '.join(f'{x:.3f}' for x in mpl)} vs oracle_invvar {', '.join(f'{x:.3f}' for x in orl)}",
            "OK" if mpl[0] < 0.5 * orl[0] else "MISMATCH", "partA_controlled_noise_strong_normmatched.jsonl (B5f2)")
    vals, ln = rx(R, r"update spread at ([+−]?[\d.]+), so diffuse means better, and with mean\s+log κ inconsistently \(([−+]?\d\.\d+) … ([−+]?\d\.\d+)\)", "quality vs spread +0.5; vs log κ −0.31…+0.60", "§4")
    if vals:
        b_ = V["soup_B_corr"]
        num_check(f"§4 l.{ln}", "quality correlates with update spread at +0.5 (mean over tasks)", vals[0], b_["srank_mean"], 1, "partB_meta_v2.json")
        num_check(f"§4 l.{ln}", "quality vs mean log κ: lowest −0.31", vals[1], b_["logk_min"], 2, "partB_meta_v2.json")
        num_check(f"§4 l.{ln}", "quality vs mean log κ: highest +0.60", vals[2], b_["logk_max"], 2, "partB_meta_v2.json")
    vals, ln = rx(R, r"uniform soup is already near the D-oracle \(\+([\d.]+)\)", "uniform soup near the D-oracle (+0.6)", "§4")
    if vals:
        fn = POOL["B"]
        num_check(f"§4 l.{ln}", "uniform soup is already near the D-oracle (+0.6)", vals[0], smean(fn, "oracle_quality_softmax100", None) - smean(fn, "uniform", 0.0), 1, "B-v2: oracle_quality_softmax100 minus uniform",
                  note="the report subtracts the rounded 91.6 - 91.0")
    vals, ln = rx(R, r"noise rule hurts \(([\d.]+), (\d)/8 tasks above\s+uniform\)", "noise rule hurts (90.1, 0/8 tasks above uniform)", "§4")
    if vals:
        fn = POOL["B"]
        num_check(f"§4 l.{ln}", "mp_invvar on B: 90.1", vals[0], smean(fn, "mp_invvar", 1.0), 1, "partB_lora_variants_v2.jsonl")
        add(f"§4 l.{ln}", "mp_invvar on B: 0/8 tasks above uniform", f"{int(vals[1])}/8", f"{swins(fn, 'mp_invvar', 1.0)}/8", "OK" if swins(fn, "mp_invvar", 1.0) == int(vals[1]) else "MISMATCH", "partB_lora_variants_v2.jsonl")
    m = re.search(r"This replicates dev exactly\.", R)
    if m:
        diffs, winchg = [], []
        for pool, devfn in (("A_strong", POOL["A"]), ("A_strong_normmatched", POOL["A-nm"]), ("B", POOL["B"])):
            for lab_, key_ in (("uniform", ("uniform", 0.0)), ("kappa b=1", ("kappa", 1.0)), ("kappa b=4", ("kappa", 4.0)), ("dsrank b=4", ("dsrank", 4.0)), ("mp_invvar", ("mp_invvar", 1.0))):
                dm, tm_ = smean(devfn, *key_), tmean(pool, lab_)
                diffs.append((tm_ - dm, pool, lab_))
                if lab_ != "uniform":
                    dw, tw = swins(devfn, *key_), twins(pool, lab_)
                    if dw != tw:
                        winchg.append(f"{TN[pool]} {lab_}: dev {dw}/8 -> test {tw}/8")
        lo_, hi_ = min(diffs)[0], max(diffs)[0]
        add(f"§4 l.{line_of(R, m.start())}", "Frozen TEST: 'This replicates dev exactly.'", "exact replication", f"test minus dev mean accuracy for the 15 frozen configurations shown in the report: {lo_:+.2f} to {hi_:+.2f} pt; win counts that change: " + "; ".join(winchg),
            "IMPRECISE (qualitative pattern replicates, numbers do not match exactly)", "B5g (dev and test side by side)")
    # §5 text
    vals, ln = rx(R, r"The edit is ([\d.]+)% of\s+‖Δ‖ \(median\), and the expert's accuracy is unchanged \(≤ ([\d.]+) pt on every task\)", "edit 0.47% of ‖Δ‖; ≤ 0.08 pt", "§5")
    if vals:
        num_check(f"§5 l.{ln}", "edit is 0.47% of ‖Δ‖ (median) -- SVHN", vals[0], 100 * V["tail"]["svhn"]["rel_edit_median"], 2, "tail_stress_svhn.json", note=f"SUN397 edit: {100 * V['tail']['sun397']['rel_edit_median']:.2f}%")
        mx_ = max(V["tail_alone_maxabs"].values())
        bound_check(f"§5 l.{ln}", "expert accuracy unchanged (≤ 0.08 pt on every task)", vals[1], mx_, 2, "tail_stress_*.json expert_alone_before/after", note=f"largest change {mx_:.3f} pt")
    # §8/other: omissions flagged separately in the text notes
    # ---------- math note
    m = re.search(r"layer-specific association is\s+\+([\d.]+) \(p ([\d.]+)\)", M)
    if m:
        num_check(f"math note l.{line_of(M, m.start())}", "layer-specific association +0.095", float(m.group(1)), V["gp_dec"]["1"]["kappa_W"]["within_expert_corr"], 3, "decompose_pref.json")
        num_check(f"math note l.{line_of(M, m.start())}", "its p 0.017 (one-sided, null >= obs)", float(m.group(2)), V["gp_dec"]["1"]["kappa_W"]["p_within_ge"], 3, "decompose_pref.json p_within_ge (the report/§2 quotes the two-sided 0.03)")
    m = re.search(r"Spearman\(κ_i, σ_q\(W_i\)\) = ([−-]\d\.\d+)", M)
    if m:
        num_check(f"math note l.{line_of(M, m.start())}", "within-layer Spearman(κ_i, σ_q(W_i)) = −0.91", float(m.group(1).replace("−", "-")), V["gp_ana"]["1"]["spearman_kappa_vs_sminW"], 2, "analysis_grad_vs_kappa.json")
    m = re.search(r"(\d+)% of its first-order effect on dev loss comes from 5 maps", M)
    if m:
        a1_, a24_ = V["gp_ana"]["1"]["kappa"]["frac_dL_from_top5"], V["gp_ana"]["2.4"]["kappa"]["frac_dL_from_top5"]
        add(f"math note l.{line_of(M, m.start())}", "100% of its first-order effect on dev loss comes from 5 maps", f"{m.group(1)}%", f"{100 * a1_:.2f}% (alpha 1), {100 * a24_:.2f}% (alpha 2.4)", "ROUNDING", "analysis_grad_vs_kappa.json frac_dL_from_top5")
    m = re.search(r"κ 6\.7e6 to 1\.5e3 as the noise\s+goes from 1e-6 to 1e-2", M)
    if m:
        c11 = V["synth_c11"]
        add(f"math note l.{line_of(M, m.start())}", "128x128 matrix: κ 6.7e6 to 1.5e3 as noise goes 1e-6 to 1e-2", "6.7e6 -> 1.5e3", f"{c11['k_s1e6']:.3g} -> {c11['k_s1e2']:.3g}", "OK" if (f"{c11['k_s1e6']:.2g}" == "6.7e+06" and f"{c11['k_s1e2']:.2g}" == "1.5e+03") else "MISMATCH", "synthetic_checks.json C11")
    m = re.search(r"L0 k_proj goes from 1\.04e8 to 9\.9e3 at noise std 1e-2\. Even the\s+tall, well-conditioned L6 fc1 goes from 14\.4 to 7\.5", M)
    if m:
        c11 = V["synth_c11"]
        ok = (f"{c11['l0k_k0']:.3g}" == "1.04e+08" and f"{c11['l0k_s1e2']:.2g}" == "9.9e+03" and f"{c11['l6fc1_k0']:.3g}" == "14.4" and f"{c11['l6fc1_s1e2']:.2g}" == "7.5")
        add(f"math note l.{line_of(M, m.start())}", "L0 k_proj 1.04e8 -> 9.9e3; L6 fc1 14.4 -> 7.5", "1.04e8 -> 9.9e3; 14.4 -> 7.5", f"{c11['l0k_k0']:.3g} -> {c11['l0k_s1e2']:.3g}; {c11['l6fc1_k0']:.3g} -> {c11['l6fc1_s1e2']:.3g}", "OK" if ok else "MISMATCH", "synthetic_checks.json C11_real_base")
    m = re.search(r"κ falls with the noise level in (\d+)% of the maps", M)
    if m:
        mm0, mm1 = V["soup_metaA_mean"]["partA_controlled_noise"], V["soup_metaA_mean"]["partA_controlled_noise_strong_v2"]
        add(f"math note l.{line_of(M, m.start())}", "noisy copies: κ falls with the noise level in 75% of the maps", f"{m.group(1)}%",
            f"mean over experts {100 * mm0['frac']:.1f}% (mild noise run), {100 * mm1['frac']:.1f}% (strong run); max over experts {100 * mm0['frac_max']:.1f}% / {100 * mm1['frac_max']:.1f}%", "IMPRECISE (75% is the maximum over experts of the mild-noise run)", "partA_meta*.json (B5b)")
    m = re.search(r"stable rank of the noisy copies' updates tracks the noise level with rank correlation \+(\d+\.\d+)", M)
    if m:
        add(f"math note l.{line_of(M, m.start())}", "stable rank of the updates tracks the noise level with rank correlation +1.00", f"+{m.group(1)}",
            f"{V['soup_metaA_mean']['partA_controlled_noise']['srank']:+.3f} (mild-noise run), {V['soup_metaA_mean']['partA_controlled_noise_strong_v2']['srank']:+.3f} (strong run, mean over experts)", f"OK (mild-noise run; {min(d_['mean_rank_corr_srank_vs_noise'] for d_ in V['soup_meta']['partA_controlled_noise_strong_v2']['per'].values()):.2f}-{max(d_['mean_rank_corr_srank_vs_noise'] for d_ in V['soup_meta']['partA_controlled_noise_strong_v2']['per'].values()):.2f} in the strong run)", "partA_meta*.json (B5b)")
    m = re.search(r"its global variant puts ([\d.]+)–([\d.]+) there", M)
    if m:
        gl = V["soup"]["partA_controlled_noise_strong_v2"]["lam"]
        txt = "; ".join(f"b={fb(b)}: {np.mean([gl[('global:logkappa', b)][t][-1] for t in TASKS]):.2f}" for b in (1.0, 2.0, 4.0, 8.0, 16.0, 128.0))
        add(f"math note l.{line_of(M, m.start())}", "the literal rule's global variant puts 0.73-1.0 on the noisiest copy", f"{m.group(1)}-{m.group(2)}", "global:logkappa weight on the noisiest copy (A-strong, mean over tasks): " + txt, "DEPENDS ON b (not reproducible as one range)", "partA_controlled_noise_strong_v2.jsonl (B5f1)")

    # ---------- emit
    stat_counts = defaultdict(int)
    for r in rows_out:
        stat_counts[kind_of(r[5])] += 1
    V["x_rows"] = rows_out
    V["x_unver"] = UNVER
    V["x_counts"] = dict(stat_counts)
    add_table("X1", "Cross-check of every number of the report that lives in this scope (recomputed from the raw files)",
              "For each quoted number of docs/aa/AA_SIDE_REPORT.md (sections 1-5) and docs/aa/MATH_kappa_of_finetuned_weights.md the value is parsed out of the report text by the script and compared with the value recomputed from the result files; "
              "metric and units are those of the quoted number (accuracy in %, absolute unless stated; correlations dimensionless).",
              [rel(rep_path), rel(math_path)],
              ["#", "report location", "claim (as written)", "reported", "recomputed", "status", "source of the recomputed value / note"], rows_out, align="rllllll",
              legend=["status OK = the recomputed value rounds to the reported number at the reported precision. ROUNDING = differs by one unit of the last reported digit (the report rounds a rounded number, or uses another seed). MISMATCH = larger difference. "
                      "IMPRECISE / NOT UNIQUE / DEPENDS ON b = the statement is qualitatively right but its stated range or exclusivity does not match the data.",
                      "Where several candidates exist (seeds, rules at the same b) all are listed; the first (seed 0 / the named rule) is the primary comparison. Test accuracy of tuned uniform comes from the frozen-test files named in the source column.",
                      "Status counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(stat_counts.items(), key=lambda kv: -kv[1]))])


def kind_of(st: str) -> str:
    for k in ("OK", "ROUNDING", "MISMATCH", "IMPRECISE", "PARTIAL", "NOT UNIQUE", "DEPENDS ON b", "DIFFERS", "CLAIM TEXT NOT FOUND"):
        if st.strip().startswith(k):
            return k
    return st.split()[0]


def build_X2():
    """Discrepancy list: every non-OK row of X1 plus omissions found in the files (all numbers computed)."""
    order = ["MISMATCH", "PARTIAL", "NOT UNIQUE", "IMPRECISE", "DEPENDS ON b", "DIFFERS", "CLAIM TEXT NOT FOUND", "ROUNDING"]
    xr = [r for r in V["x_rows"] if kind_of(r[5]) != "OK"]
    xr.sort(key=lambda r: (order.index(kind_of(r[5])), r[0]))
    rows_, n = [], 0
    for r in xr:
        n += 1
        rows_.append([n, kind_of(r[5]), r[1], r[2], r[3], r[4], f"X1 #{r[0]}; status text: {r[5]}; source: {r[6]}"])
    S = V["soup"]
    nmfn, afn, bfn = "partA_controlled_noise_strong_normmatched", "partA_controlled_noise_strong_v2", "partB_lora_variants_v2"

    def sm(fn, rule, b):
        v = S[fn]["acc"][(rule, b)]
        return 100 * float(np.mean([v[t] for t in TASKS]))

    def sw(fn, rule, b):
        return wins_of(S[fn]["acc"][(rule, b)], S[fn]["acc"][("uniform", 0.0)])

    om = []
    om.append(("§4 (A-nm column)", "reversed kappa rule (kappa_rev, hard selection b=inf = highest-kappa copy) is not discussed",
               "–", f"A-nm: kappa_rev b=inf {sm(nmfn, 'kappa_rev', math.inf):.2f} vs uniform {sm(nmfn, 'uniform', 0.0):.2f} ({sw(nmfn, 'kappa_rev', math.inf)}/8 tasks above uniform); A-strong: {sm(afn, 'kappa_rev', math.inf):.2f} vs {sm(afn, 'uniform', 0.0):.2f} ({sw(afn, 'kappa_rev', math.inf)}/8); B: {sm(bfn, 'kappa_rev', math.inf):.2f} vs {sm(bfn, 'uniform', 0.0):.2f} ({sw(bfn, 'kappa_rev', math.inf)}/8)",
               "B5d3, B5e3"))
    tp = [p for p in V["tail_pairs"] if abs(p["alpha"] - 1.0) < 1e-9]
    om.append(("§5", "tail-stress rows at alpha = 1.0 (b = 4) are in the JSON files but not in the report table",
               "–", "; ".join(f"{p['expert']}: {100 * p['acc0']:.2f} -> {100 * p['acc1']:.2f} ({p['d_all']:+.2f} pt), lambda {p['lam0']:.3f} -> {p['lam1']:.3f}" for p in tp) + " (the SVHN edit does not lower the 8-task accuracy at alpha = 1)", "B6c, B6d"))
    big = V["ds"][V["ds"].kappa_WC > 1e12]
    om.append(("§2 table row 'κ(W C^{1/2}) (budget B)'", "the budget-B kappa statistic contains round-off values", "–",
               f"{len(big)} of 584 (map, expert) rows have kappa_WC > 1e12 (maps: {', '.join(short(x) for x in big.param.unique())}); the within-map ranks of those maps are determined by round-off", "B7e"))
    pb = V["soup"]["partB_lora_variants"]
    om.append(("results/aa/soup/partB_lora_variants.jsonl, summary.md", "the first B run is incomplete and still listed in summary.md/json as '(2 tasks)'", "–",
               f"{len(pb['tasks'])} tasks present; {len(pb['rows'])} rows; superseded by partB_lora_variants_v2.jsonl (identical on the common rows)", "B5a"))
    om.append(("results/aa/oracle_lambda/analysis_oracle_vs_kappa.json", "contains only the 4 layerwise fits and a seed_agreement block for the layerwise families", "–",
               "no entry for global_alpha_seed0/1 (their lambda CSVs and JSONs exist); the global fits are summarised from the CSVs in B3d/B3g", "B3d, B3g, B3h"))
    om.append(("§4 'update norm, b = 4' (A-nm: 78.9)", "dnorm in the norm-matched pool is a round-off-driven pseudo-random choice, not a signal",
               "–", f"max deviation of the mean lambda of dnorm from uniform: {V['soup_dnorm_dev'][0]:.3f} in A-nm vs {V['soup_dnorm_dev'][1]:.3f} in A-strong; accuracy at b=4 is {sm(nmfn, 'dnorm', 4.0):.2f} (< uniform {sm(nmfn, 'uniform', 0.0):.2f})", "B5a legend, B5d3, B5f2"))
    rows2 = []
    for o in om:
        n += 1
        rows2.append([n, "OMISSION / NOTE", o[0], o[1], o[2], o[3], o[4]])
    rows_ += rows2
    add_table("X2", "Discrepancies between the report and the files (all non-OK rows of X1, plus omissions found while building B1-B7)",
              "Collects the rows of X1 whose status is not OK and notes on results that exist in the files but are absent from or contradicted by the report; numbers as in X1 (accuracy in %, correlations dimensionless).",
              [rel(WT / "docs" / "aa" / "AA_SIDE_REPORT.md"), rel(WT / "docs" / "aa" / "MATH_kappa_of_finetuned_weights.md"), "tables B1-B7 of this document"],
              ["#", "kind", "report / file location", "claim or note", "reported", "recomputed / found", "where"], rows_, align="rllllll",
              legend=["kinds: MISMATCH = the reported number is not supported (beyond rounding); ROUNDING = off by one unit of the last digit (rounded-number arithmetic, another seed, or the report rounding up a 3-decimal value); "
                      "IMPRECISE = the statement is qualitatively right but its range/wording is looser than the data; PARTIAL / NOT UNIQUE / DEPENDS ON b = holds only for some targets / not exclusively / only for some temperatures; OMISSION / NOTE = a result in the files that the report does not mention.",
                      "rows 'DIFFERS' would mark a p-value of the section 2b re-derivation that deviates by more than 0.03 from the report (none unless listed)."])
    # not verifiable
    unv = list(V["x_unver"]) + [("§6, §9", "LoRA selection/replication numbers and the RegMean results", "outside this table set (results/aa/lora/*); only the LoRA oracle rows and the LoRA rho(kappa, sigma_min) were checked here")]
    add_table("X3", "Claims of the report that cannot be verified from the files in this scope",
              "Lists the report statements for which no result file or script exists in the scope (or which are outside it); no numbers are produced.",
              [rel(WT / "docs" / "aa" / "AA_SIDE_REPORT.md")], ["location", "statement", "why not verified"], [list(u) for u in unv], align="lll")


# ==================================================================== assembly
def assemble():
    import platform
    import time
    n_dev = jload(WT / "results" / "sanity" / "expert_matrix_dev.json")["n"]
    assert list(n_dev) == TASKS
    head = [
        "# Appendix tables B1-B7: diagnostics of the aa/side track (condition-number merging)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `make_tables.py` (python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}); "
        f"every number below is computed by that script from the raw result files under `{rel(AA)}` (worktree `{WT}`, branch aa/side, read-only). "
        "Source paths in each table are relative to the worktree root.",
        "",
        "## Conventions used in all tables",
        "",
        "- **Rule.** lambda_i^p = softmax_i(-beta * kappa_2(W_i^p)) per linear map p, merged weight W0 + alpha * sum_i lambda_i Delta_i with Delta_i = W_i - W0; alpha = 1 is the normalised average. "
        "beta = b / scale with scale = median over maps of the across-expert std of the rule's score; b in {" + ", ".join(fb(b_) for b_ in BGRID) + "}; b = inf is per-map argmin selection; b = 0 is uniform.",
        "- **Banks.** Full fine-tuning: 8 FusionBench CLIP ViT-B/32 experts (sun397, cars, resisc45, eurosat, svhn, gtsrb, mnist, dtd), 73 linear maps (12 x q,k,v,out_proj + 12 x fc1,fc2 + patch embedding; biases tied to their weight, LayerNorm/embeddings get the prior). "
        "LoRA: KnOTS rank-16 bank, 48 adapted attention maps (q,k,v,out of 12 blocks) unless a table says another bank (the soup part B uses 13 same-task LoRA variants per task).",
        "- **Splits.** dev = 20% of each official test set (permutation seed 0), used for all fitting/selection/diagnostics (n per task: " + ", ".join(f"{t} {n_dev[t]}" for t in TASKS) +
        f"; total {sum(n_dev.values())}; source `results/sanity/expert_matrix_dev.json`, master's file). test = the other 80%, touched only for frozen configurations and for the held-out evaluation of oracle fits. "
        "Oracle fits (B3) use dev labels, so their dev numbers are in-sample.",
        "- **Information budgets.** A = weights only; B = pooled unlabeled inputs without task IDs; C = task-separated unlabeled inputs; D = labeled dev data (oracle/diagnostic only). Every table states which budget its rule/score uses.",
        "- **Accuracy.** Every accuracy in B1-B7 is ABSOLUTE accuracy (fraction correct on the task's split, shown in %); 'avg acc' = unweighted mean of the 8 per-task accuracies. "
        "NORMALISED accuracy (merged accuracy / own expert's accuracy, averaged) is not used by any file of this scope; it only appears in master's files quoted as external references (B3a legend, X1). "
        "Soup tables (B5) score each soup on its own task only. CE = mean cross-entropy of the CLIP logits scaled by 100, in nats.",
        "- **Correlations.** rho = Spearman rank correlation across the 8 experts inside a map, averaged over maps (within-map); 'between-expert' = Pearson correlation over experts of the mean within-map ranks; "
        "'within-expert' = Pearson correlation of the ranks after removing each expert's mean rank (layer-specific part). Permutation p-values: one-sided where two columns '(null <= obs)' / '(null >= obs)' are given, two-sided where a single p is given; the number of shuffles is in each table's description.",
        "- **Scores.** kappa_W = kappa_2(W_i) = sigma_1/sigma_q; smin_W = sigma_min(W_i); fro_D = ||Delta_i||_F; stable_rank_D, entropy_rank_D, kplus_D_0.01 = spread measures of the update's singular values; "
        "logkappa_rel = log kappa(W_i) - log kappa(W0); B-budget scores use the pooled input second moment C (B7).",
        "- **Map labels** are shortened to L<layer>.<module> (L3.fc1 = encoder.layers.3.mlp.fc1.weight).",
        "- **Reproduction.** `OMP_NUM_THREADS=4 /data/home/azpitarte/lora_merging_ica/.venv/bin/python make_tables.py` (CPU, about 30 seconds); also writes `csv/soup_full_grid.csv` and keeps scratch files in `_work/`.",
        "",
    ]
    idx_rows = [[tid, title, n] for tid, title, n in INDEX]
    head += ["## Index of tables", "", md_table(["ID", "title", "rows"], idx_rows, "llr"), ""]
    # parse notes
    pn = [
        "- Every file listed in the task scope was parsed (JSON, JSONL, CSV, NPZ, MD); no parsing failure. `results/aa/figures/*.png` are outside the scope and were not read.",
        "- `synthetic_checks.json` stores numbers only for C1, C2, C3, C5, C6, C11, C11_real_base and `fails: []`; the PASS/FAIL line of each sub-check and the numbers of C4, C7-C10 come from re-running a copy of `scripts/aa_synthetic_checks.py` "
        f"in the scratchpad (seed 0; its JSON equals the stored one, max relative deviation {V['synthetic_rerun_maxrel']:.1e}). `C11_real_base` could not be re-run (needs a base-weight npz).",
        "- `oracle_lambda/analysis_oracle_vs_kappa.json` has no entry for the global_alpha fits; those are summarised from the lambda CSVs (B3d, B3e, B3g). `oracle_lambda_lora/` has no analysis file: the LoRA oracle correlations are in `results/aa/lora/oracle_vs_scores.json` (B3i), outside the scope list but the only place they are recorded.",
        f"- `soup/partA_meta.json` (first A run) has no 'levels' key: the noise levels were recovered from the oracle_invvar weights and the script default (B5a). `soup/test_freeze_summary.json` is not written by any script of `scripts/`; it is reproduced from `test_freeze.jsonl` (all {V['test_freeze_check'][0]} configurations equal).",
        f"- `score_vs_targets.md` equals `score_vs_targets.json` rendered in the script's format: {V['svt_md_ok']}.",
        f"- The cross-check (X1-X3) parses `docs/aa/AA_SIDE_REPORT.md` as it was at generation time (sha256 {hashlib.sha256((WT / 'docs' / 'aa' / 'AA_SIDE_REPORT.md').read_bytes()).hexdigest()[:16]}..., modified {time.strftime('%H:%M', time.localtime((WT / 'docs' / 'aa' / 'AA_SIDE_REPORT.md').stat().st_mtime))}) "
        f"and `docs/aa/MATH_kappa_of_finetuned_weights.md` (sha256 {hashlib.sha256((WT / 'docs' / 'aa' / 'MATH_kappa_of_finetuned_weights.md').read_bytes()).hexdigest()[:16]}...); the report is being edited by another session (a paragraph was appended to section 9 while this was built; sections 0-5 and 7 are unchanged), so re-run the script to refresh X1-X3.",
        "- Files read outside the scope list, only for cross-checks: `results/freeze/test.jsonl` (tuned-uniform test accuracy, master's numbers), `results/aa/lora/test_freeze.jsonl`, `results/aa/lora/lora_stats.csv`, `results/aa/lora/oracle_vs_scores.json`, "
        "`results/spectra/stats.csv` (kappa_q90 for B4b), `results/sanity/expert_matrix_dev.json` (dev sizes). Nothing in the worktree was modified.",
    ]
    tail = ["## Notes on parsing and provenance", ""] + pn + [""]
    return "\n".join(head + BODY + tail)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    steps = [("B1", build_B1), ("B2", build_B2), ("B3", build_B3), ("B4", build_B4), ("B4b", build_B4x), ("B5", build_B5), ("B6", build_B6), ("B7", build_B7), ("X", build_X), ("X2", build_X2)]
    for name, fn in steps:
        if only and name not in only:
            continue
        # section headings for the main groups
        titles = {"B1": "B1. Synthetic checks", "B2": "B2. Gradient preference vs every score", "B3": "B3. Oracle weights, full fine-tuning and LoRA", "B4": "B4. Score vs targets",
                  "B5": "B5. Shared-target soups", "B6": "B6. Tail-stress test", "B7": "B7. Data-supported score statistics", "X": "X. Cross-check against docs/aa/AA_SIDE_REPORT.md"}
        if name in titles:
            BODY.append(f"## {titles[name]}")
            BODY.append("")
        fn()
    text = assemble()
    (OUT_DIR / "tables.md").write_text(text)
    print(f"wrote {OUT_DIR / 'tables.md'} ({len(text.splitlines())} lines, {len(INDEX)} tables)")
    for tid, title, n in INDEX:
        print(f"{tid:7s} {n:4d}  {title}")


if __name__ == "__main__":
    main()
