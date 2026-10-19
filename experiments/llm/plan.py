"""Experiment plan for the LLM settings: dev grids (weights-only b selection), job specs, frozen selection.

Pure functions used by orchestrate.py (and unit-tested); nothing here touches a GPU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from condmerge.llm import banks as BK  # noqa: E402
from condmerge.llm import jobs as J  # noqa: E402
from condmerge.llm import rules  # noqa: E402

ALPHAS = [0.8, 1.0, 1.3, 1.6, 2.0]
TARGET_FRACS = [0.85, 0.65, 0.45, 0.25]
CONFIGS = ROOT / "configs" / "llm"
SMOKE = J._SMOKE
if SMOKE:  # plumbing test only: tiny grids, separate roots, no git
    ALPHAS = [1.0, 1.6]
    TARGET_FRACS = [0.45]
    CONFIGS = ROOT / "artifacts" / f"llm_smoke/{SMOKE}" / "configs"

SETTINGS = {
    "S1": {"bank": "l31_8b", "T": 8, "kappa_shuffles": 5, "other_shuffles": 2, "matched": list(range(5, 20)),
           "tail": True, "gradpref": True, "dev": True},
    "S2": {"bank": "l32_3b", "T": 8, "kappa_shuffles": 10, "other_shuffles": 3, "matched": list(range(10, 25)),
           "tail": True, "gradpref": True, "dev": True, "transfer_from": "S1"},
    "S3": {"bank": "l32_3b", "T": 20, "kappa_shuffles": 10, "other_shuffles": 3, "matched": list(range(10, 20)),
           "tail": False, "gradpref": False, "dev": True},
    "S4": {"bank": "l31_8b", "T": 20, "dev": False, "transfer_from": "S1", "optional": True, "matched": list(range(0, 5)),
           "tail": False, "gradpref": False},
}
if SMOKE:
    SETTINGS = {"X1": {"bank": "l32_3b", "T": 8, "kappa_shuffles": 1, "other_shuffles": 1, "matched": [1],
                       "tail": True, "gradpref": True, "dev": True},
                "X2": {"bank": "l32_3b", "T": 20, "dev": False, "transfer_from": "X1", "optional": True,
                       "matched": [0], "tail": False, "gradpref": False}}
BASE_RULES = ["kappa", "global:kappa", "logkappa", "kappa_res"]


def fmt_b(b: float) -> float | str:
    return "inf" if math.isinf(b) else float(b)


def coeff_diag(table: np.ndarray, b: float) -> dict:
    lam, info = rules.coefficients(table, b)
    ne = rules.n_eff(lam)
    mx = lam.max(1)
    k = table.shape[1]
    return {"b": fmt_b(b), "mean_n_eff": float(ne.mean()), "median_n_eff": float(np.median(ne)),
            "frac_one_hot": float(np.mean(mx > 0.99)), "frac_near_uniform": float(np.mean(ne > 0.99 * k)),
            "n_eff_quartiles": [float(x) for x in np.quantile(ne, [0.25, 0.5, 0.75])],
            "special_cases": {k2: v for k2, v in info.items() if k2.startswith("n_")}}


def devgrid(setting: str) -> dict:
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    K = len(BK.tasks(bank, T))
    out = {"setting": setting, "bank": bank, "T": T, "K": K, "alphas": ALPHAS,
           "targets": [f * K for f in TARGET_FRACS], "rules": {}}
    for r in BASE_RULES:
        table = rules.score_table(bank, T, r)
        picks = rules.b_for_targets(table, out["targets"])
        bs = [p["b"] for p in picks] + [math.inf]
        out["rules"][r] = {"b": [fmt_b(b) for b in bs], "picks": picks, "scale": rules.scale_of(table),
                           "diagnostics": [coeff_diag(table, b) for b in bs]}
    out["rules"]["kappa_rev"] = {"b": out["rules"]["kappa"]["b"], "note": "kappa's b values (same units)",
                                 "diagnostics": [coeff_diag(rules.score_table(bank, T, "kappa_rev"),
                                                            math.inf if b == "inf" else b)
                                                 for b in out["rules"]["kappa"]["b"]]}
    return out


def dev_jobs(setting: str, grid: dict) -> list[tuple[dict, int]]:
    """(cfg, priority) of every dev evaluation of a setting."""
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    pbase = 10 if bank == "l31_8b" else 4
    jobs = []

    def merge(rule, b, a, prio, group="dev", **kw):
        jobs.append(({"bank": bank, "T": T, "kind": "merge", "rule": rule, "b": b, "alpha": a, "split": "val",
                      "group": group, "setting": setting, **kw}, prio))

    for a in grid["alphas"]:
        merge("uniform", 0.0, a, pbase)
    for rule, prio_off in (("kappa", 0), ("kappa_rev", 1)):
        for b in grid["rules"][rule]["b"]:
            for a in grid["alphas"]:
                merge(rule, b, a, pbase + prio_off)
    for seed in range(s["kappa_shuffles"]):
        for b in grid["rules"]["kappa"]["b"]:
            for a in grid["alphas"]:
                merge(f"shuf{seed}:kappa", b, a, pbase + 2 + (seed >= 2))
    for rule, null in (("global:kappa", "gshuf{}:global:kappa"), ("logkappa", "shuf{}:logkappa"),
                       ("kappa_res", "shuf{}:kappa_res")):
        for b in grid["rules"][rule]["b"]:
            for a in grid["alphas"]:
                merge(rule, b, a, pbase + 4)
                for seed in range(s["other_shuffles"]):
                    merge(null.format(seed), b, a, pbase + 5)
    return jobs


def ref_jobs(setting: str) -> list[tuple[dict, int]]:
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    out = []
    for split, prio in (("val", 0), ("test", 1), ("fresh", 1)):
        out.append(({"bank": bank, "T": T, "kind": "expert", "split": split, "group": "experts", "setting": setting,
                     "full_matrix": bool(split == "val" and bank == "l32_3b")}, prio))
        out.append(({"bank": bank, "T": T, "kind": "base", "split": split, "group": "base", "setting": setting}, prio + 1))
    return out


def recorded_alpha(bank: str, T: int, method: str) -> float | None:
    p = Path(BK.BANKS[bank].lora_dir).parent / f"T{T}" / f"eval_base_{method}.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())[f"{method}|alpha"])


def context_jobs(setting: str, splits=("val", "test", "fresh")) -> list[tuple[dict, int]]:
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    a = recorded_alpha(bank, T, "IsoC")
    if a is None:
        return []
    return [({"bank": bank, "T": T, "kind": "context", "method": "isoc", "alpha": a, "split": sp, "group": "context",
              "setting": setting, "alpha_source": "source project's val selection (eval_base_IsoC.json)"}, 2)
            for sp in splits]


def noise_jobs(setting: str, alpha: float, split: str, prio: int) -> list[tuple[dict, int]]:
    s = SETTINGS[setting]
    out = [({"bank": s["bank"], "T": s["T"], "kind": "merge", "rule": "uniform", "b": 0.0, "alpha": alpha,
             "split": split, "group": "noise", "setting": setting, "jitter_rel": 1e-7, "jitter_seed": seed}, prio)
           for seed in range(3)]
    out.append(({"bank": s["bank"], "T": s["T"], "kind": "merge", "rule": "uniform", "b": 0.0, "alpha": alpha,
                 "split": split, "group": "noise", "setting": setting, "repeat": 1}, prio))
    return out


# ----------------------------------------------------------------------------------------------- selection
def dev_table(setting: str) -> list[dict]:
    s = SETTINGS[setting]
    refs = None
    for r in J.read_results(s["bank"], s["T"], "experts"):
        if r["cfg"]["split"] == "val":
            refs = np.asarray(r["own_local"], np.float64)
    if refs is None:
        raise RuntimeError("no val expert references")
    out = []
    for r in J.read_results(s["bank"], s["T"], "dev"):
        loc = np.asarray(r["metrics"]["local_per_task"], np.float64)
        out.append({"rule": r["cfg"]["rule"], "b": r["cfg"]["b"], "alpha": r["cfg"]["alpha"],
                    "norm": float(np.mean(loc / refs)), "joint": r["metrics"]["joint_macro"],
                    "n_eff": r["rule_info"].get("n_eff_mean"), "key": r["key"]})
    return out


def _bval(b) -> float:
    return math.inf if b in ("inf", math.inf) else float(b)


def best(rows: list[dict], rule: str, alpha: float | None = None) -> dict | None:
    cand = [r for r in rows if r["rule"] == rule and (alpha is None or abs(r["alpha"] - alpha) < 1e-9)]
    if not cand:
        return None
    return sorted(cand, key=lambda r: (-round(r["norm"], 10), _bval(r["b"]), abs(r["alpha"] - 1.0)))[0]


def freeze(setting: str, grid: dict) -> dict:
    """Frozen configurations from the dev records (selection rule of protocol 4.4)."""
    s = SETTINGS[setting]
    rows = dev_table(setting)
    fz = {"setting": setting, "bank": s["bank"], "T": s["T"], "selection": "max dev mean normalised accuracy; ties ->"
          " smaller b, then alpha closer to 1", "configs": []}

    def add(tag, r, extra=None):
        if r is not None:
            fz["configs"].append({"tag": tag, "rule": r["rule"], "b": r["b"], "alpha": r["alpha"],
                                  "dev_norm": r["norm"], "dev_joint": r["joint"], "dev_n_eff": r["n_eff"], **(extra or {})})

    add("uniform|tuned", best(rows, "uniform"))
    add("uniform|alpha1", best(rows, "uniform", 1.0))
    rules_all = sorted({r["rule"] for r in rows if r["rule"] != "uniform"})
    for rule in rules_all:
        add(f"{rule}|tuned", best(rows, rule))
    add("kappa|alpha1", best(rows, "kappa", 1.0))
    # concentrated stress setting for the tail edit: kappa's b with mean n_eff closest to 0.45 K, its best alpha
    kb = [b for b in grid["rules"]["kappa"]["b"] if b != "inf"]
    diag = {d["b"]: d["mean_n_eff"] for d in grid["rules"]["kappa"]["diagnostics"]}
    target = 0.45 * grid["K"]
    bc = min(kb, key=lambda b: abs(diag[b] - target))
    rc = [r for r in rows if r["rule"] == "kappa" and r["b"] == bc]
    if rc:
        add("kappa|concentrated", sorted(rc, key=lambda r: (-r["norm"], abs(r["alpha"] - 1)))[0],
            {"note": "stress setting for the tail edit (mean n_eff ~ 0.45 K)"})
    return fz


def frozen_eval_jobs(setting: str, fz: dict, splits=("test", "fresh")) -> list[tuple[dict, int]]:
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    pbase = 1 if bank == "l31_8b" else 0
    out = []
    for c in fz["configs"]:
        main = c["rule"] in ("uniform", "kappa", "kappa_rev") or c["tag"].startswith("kappa|")
        for sp in splits:
            out.append(({"bank": bank, "T": T, "kind": "merge", "rule": c["rule"], "b": c["b"], "alpha": c["alpha"],
                         "split": sp, "group": "frozen", "setting": setting, "tag": c["tag"]},
                        pbase + (0 if main else 1)))
    kap = next(c for c in fz["configs"] if c["tag"] == "kappa|tuned")
    for seed in s.get("matched", []):
        out.append(({"bank": bank, "T": T, "kind": "merge", "rule": f"shuf{seed}:kappa", "b": kap["b"],
                     "alpha": kap["alpha"], "split": "test", "group": "matched", "setting": setting,
                     "tag": "matched-config shuffle at kappa|tuned"}, pbase + 2))
    return out


def transfer_jobs(setting: str, src_fz: dict, splits=("test", "fresh")) -> list[tuple[dict, int]]:
    """Unchanged (rule, b, alpha) from the source setting's freeze, evaluated on this setting."""
    s = SETTINGS[setting]
    bank, T = s["bank"], s["T"]
    out = []
    keep = ("uniform|tuned", "uniform|alpha1", "kappa|tuned", "kappa|alpha1", "kappa_rev|tuned", "global:kappa|tuned",
            "logkappa|tuned", "kappa_res|tuned")
    for c in src_fz["configs"]:
        if c["tag"] in keep or c["tag"].startswith("shuf"):
            for sp in splits:
                out.append(({"bank": bank, "T": T, "kind": "merge", "rule": c["rule"], "b": c["b"], "alpha": c["alpha"],
                             "split": sp, "group": "transfer", "setting": setting, "tag": c["tag"],
                             "transfer_from": src_fz["setting"]}, 3 if bank == "l31_8b" else 1))
    return out
