"""Unattended orchestration of the LLM study (protocol docs/PROTOCOL_02_llm_followups.md).

State machine, restartable (state in artifacts/llm/orchestrator_state.json; every step is idempotent):
  1. wait for the exact spectra of the needed banks (compute_spectra.py, own tmux session)
  2. manifests + weights-only dev grids -> configs/llm/devgrid_<S>.json, committed BEFORE any evaluation
  3. enqueue expert/base references, Iso-C context (val), dev grids, numerical-noise replicates (val)
  4. per setting, when its dev jobs are finished: freeze -> configs/llm/frozen_<S>.json, committed (+ pushed)
     BEFORE any test / fresh job of that setting is enqueued; then test + fresh + matched shuffles + noise (test)
     + Iso-C test/fresh + gradient-preference diagnostics + tail-edit preparation (CPU) and tail jobs
  5. transfer settings (S2 from S1, optional S4 from S1) once the source is frozen
  6. when the queue is drained: analysis (tables, figures) -> commit + push -> ALL_DONE (workers exit)
It only watches OUR queue and result files; it never looks for GPUs (workers run on fixed cards).
Usage: orchestrate.py [--settings S1,S2,S3,S4] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import plan  # noqa: E402
from condmerge.llm import banks as BK  # noqa: E402
from condmerge.llm import jobs as J  # noqa: E402

STATE = (ROOT / "artifacts" / "llm" / "orchestrator_state.json") if not plan.SMOKE else \
    (ROOT / "artifacts" / f"llm_smoke/{plan.SMOKE}" / "orchestrator_state.json")
LOG = (ROOT / "logs" / "llm" / "orchestrator.log") if not plan.SMOKE else (STATE.parent / "orchestrator.log")
PY = "/data/home/azpitarte/lora_merging_ica/.venv/bin/python"


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def git(*args, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.name=Gabriel Azpitarte", "-c", "user.email=gabriel.azpitarte@icloud.com",
                           *args], cwd=ROOT, capture_output=True, text=True, check=check)


def commit(paths: list[str], msg: str, push: bool) -> None:
    if plan.SMOKE:
        log(f"(smoke) would commit {len(paths)} paths: {msg}")
        return
    git("add", "--", *paths)
    r = git("commit", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    log(f"git commit: {r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr.strip()[:200]}")
    b = subprocess.run(["bash", str(ROOT / "scripts" / "backup.sh")], cwd=ROOT, capture_output=True, text=True)
    log(f"local backup rc={b.returncode} {b.stdout.strip()[-120:]}")
    if push:
        p = subprocess.run(["git", "push", "origin", "master:main"], cwd=ROOT, capture_output=True, text=True)
        log(f"git push rc={p.returncode} {p.stderr.strip()[-200:]}")


def setting_results(setting: str, group: str) -> list[dict]:
    s = plan.SETTINGS[setting]
    return [r for r in J.read_results(s["bank"], s["T"], group) if r["cfg"].get("setting") == setting]


def outstanding(keys: set[str]) -> int:
    """Number of the given job keys still pending or running."""
    d = J.dirs()
    live = {p.name.split("_", 1)[1].split(".json")[0] for sub in ("pending", "running") for p in d[sub].glob("*.json*")}
    return len(keys & live)


def failed_keys() -> dict[str, Path]:
    d = J.dirs()
    return {p.name.split("_", 1)[1].split(".json")[0]: p for p in d["failed"].glob("*.json")}


def enqueue_all(jobs: list[tuple[dict, int]]) -> set[str]:
    keys = set()
    for cfg, prio in jobs:
        J.enqueue(cfg, prio)
        keys.add(J.key_of(cfg))
    return keys


def retry_failed(keys: set[str], st: dict) -> None:
    """Re-enqueue failed jobs among keys once (a second failure is final and logged)."""
    fk = failed_keys()
    for k in keys & set(fk):
        n = st.setdefault("retries", {}).get(k, 0)
        if n >= 1:
            continue
        spec = json.loads(fk[k].read_text())
        log(f"retrying failed job {k}: {spec['cfg'].get('kind')} {spec['cfg'].get('rule', '')} "
            f"({(spec.get('error') or '')[-160:]!r})")
        fk[k].rename(fk[k].with_suffix(".json.failed1"))
        J.enqueue(spec["cfg"], spec["prio"])
        st["retries"][k] = n + 1
        save_state(st)


def finished(keys: set[str], st: dict) -> bool:
    retry_failed(keys, st)
    return outstanding(keys) == 0


# ----------------------------------------------------------------------------------------------- phases
def phase_grids(settings: list[str], st: dict, push: bool) -> None:
    if st.get("grids_done"):
        return
    paths = []
    (plan.CONFIGS).mkdir(parents=True, exist_ok=True)
    for sname in settings:
        s = plan.SETTINGS[sname]
        mpath = J.RESULTS / f"manifest_{s['bank']}_T{s['T']}.json"
        if not mpath.exists():
            man = BK.manifest(s["bank"], s["T"], hash_weights=not plan.SMOKE)
            man["split_sizes"] = BK.split_sizes(s["bank"], s["T"])
            mpath.parent.mkdir(parents=True, exist_ok=True)
            mpath.write_text(json.dumps(man, indent=1))
            log(f"manifest {mpath.name}: fresh sizes {man['split_sizes']['fresh']}")
        paths.append(str(mpath.relative_to(ROOT)))
        if s.get("dev"):
            gpath = plan.CONFIGS / f"devgrid_{sname}.json"
            if not gpath.exists():
                grid = plan.devgrid(sname)
                gpath.write_text(json.dumps(grid, indent=1))
                log(f"devgrid {sname}: " + "; ".join(f"{r}: b={v['b']}" for r, v in grid["rules"].items()))
            paths.append(str(gpath.relative_to(ROOT)))
    commit(paths, "LLM study: bank manifests and weights-only dev grids (pre-registered, before any evaluation)", push)
    st["grids_done"] = True
    save_state(st)


def phase_enqueue_dev(settings: list[str], st: dict) -> None:
    if st.get("dev_enqueued"):
        return
    for sname in settings:
        s = plan.SETTINGS[sname]
        rj = plan.ref_jobs(sname)
        if s.get("optional"):
            rj = [(c, max(p, 30)) for c, p in rj]  # optional setting: after everything else
        keys = enqueue_all(rj)
        st.setdefault("ref_keys", {})[sname] = sorted(keys)
        if s.get("dev"):
            grid = json.loads((plan.CONFIGS / f"devgrid_{sname}.json").read_text())
            dj = plan.dev_jobs(sname, grid)
            st.setdefault("dev_keys", {})[sname] = sorted(enqueue_all(dj))
            nj = []
            for a in (1.0, 1.6):
                nj += plan.noise_jobs(sname, a, "val", 3 if s["bank"] == "l31_8b" else 2)
            st.setdefault("noise_keys", {})[sname] = sorted(enqueue_all(nj))
            st.setdefault("context_keys", {})[sname] = sorted(enqueue_all(plan.context_jobs(sname, ("val",))))
            log(f"{sname}: enqueued {len(dj)} dev jobs (+refs, noise, context)")
    st["dev_enqueued"] = True
    save_state(st)


def choose_tail_expert(sname: str, fz: dict) -> str:
    """The expert with the smallest mean lambda under the frozen kappa|tuned configuration."""
    from condmerge.llm import rules

    s = plan.SETTINGS[sname]
    c = next(c for c in fz["configs"] if c["tag"] == "kappa|tuned")
    b = math.inf if c["b"] == "inf" else float(c["b"])
    lam, _ = rules.build(s["bank"], s["T"], "kappa", b)
    return BK.tasks(s["bank"], s["T"])[int(np.argmin(lam.mean(0)))]


def tail_jobs(sname: str, fz: dict, edit_file: str) -> list[tuple[dict, int]]:
    s = plan.SETTINGS[sname]
    bank, T = s["bank"], s["T"]
    out = []
    pr = 2 if bank == "l31_8b" else 1
    for tag, splits in (("kappa|tuned", ("val", "test")), ("kappa|concentrated", ("val",))):
        c = next((c for c in fz["configs"] if c["tag"] == tag), None)
        if c is None:
            continue
        for sp in splits:
            for mats, coeffs, cell in (("orig", "orig", "A"), ("edited", "orig", "B"), ("orig", "recomputed", "C"),
                                       ("edited", "recomputed", "D")):
                out.append(({"bank": bank, "T": T, "kind": "tail", "rule": "kappa", "b": c["b"], "alpha": c["alpha"],
                             "split": sp, "group": "tail", "setting": sname, "matrices": mats, "coeffs": coeffs,
                             "cell": cell, "config_tag": tag, "edit_file": edit_file}, pr))
    cu = next(c for c in fz["configs"] if c["tag"] == "uniform|tuned")
    for sp in ("val", "test"):
        for mats in ("orig", "edited"):
            out.append(({"bank": bank, "T": T, "kind": "tail", "rule": "uniform", "b": 0.0, "alpha": cu["alpha"],
                         "split": sp, "group": "tail", "setting": sname, "matrices": mats, "coeffs": "orig",
                         "cell": f"U-{mats}", "config_tag": "uniform|tuned", "edit_file": edit_file}, pr))
    for sp in ("val", "test"):
        out.append(({"bank": bank, "T": T, "kind": "expert_edit", "split": sp, "group": "tail_expert",
                     "setting": sname, "edit_file": edit_file}, pr))
    return out


def phase_freeze(sname: str, st: dict, push: bool) -> bool:
    """Returns True once the setting is frozen and its confirmation jobs are enqueued."""
    s = plan.SETTINGS[sname]
    if sname in st.get("frozen", {}):
        return True
    keys = set(st["dev_keys"][sname]) | set(st["ref_keys"][sname])
    if not finished(keys, st):
        return False
    grid = json.loads((plan.CONFIGS / f"devgrid_{sname}.json").read_text())
    fz = plan.freeze(sname, grid)
    n_dev = len(setting_results(sname, "dev"))
    fz["n_dev_records"] = n_dev
    fz["n_dev_jobs"] = len(st["dev_keys"][sname])
    fz["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fpath = plan.CONFIGS / f"frozen_{sname}.json"
    fpath.write_text(json.dumps(fz, indent=1))
    log(f"{sname}: FROZEN ({n_dev}/{fz['n_dev_jobs']} dev records): " +
        "; ".join(f"{c['tag']} b={c['b']} a={c['alpha']} dev={c['dev_norm']:.4f}" for c in fz["configs"][:6]))
    dev_paths = [str(p.relative_to(ROOT)) for p in (J.RESULTS / f"{s['bank']}_T{s['T']}").glob("*/*.jsonl")]
    commit([str(fpath.relative_to(ROOT))] + dev_paths,
           f"LLM {sname}: frozen dev selection (before any test/fresh evaluation of {sname})", push)
    jobs = plan.frozen_eval_jobs(sname, fz)
    cu = next(c for c in fz["configs"] if c["tag"] == "uniform|tuned")
    jobs += plan.noise_jobs(sname, cu["alpha"], "test", 2)
    jobs += plan.context_jobs(sname, ("test", "fresh"))
    if s.get("gradpref"):
        for tag in ("uniform|tuned", "kappa|tuned"):
            c = next(c for c in fz["configs"] if c["tag"] == tag)
            jobs.append(({"bank": s["bank"], "T": s["T"], "kind": "gradpref", "rule": c["rule"], "b": c["b"],
                          "alpha": c["alpha"], "split": "val", "group": "gradpref", "setting": sname, "tag": tag,
                          "n_per_task": 64}, 3))
    st.setdefault("confirm_keys", {})[sname] = sorted(enqueue_all(jobs))
    st.setdefault("frozen", {})[sname] = str(fpath.relative_to(ROOT))
    save_state(st)
    log(f"{sname}: enqueued {len(jobs)} confirmation jobs")
    if s.get("tail"):
        try:
            expert = choose_tail_expert(sname, fz)
            tag = f"lift_tau_{expert}"
            efile = ROOT / "artifacts" / "llm" / "edits" / f"{s['bank']}_T{s['T']}" / f"{tag}.npz"
            if not efile.exists():
                log(f"{sname}: tail edit prep for expert {expert} (CPU)")
                r = subprocess.run([PY, str(ROOT / "experiments" / "llm" / "tail_prep.py"), "--setting", sname,
                                    "--expert", expert, "--tag", tag, "--procs", "6"], cwd=ROOT, capture_output=True,
                                   text=True)
                (ROOT / "logs" / "llm" / f"tail_prep_{sname}.log").write_text(r.stdout + r.stderr)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-500:])
            tj = tail_jobs(sname, fz, str(efile.relative_to(ROOT)))
            st.setdefault("tail_keys", {})[sname] = sorted(enqueue_all(tj))
            st.setdefault("tail_expert", {})[sname] = expert
            save_state(st)
            log(f"{sname}: enqueued {len(tj)} tail-edit jobs (expert {expert})")
        except Exception:
            log(f"{sname}: TAIL PREP FAILED\n{traceback.format_exc()}")
    return True


def phase_transfer(sname: str, st: dict) -> None:
    s = plan.SETTINGS[sname]
    src = s.get("transfer_from")
    if not src or sname in st.get("transfer", {}) or src not in st.get("frozen", {}):
        return
    fz = json.loads((ROOT / st["frozen"][src]).read_text())
    jobs = plan.transfer_jobs(sname, fz)
    if not s.get("dev"):  # transfer-only setting: matched shuffles at the transferred kappa config, on test
        kap = next(c for c in fz["configs"] if c["tag"] == "kappa|tuned")
        for seed in s.get("matched", []):
            jobs.append(({"bank": s["bank"], "T": s["T"], "kind": "merge", "rule": f"shuf{seed}:kappa", "b": kap["b"],
                          "alpha": kap["alpha"], "split": "test", "group": "matched", "setting": sname,
                          "tag": "matched-config shuffle at transferred kappa|tuned"}, 6))
        jobs = [(c, max(p, 6)) for c, p in jobs]  # optional setting: lowest priority
    st.setdefault("transfer", {})[sname] = sorted(enqueue_all(jobs))
    save_state(st)
    log(f"{sname}: enqueued {len(jobs)} transfer jobs from {src}")


def phase_gate(st: dict) -> None:
    """Uncertainty-gate stage (protocol 02 sec. 9): enqueued ONLY if the diagnostic passed its pre-registered
    criterion. The diagnostic itself runs in the CLIP chain; this only watches its summary file."""
    if "gate" in st:
        return
    summ = ROOT / "results" / "uncert" / "diagnostic_summary.json"
    marker = ROOT / "results" / "uncert" / ".diagnostic.done"
    if not (summ.exists() and marker.exists()):
        return
    passed = bool(json.loads(summ.read_text()).get("diagnostic_passed"))
    if not passed:
        st["gate"] = {"status": "not run: the diagnostic failed its pre-registered criterion"}
        log("uncertainty diagnostic FAILED its criterion -> gate stage not run (protocol 02 sec. 9)")
    else:
        cfg = {"bank": "l31_8b", "T": 8, "kind": "script", "script": "experiments/uncert/gate.py", "args": [],
               "group": "scripts", "setting": "gate"}
        st["gate"] = {"status": "enqueued", "keys": [J.enqueue(cfg, 2) or J.key_of(cfg)]}
        log("uncertainty diagnostic PASSED -> gate stage enqueued (runs on an 8B worker's GPU, model unloaded)")
    save_state(st)


def all_keys(st: dict) -> set[str]:
    ks = set(st.get("gate", {}).get("keys", []))
    for field in ("ref_keys", "dev_keys", "noise_keys", "context_keys", "confirm_keys", "tail_keys", "transfer"):
        for v in st.get(field, {}).values():
            ks |= set(v)
    return ks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="S1,S2,S3,S4")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--poll", type=float, default=60.0)
    args = ap.parse_args()
    settings = args.settings.split(",")
    push = not args.no_push
    st = load_state()
    log(f"orchestrator start: settings {settings}, state keys {sorted(st)}")
    banks = sorted({plan.SETTINGS[s]["bank"] for s in settings})
    while not all((BK.ROOT / "artifacts" / "llm" / "spectra" / b / "stats.csv").exists() for b in banks):
        log("waiting for spectra ...")
        time.sleep(120)
    phase_grids(settings, st, push)
    phase_enqueue_dev(settings, st)
    last_report = 0.0
    while True:
        for sname in settings:
            if plan.SETTINGS[sname].get("dev"):
                try:
                    phase_freeze(sname, st, push)
                except Exception:
                    log(f"{sname}: freeze step error\n{traceback.format_exc()}")
            phase_transfer(sname, st)
        for name in J.requeue_stale(900.0):
            log(f"requeued stale running job {name}")
        try:
            phase_gate(st)
        except Exception:
            log(f"gate phase error\n{traceback.format_exc()}")
        keys = all_keys(st)
        frozen_all = all(s in st.get("frozen", {}) for s in settings if plan.SETTINGS[s].get("dev"))
        transfers_all = all(s in st.get("transfer", {}) for s in settings if plan.SETTINGS[s].get("transfer_from"))
        if time.time() - last_report > 1800:
            log(f"queue {J.queue_counts()}; outstanding {outstanding(keys)} of {len(keys)} tracked")
            last_report = time.time()
        chain_done = (ROOT / "logs" / "llm" / "clip_chain.log").exists() and \
            "chain done" in (ROOT / "logs" / "llm" / "clip_chain.log").read_text()
        if frozen_all and transfers_all and finished(keys, st) and ("gate" in st or chain_done):
            break
        time.sleep(args.poll)
    log("all jobs finished: analysis")
    r = subprocess.run([PY, str(ROOT / "experiments" / "llm" / "analyze.py"), "--settings", ",".join(settings)],
                       cwd=ROOT, capture_output=True, text=True)
    (ROOT / "logs" / "llm" / "analyze.log").write_text(r.stdout + r.stderr)
    log(f"analysis rc={r.returncode}")
    paths = [str(p.relative_to(ROOT)) for p in J.RESULTS.rglob("*") if p.is_file()
             and p.suffix in (".jsonl", ".json", ".md", ".csv", ".png", ".pdf")]
    commit(paths + ["configs/llm"], "LLM study: all runs finished; raw records, summaries, figures", push)
    if not plan.SMOKE:  # raw predictions / edits live under artifacts/ (outside git): archive them too
        dest = Path("/shared/ssd_30T/azpitarte_condmerge_backup") / f"llm_artifacts_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
        t = subprocess.run(["tar", "-czf", str(dest), "-C", str(ROOT / "artifacts" / "llm"), "preds", "edits", "spectra"],
                           capture_output=True, text=True)
        log(f"artifact archive rc={t.returncode} -> {dest}")
    (J.QROOT / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("ALL_DONE")


if __name__ == "__main__":
    main()
