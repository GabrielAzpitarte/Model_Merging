"""Protocol 02 amendment 3, for the orchestrator process that was started before the amendment (a running Python
process does not re-import plan.py; orchestrate.py's phase_amend3 does the same on any later restart).

  1. once: enqueue the extra-alpha dev jobs of S1-S3 (plan.dev_jobs_extra), at a priority ahead of the remaining
     original dev jobs, so they finish first and the running orchestrator's freeze -- which reads every dev record of
     the setting -- includes them;
  2. after each setting's freeze (configs/llm/frozen_<S>.json): check that the frozen dev table contained the extra
     records, then enqueue the matched-config shuffles at kappa|alpha1 (plan.matched_alpha1_jobs);
  3. retry its own failed jobs once (the running orchestrator does not track them);
  4. after ALL_DONE: once its own jobs are finished, re-run the analysis and commit the summaries locally (the
     running orchestrator's analysis may have started before they finished).
Only our own queue and result files are touched; no GPU is looked for (workers run on fixed cards).
State: artifacts/llm/amend3_state.json; log: logs/llm/amend3_watch.log.
Usage: amend3_watch.py [--poll 120]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan  # noqa: E402
from condmerge.llm import jobs as J  # noqa: E402

STATE = ROOT / "artifacts" / "llm" / "amend3_state.json"
LOG = ROOT / "logs" / "llm" / "amend3_watch.log"
ORCH_STATE = ROOT / "artifacts" / "llm" / "orchestrator_state.json"
PY = "/data/home/azpitarte/lora_merging_ica/.venv/bin/python"
DEV_SETTINGS = [s for s, v in plan.SETTINGS.items() if v.get("dev")]


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def live_keys() -> set[str]:
    d = J.dirs()
    return {p.name.split("_", 1)[1].split(".json")[0] for sub in ("pending", "running") for p in d[sub].glob("*.json*")}


def failed_specs() -> dict[str, Path]:
    d = J.dirs()
    return {p.name.split("_", 1)[1].split(".json")[0]: p for p in d["failed"].glob("*.json")}


def enqueue(jobs: list[tuple[dict, int]]) -> list[str]:
    keys = []
    for cfg, prio in jobs:
        J.enqueue(cfg, prio)
        keys.append(J.key_of(cfg))
    return sorted(set(keys))


def tracked(st: dict) -> set[str]:
    out = set()
    for field in ("extras", "alpha1"):
        for v in st.get(field, {}).values():
            out |= set(v)
    return out


def retry_failed(st: dict) -> None:
    fk = failed_specs()
    for k in tracked(st) & set(fk):
        if st.setdefault("retries", {}).get(k, 0) >= 1:
            continue
        spec = json.loads(fk[k].read_text())
        log(f"retrying failed job {k}: {spec['cfg'].get('rule', '')} b={spec['cfg'].get('b')} "
            f"a={spec['cfg'].get('alpha')} ({(spec.get('error') or '')[-160:]!r})")
        fk[k].rename(fk[k].with_suffix(".json.failed1"))
        J.enqueue(spec["cfg"], spec["prio"])
        st["retries"][k] = 1
        save_state(st)


def commit(paths: list[str], msg: str) -> None:
    def git(*a):
        return subprocess.run(["git", "-c", "user.name=Gabriel Azpitarte", "-c", "user.email=gabriel.azpitarte@icloud.com",
                               *a], cwd=ROOT, capture_output=True, text=True)
    git("add", "--", *paths)
    r = git("commit", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    log(f"git commit: {(r.stdout.strip().splitlines() or [r.stderr.strip()[:200]])[0]}")
    b = subprocess.run(["bash", str(ROOT / "scripts" / "backup.sh")], cwd=ROOT, capture_output=True, text=True)
    log(f"local backup rc={b.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=float, default=120.0)
    args = ap.parse_args()
    st = load_state()
    orch = json.loads(ORCH_STATE.read_text()) if ORCH_STATE.exists() else {}
    if "extras" not in st:
        st["extras"] = {}
        for s in DEV_SETTINGS:
            if s in orch.get("frozen", {}) or (plan.CONFIGS / f"frozen_{s}.json").exists():
                log(f"{s}: already frozen -- its extra alphas can no longer enter the selection (not enqueued)")
                continue
            grid = json.loads((plan.CONFIGS / f"devgrid_{s}.json").read_text())
            jobs = plan.dev_jobs_extra(s, grid)
            st["extras"][s] = enqueue(jobs)
            log(f"{s}: enqueued {len(jobs)} extra-alpha dev jobs (alphas {plan.ALPHAS_EXTRA.get(s)}, "
                f"priority {plan.EXTRA_PRIO.get(s)})")
        st["n_original_dev"] = {s: len(orch.get("dev_keys", {}).get(s, [])) for s in DEV_SETTINGS}
        save_state(st)
    while True:
        for s in DEV_SETTINGS:
            fpath = plan.CONFIGS / f"frozen_{s}.json"
            if s in st.setdefault("alpha1", {}) or not fpath.exists():
                continue
            fz = json.loads(fpath.read_text())
            expected = st.get("n_original_dev", {}).get(s, 0) + len(st["extras"].get(s, []))
            n = fz.get("n_dev_records")
            ok = n is not None and n >= expected
            st.setdefault("freeze_check", {})[s] = {"n_dev_records": n, "expected_with_extras": expected, "ok": ok}
            log(f"{s}: frozen with {n} dev records (expected {expected} incl. extra alphas) -> "
                f"{'OK' if ok else 'WARNING: some dev records were missing at the freeze'}")
            jobs = plan.matched_alpha1_jobs(s, fz)
            st["alpha1"][s] = enqueue(jobs)
            log(f"{s}: enqueued {len(jobs)} matched-config shuffles at kappa|alpha1"
                + ("" if jobs else " (kappa|alpha1 coincides with kappa|tuned: the ordinary matched set covers it)"))
            save_state(st)
        retry_failed(st)
        if (J.QROOT / "ALL_DONE").exists() and not (tracked(st) & live_keys()):
            log("ALL_DONE and every amendment-3 job finished: re-running the analysis")
            r = subprocess.run([PY, str(ROOT / "experiments" / "llm" / "analyze.py")], cwd=ROOT, capture_output=True,
                               text=True)
            (ROOT / "logs" / "llm" / "analyze_amend3.log").write_text(r.stdout + r.stderr)
            log(f"analysis rc={r.returncode}")
            paths = [str(p.relative_to(ROOT)) for p in J.RESULTS.rglob("*") if p.is_file()
                     and p.suffix in (".jsonl", ".json", ".md", ".csv", ".png", ".pdf")]
            commit(paths + ["configs/llm"], "LLM study: analysis re-run after the amendment-3 jobs finished")
            st["final_done"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(st)
            log("done")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
