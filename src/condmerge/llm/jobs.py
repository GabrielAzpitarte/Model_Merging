"""A file-based job queue for our own GPUs (no GPU discovery: workers are started on fixed cards).

artifacts/llm/queue/
  pending/<prio>_<key>.json    job specs, claimed by atomic rename into running/
  running/<prio>_<key>.json.<worker>
  done/  failed/               finished specs (failed ones carry the traceback)
  workers/<worker>.json        heartbeat: current job, time, loaded bank
  STOP                         workers exit after their current job
  ALL_DONE                     written by the orchestrator at the end; idle workers exit
Results are append-only JSONL shards, one per worker (no concurrent writers on a file), keyed by the job key.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from . import banks as BK

_SMOKE = os.environ.get("CONDMERGE_LLM_SMOKE")  # smoke tests: everything under artifacts/llm_smoke/<name>
QROOT = BK.ROOT / "artifacts" / ("llm" if not _SMOKE else f"llm_smoke/{_SMOKE}") / "queue"
RESULTS = BK.ROOT / "results" / "llm" if not _SMOKE else BK.ROOT / "artifacts" / f"llm_smoke/{_SMOKE}" / "results"
PREDS = BK.ROOT / "artifacts" / ("llm" if not _SMOKE else f"llm_smoke/{_SMOKE}") / "preds"


def key_of(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dirs() -> dict[str, Path]:
    d = {k: QROOT / k for k in ("pending", "running", "done", "failed", "workers")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BK.ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "src", "experiments",
                                "vendor"], cwd=BK.ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def result_dir(bank: str, T: int, group: str) -> Path:
    d = RESULTS / f"{bank}_T{T}" / group
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_results(bank: str, T: int, group: str | None = None) -> list[dict]:
    root = RESULTS / f"{bank}_T{T}"
    paths = sorted(root.glob("*/*.jsonl")) if group is None else sorted((root / group).glob("*.jsonl"))
    out, seen = [], set()
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line of a crashed writer
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            out.append(r)
    return out


def done_keys(bank: str, T: int) -> set[str]:
    return {r["key"] for r in read_results(bank, T)}


def enqueue(cfg: dict, prio: int) -> str | None:
    """Write a pending job unless it is already pending / running / done (by key) or has a result."""
    d = dirs()
    k = key_of(cfg)
    for sub in ("pending", "running", "done"):
        if any(d[sub].glob(f"*_{k}.json*")):
            return None
    name = f"{prio:03d}_{k}.json"
    tmp = d["pending"] / (name + ".tmp")
    tmp.write_text(json.dumps({"key": k, "prio": prio, "cfg": cfg}, indent=1))
    os.replace(tmp, d["pending"] / name)
    return k


def claim(worker: str, prefer_bank: str | None, strict: bool = False, loaded: tuple | None = None,
          only_bank: str | None = None) -> tuple[Path, dict] | None:
    """Claim the best pending job: by priority, then the loaded (bank, T) first (no model reload). strict: the
    preferred bank's jobs first whatever their priority (a worker dedicated to one bank while it has work)."""
    d = dirs()
    cands = sorted(p for p in d["pending"].glob("*.json"))

    def rank(p):
        try:
            cfg = json.loads(p.read_text())["cfg"]
        except Exception:
            return (1, 10 ** 6, 2, p.name)
        mis = 0 if cfg.get("bank") == prefer_bank else 1
        same = 0 if loaded and (cfg.get("bank"), cfg.get("T")) == tuple(loaded) else 1
        pr = int(p.name.split("_")[0])
        return (mis, pr, same, p.name) if strict else (pr, same, mis, p.name)
    for p in sorted(cands, key=rank):
        if only_bank is not None:
            try:
                if json.loads(p.read_text())["cfg"].get("bank") != only_bank:
                    continue
            except Exception:
                continue
        dst = d["running"] / f"{p.name}.{worker}"
        try:
            os.rename(p, dst)
        except FileNotFoundError:
            continue
        return dst, json.loads(dst.read_text())
    return None


def finish(path: Path, ok: bool, error: str | None = None) -> None:
    d = dirs()
    spec = json.loads(path.read_text())
    if error:
        spec["error"] = error
    target = d["done" if ok else "failed"] / path.name.rsplit(".", 1)[0]
    target.write_text(json.dumps(spec, indent=1))
    path.unlink(missing_ok=True)


def heartbeat(worker: str, info: dict) -> None:
    d = dirs()
    (d["workers"] / f"{worker}.json").write_text(json.dumps({**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             "host": socket.gethostname(), "pid": os.getpid()}))


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def queue_counts() -> dict[str, int]:
    d = dirs()
    return {k: len(list(d[k].glob("*.json*"))) for k in ("pending", "running", "done", "failed")}


def requeue_own(worker: str) -> list[str]:
    """On (re)start: jobs this worker had claimed were interrupted -> back to pending."""
    d = dirs()
    out = []
    for p in d["running"].glob(f"*.json.{worker}"):
        target = d["pending"] / p.name[: -(len(worker) + 1)]
        os.rename(p, target)
        out.append(target.name)
    return out


def requeue_stale(max_age_s: float = 900.0) -> list[str]:
    """Running jobs whose worker heartbeat is missing or older than max_age_s -> back to pending."""
    d = dirs()
    out = []
    now = time.time()
    for p in d["running"].glob("*.json.*"):
        worker = p.name.rsplit(".", 1)[1]
        hb = d["workers"] / f"{worker}.json"
        age = now - hb.stat().st_mtime if hb.exists() else float("inf")
        if age > max_age_s:
            try:
                os.rename(p, d["pending"] / p.name[: -(len(worker) + 1)])
                out.append(f"{p.name} (heartbeat age {age:.0f}s)")
            except FileNotFoundError:
                pass
    return out
