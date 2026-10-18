"""Append-only JSONL run records keyed by a config hash (makes sweeps resumable)."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from pathlib import Path

from .paths import git_commit


def config_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_record(path: Path, cfg: dict, metrics: dict, extra: dict | None = None) -> dict:
    rec = {"key": config_key(cfg), "cfg": cfg, "metrics": metrics, "commit": git_commit(),
           "host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec
