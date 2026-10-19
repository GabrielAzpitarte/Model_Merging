#!/bin/bash
# One-screen status of the unattended program.
cd "$(dirname "$0")/../.."
echo "== $(date '+%F %T')"; tmux ls 2>/dev/null | grep claude_cm2 | cut -d: -f1 | tr '\n' ' '; echo
python3 - <<'PY'
import json, os, time, glob
q = "artifacts/llm/queue"
print("queue:", {k: len(os.listdir(f"{q}/{k}")) for k in ("pending", "running", "done", "failed") if os.path.isdir(f"{q}/{k}")})
for f in sorted(glob.glob(f"{q}/workers/*.json")):
    d = json.load(open(f)); c = d.get("cfg", {})
    print(f"  {os.path.basename(f)[:-5]:4s} {d['state']:8s} age {time.time()-os.path.getmtime(f):5.0f}s",
          c.get("bank", ""), c.get("T", ""), c.get("kind", ""), c.get("rule", c.get("method", "")), c.get("b", ""), c.get("alpha", ""), c.get("split", ""))
PY
echo "-- orchestrator:"; tail -n 4 logs/llm/orchestrator.log 2>/dev/null | cut -c1-220
echo "-- clip chain:"; tail -n 3 logs/llm/clip_chain.log 2>/dev/null
ls artifacts/llm/queue/failed/*.json 2>/dev/null | head -5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
