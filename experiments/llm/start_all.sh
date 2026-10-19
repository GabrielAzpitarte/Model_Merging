#!/bin/bash
# Launch or RESUME the whole unattended program (idempotent: existing tmux sessions are left alone; the
# orchestrator resumes from artifacts/llm/orchestrator_state.json; workers requeue their interrupted jobs).
# GPUs: 0-3 LLM workers preferring the 8B bank; 4 = 3B-only worker + the CLIP chain (then an extra 8B worker).
cd "$(dirname "$0")/../.."
PY=/data/home/azpitarte/lora_merging_ica/.venv/bin/python
mkdir -p logs/llm
rm -f artifacts/llm/queue/STOP
start() { tmux has-session -t "$1" 2>/dev/null && echo "exists: $1" || { tmux new-session -d -s "$1" "$2"; echo "started: $1"; }; }
start claude_cm2_orch "cd $PWD && $PY experiments/llm/orchestrate.py --settings S1,S2,S3,S4 --no-push >> logs/llm/orchestrator.out 2>&1"
for g in 0 1 2 3; do start claude_cm2_w$g "cd $PWD && bash experiments/llm/run_worker.sh $g g$g l31_8b"; done
start claude_cm2_w4 "cd $PWD && bash experiments/llm/run_worker.sh 4 g4 l32_3b l32_3b"
if [ ! -f logs/llm/clip_chain.log ] || ! grep -q "chain done" logs/llm/clip_chain.log; then
  start claude_cm2_clip "cd $PWD && bash experiments/run_clip_chain.sh 4"
fi
