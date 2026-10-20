#!/bin/bash
# Pre-registered gate stage (protocol 02 sec. 9): the diagnostic passed on 2026-10-19 but the queued gate job failed
# (the LLM worker's HF_HOME hid the CLIP checkpoint). Runs gate.py on GPU 0 (our allocation) once the LLM program has
# written ALL_DONE and worker g0 has exited. Never looks for other GPUs.
cd "$(dirname "$0")/../.."
mkdir -p logs/uncert
LOG=logs/uncert/gate_rerun.log
echo "$(date '+%F %T') waiting for ALL_DONE and the g0 worker to exit" >> $LOG
until [ -f artifacts/llm/queue/ALL_DONE ] && ! tmux has-session -t claude_cm2_w0 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') start gate.py on GPU 0" >> $LOG
env -u HF_HOME CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /data/home/azpitarte/lora_merging_ica/.venv/bin/python \
  experiments/uncert/gate.py >> $LOG 2>&1
echo "$(date '+%F %T') gate.py rc=$?" >> $LOG
