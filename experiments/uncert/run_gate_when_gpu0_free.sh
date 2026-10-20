#!/bin/bash
# Variant of run_gate_after_llm.sh for an early stop of the LLM program (S3/S4 cut on 2026-10-20): runs the
# pre-registered gate stage on GPU 0 (our allocation) as soon as worker g0 has exited, whether by ALL_DONE or STOP.
cd "$(dirname "$0")/../.."
mkdir -p logs/uncert
LOG=logs/uncert/gate_rerun.log
echo "$(date '+%F %T') waiting for the g0 worker to exit" >> $LOG
while tmux has-session -t claude_cm2_w0 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start gate.py on GPU 0" >> $LOG
env -u HF_HOME CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /data/home/azpitarte/lora_merging_ica/.venv/bin/python \
  experiments/uncert/gate.py >> $LOG 2>&1
echo "$(date '+%F %T') gate.py rc=$?" >> $LOG
