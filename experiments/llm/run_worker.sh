#!/bin/bash
# One persistent worker on ONE fixed GPU (our allocation); restarts the process at most 5 times after a crash
# (e.g. a corrupted CUDA context, exit 3). Never looks for other GPUs.
GPU=$1; NAME=${2:-g$1}; PREFER=${3:-}; ONLY=${4:-}
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
PY=/data/home/azpitarte/lora_merging_ica/.venv/bin/python
mkdir -p logs/llm
for i in 1 2 3 4 5 6; do
  echo "$(date '+%F %T') worker $NAME start (attempt $i) on GPU $GPU" >> logs/llm/worker_$NAME.log
  $PY experiments/llm/worker.py --name $NAME ${PREFER:+--prefer-bank $PREFER} ${ONLY:+--only-bank $ONLY} >> logs/llm/worker_$NAME.log 2>&1
  rc=$?
  echo "$(date '+%F %T') worker $NAME exited rc=$rc" >> logs/llm/worker_$NAME.log
  if [ $rc -eq 0 ]; then exit 0; fi
  if [ -f artifacts/llm/queue/STOP ]; then exit 0; fi
  sleep 30
done
