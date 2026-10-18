#!/usr/bin/env bash
# Per-expert input Gram statistics for RegMean-type solves (one sequential tmux job per GPU).
# Usage: scripts/run_expert_grams.sh GPU MODE   (MODE = C: own task's calib; B: pooled calib)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/lora_merging_ica/.venv/bin/python
GPU=$1; MODE=$2
FILTER='Loading weights|UNEXPECTED|Using the latest|Found the latest|^Key|^---|^Notes|can be ignored|LOAD REPORT|^$'
CMD=""
for e in sun397 cars resisc45 eurosat svhn gtsrb mnist dtd; do
  if [ "$MODE" = C ]; then D="task:$e"; else D=pool; fi
  CMD+="$PY scripts/compute_input_grams.py --model $e --data $D --device cuda:0 2>&1 | grep --line-buffered -v -E '$FILTER' | tee -a logs/grams_$MODE.log; "
done
tmux new-session -d -s "claude_cm_grams_$MODE" "cd $PWD && export CUDA_VISIBLE_DEVICES=$GPU; $CMD"
echo "launched grams $MODE on GPU $GPU"
