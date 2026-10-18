#!/usr/bin/env bash
# Stage 3 subsets (pre-registered in configs/subsets.json): literal rule vs reversed/shuffled/log
# on representative and random expert subsets, reduced shared grid (b in {1,4,16,inf},
# alpha in {1.0,1.6,2.2}), dev only. One sequential tmux job per GPU.
# Usage: scripts/run_subset_sweeps.sh GPU "subset1 subset2 ..."
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/lora_merging_ica/.venv/bin/python
GPU=$1; SUBSETS=$2
FILTER='Loading weights|UNEXPECTED|Using the latest|Found the latest|^Key|^---|^Notes|can be ignored|LOAD REPORT|^$'
CMD=""
for s in $SUBSETS; do
  EXP=$($PY -c "import json;print(','.join(json.load(open('configs/subsets.json'))['subsets']['$s']))")
  CMD+="$PY experiments/rule_sweep.py --experts $EXP --rules uniform,kappa,kappa_rev,logkappa,shuf0:kappa,shuf1:kappa,shuf2:kappa \
 --betas 1,4,16,inf --alphas 1.0,1.6,2.2 --device cuda:0 --out results/sweeps/subsets_dev_$s.jsonl 2>&1 | grep --line-buffered -v -E '$FILTER' | tee -a logs/subset_$s.log; "
done
tmux new-session -d -s "claude_cm_subsets_g$GPU" "cd $PWD && export CUDA_VISIBLE_DEVICES=$GPU; $CMD"
echo "launched subsets [$SUBSETS] on GPU $GPU"
