#!/usr/bin/env bash
# Pilot (8 experts, dev): literal rule + controls on three GPUs, one tmux session each.
# Every rule uses the shared grid in experiments/rule_sweep.py (11 b x 6 alpha).
# Usage: scripts/run_pilot_sweeps.sh "0 1 2"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/lora_merging_ica/.venv/bin/python
read -r G0 G1 G2 <<<"${1:-0 1 2}"
FILTER='Loading weights|UNEXPECTED|Using the latest|Found the latest|^Key|^---|^Notes|can be ignored|LOAD REPORT|^$'
launch() {  # gpu shard rules
  tmux new-session -d -s "claude_cm_sweep_$2" "cd $PWD && CUDA_VISIBLE_DEVICES=$1 $PY experiments/rule_sweep.py \
    --rules $3 --device cuda:0 --out results/sweeps/rule_sweep_dev_$2.jsonl 2>&1 \
    | grep --line-buffered -v -E '$FILTER' | tee -a logs/rule_sweep_$2.log"
}
launch "$G0" literal "kappa,kappa_rev,logkappa,logkappa_rev,kappa_strict"
launch "$G1" shuffled "shuf0:kappa,shuf1:kappa,shuf2:kappa,shuf3:kappa,shuf4:kappa"
launch "$G2" controls "kappa_z,kappa_z_rev,global:kappa_z,global:kappa_z_rev,dnorm,dnorm_rev,logsmax,neglogsmin"
tmux ls | grep claude_cm_sweep
