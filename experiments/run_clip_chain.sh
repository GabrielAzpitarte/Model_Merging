#!/bin/bash
# CLIP follow-ups on ONE fixed GPU (default 4), sequentially; each step runs only if its completion marker
# (results/<area>/.<step>.done) is absent, and failures do not block later steps. At the end the same card is
# handed to an extra 8B LLM worker (one continuous job on a card we already hold).
GPU=${1:-4}
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/data/home/azpitarte/lora_merging_ica/.venv/bin/python
L=logs/llm/clip_chain.log
log() { echo "$(date '+%F %T') $*" >> $L; }
step() {  # step <marker> <log> <command...>
  local marker=$1 out=$2; shift 2
  if [ -f "$marker" ]; then log "skip (done): $marker"; return; fi
  log "start: $*"
  "$@" > "$out" 2>&1; local rc=$?
  log "rc=$rc: $*"
  [ $rc -eq 0 ] && touch "$marker"
}
log "chain start on GPU $GPU"
step results/repro/.rules.done   logs/llm/clip_repro_rules.log   $PY experiments/repro/clip_repro.py --parts rules
step results/repro/.lora.done    logs/llm/clip_repro_lora.log    $PY experiments/repro/clip_repro.py --parts lora
step results/repro/.regmean.done logs/llm/clip_repro_regmean.log $PY experiments/repro/clip_repro.py --parts regmean
step results/inverse/.collapse_lora.done   logs/llm/collapse_audit.log        $PY experiments/inverse/collapse_audit.py
step results/inverse/.collapse_fullft.done logs/llm/collapse_audit_fullft.log $PY experiments/inverse/collapse_audit_fullft.py
step results/inverse/.ridge_equal.done     logs/llm/lora_ridge_equal.log      $PY experiments/inverse/lora_ridge_equal.py
step results/uncert/.diagnostic.done       logs/llm/uncert_diagnostic.log     $PY experiments/uncert/diagnostic.py --configs C:16,C:64,B:16
log "chain done; GPU $GPU continues as an extra 8B worker (g4b)"
exec bash experiments/llm/run_worker.sh $GPU g4b l31_8b
