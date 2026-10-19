# Model_Merging: conditioning-guided model merging

Research code and records for testing the proposal: *weight each merged matrix by an exponential function of its
condition number, so that larger condition numbers get smaller weights*, lambda_i[p] = softmax_i(-beta kappa(W_i[p])).
The repository also covers the conditioning of RegMean-type merge solves and an uncertainty diagnostic for
calibration-derived merge corrections.

| part | where | status |
|---|---|---|
| CLIP ViT-B/32 and B/16 full fine-tunes, 4 LoRA banks: literal rule, controls, variants, diagnostics, RegMean | `docs/FULL_REPORT.md` (+ `docs/appendix/`) | done 2026-10-18 |
| Protocol for the LLM study and follow-ups (pre-registration + amendments) | `docs/PROTOCOL_02_llm_followups.md` | 2026-10-19 |
| LLM banks: Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct LoRA experts (exact spectra, exact merges) | `src/condmerge/llm/`, `experiments/llm/`, `results/llm/` | running |
| Reproduction audit of the CLIP results | `experiments/repro/`, `results/repro/` | running |
| Inverse-solve audit (fp16/fp32), equal-budget ridge comparison on LoRA | `experiments/inverse/`, `results/inverse/` | running |
| Uncertainty diagnostic for merge corrections | `experiments/uncert/`, `results/uncert/` | running |
| Prior-work check | `docs/lit/PRIOR_WORK_02.md` | done |

## Running

* Environment: `/data/home/azpitarte/lora_merging_ica/.venv/bin/python` (torch 2.11, transformers 5.17).
* Large artifacts (pixel caches, spectra, predictions) live under `artifacts/` (a symlink outside git).
* Unattended LLM program (resumable): `bash experiments/llm/start_all.sh`; status: `bash experiments/llm/status.sh`.
  * The orchestrator (`experiments/llm/orchestrate.py`) freezes each setting's dev selection into
    `configs/llm/frozen_<setting>.json` and commits it before any test or fresh job is enqueued.
  * Workers run on fixed GPUs (`experiments/llm/run_worker.sh`).
* Tests: `python -m pytest tests/`.
