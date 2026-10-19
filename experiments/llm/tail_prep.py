"""Tail-edit intervention, preparation (CPU): the edit of ONE expert's matrices within the checkpoint's own
BF16 uncertainty.

For every adapted matrix p where the chosen expert's sigma_min(W_i[p]) < tau_bf16[p] (the base's BF16 resolution
floor, compute_spectra.py), lift its singular values below the floor to the floor:

    W_i'[p] = U diag(max(sigma_j, tau)) V^T   =>   E_i[p] = W_i' - W_i = U_t diag(tau - sigma_t) V_t^T,

t = the indices with sigma_t < tau. ||E||_2 < tau: the edit is smaller than the rounding uncertainty of the stored
base (an equally valid preimage of the checkpoint differs from it by about that much). The edit changes the
expert's kappa (sigma_min -> tau) and hence its literal-rule coefficients.
Writes artifacts/llm/edits/<bank>_T<T>/<tag>.npz with U_<k> = U_t diag(tau - sigma_t) (out x |t|, fp32),
V_<k> = V_t^T (|t| x in, fp32), modules, expert index, and the edited scores of every rule (JSON string).
Usage: tail_prep.py --setting S1 --expert <task> --tag <tag> [--procs 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _init(threads):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _edit_one(job):
    bank, module, task, tau = job
    import numpy as np
    import torch

    from condmerge.llm import banks as BK

    w0 = BK.base_weight(bank, module).double()
    a, b = BK.load_lora_module(bank, task, module)
    w = w0 + torch.from_numpy(b).double() @ torch.from_numpy(a).double()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    t = torch.nonzero(s < tau).flatten()
    us = (u[:, t] * (tau - s[t])).float().numpy()
    v = vh[t].float().numpy()
    s_new = torch.clamp(s, min=tau)
    # exactness check of the edit on the full matrix (float64)
    e = u[:, t] @ torch.diag(tau - s[t]) @ vh[t]
    s_chk = torch.linalg.svdvals(w + e)
    err = float((s_chk - s_new).abs().max() / s[0])
    return module, us, v, float(s[0]), float(s[-1]), float(s_new[-1]), int(len(t)), err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--expert", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    _init(args.threads)
    import multiprocessing as mp

    import numpy as np

    import plan
    from condmerge.llm import banks as BK
    from condmerge.llm import rules

    s = plan.SETTINGS[args.setting]
    bank, T = s["bank"], s["T"]
    tasks = BK.tasks(bank, T)
    e_idx = tasks.index(args.expert)
    sp = rules.spectra(bank)
    row = sp[BK.module_names(bank)[0]]["versions"].index(args.expert)
    todo = [(bank, m, args.expert, sp[m]["tau"]) for m in BK.module_names(bank) if sp[m]["s"][row, -1] < sp[m]["tau"]]
    outdir = ROOT / "artifacts" / "llm" / "edits" / f"{bank}_T{T}"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(args.procs, initializer=_init, initargs=(args.threads,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_edit_one, todo)):
            res[r[0]] = r
            print(f"[{i + 1}/{len(todo)}] {r[0]} lifted {r[6]} values, sigma_min {r[4]:.3g} -> {r[5]:.3g}, "
                  f"exactness {r[7]:.1e} ({time.time() - t0:.0f}s)", flush=True)
    mods = [m for m in BK.module_names(bank) if m in res]
    arrays = {}
    for k, m in enumerate(mods):
        arrays[f"U_{k}"] = res[m][1]
        arrays[f"V_{k}"] = res[m][2]
    # edited scores of every rule for the edited expert (row e_idx of each rule's table)
    edited = {}
    for rule in ("kappa", "logkappa", "kappa_res", "kappa_rev"):
        edited[rule] = {}
    for m in mods:
        s1, smin_new = res[m][3], res[m][5]
        k_new = s1 / smin_new
        edited["kappa"][m] = k_new
        edited["kappa_rev"][m] = -k_new
        edited["logkappa"][m] = float(np.log(k_new))
        edited["kappa_res"][m] = s1 / max(smin_new, sp[m]["tau"])
    info = {"bank": bank, "T": T, "expert": args.expert, "n_modules_edited": len(mods),
            "lifted_per_module": {m: res[m][6] for m in mods}, "max_exactness_error": max(res[m][7] for m in mods),
            "edit_spectral_norm_over_tau": "< 1 by construction (max lift = tau - sigma_min < tau)"}
    np.savez(outdir / f"{args.tag}.npz", modules=np.asarray(mods), expert=e_idx, edit_tag=args.tag,
             edited_scores=json.dumps(edited), info=json.dumps(info), **arrays)
    (outdir / f"{args.tag}.json").write_text(json.dumps(info, indent=1))
    print(f"DONE {len(mods)} modules edited ({time.time() - t0:.0f}s) -> {outdir / (args.tag + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
