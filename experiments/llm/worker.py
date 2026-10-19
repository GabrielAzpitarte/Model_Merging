"""Persistent GPU worker: loads one bank once, claims jobs from the queue (src/condmerge/llm/jobs.py), evaluates.

Job kinds (cfg["kind"]):
  merge     rule/b/alpha merge (rules.build) on a split; optional numerical jitter replicate
  expert    every expert of the bank on its own task (normalisation reference) [+ full cross-task matrix]
  base      zero-shot base model
  context   weight-only literature baselines through the same exact path: isoc (alpha x flat(sum Delta)),
            recorded alpha of the source project
  tail      tail-edit intervention: matrices orig|edited x coefficients orig|recomputed (+ uniform), edit file
  expert_edit  one edited expert alone on its own task (+ root log-prob / hidden-state changes)
  gradpref  d(mean val loss) / d lambda_i[p] at a given merge (labelled-data diagnostic, budget D)
Every merge record stores: requested lambda (maps x K, 6 digits) and alpha, the rule log (inf / tie / fallback
counts), n_eff, per-map merged-update norms, a fingerprint computed from the request and from the installed GPU
buffers (asserted equal), per-task metrics, and the path of the saved per-example predictions.
Usage: worker.py --gpu 0 --name g0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from condmerge.llm import banks as BK  # noqa: E402
from condmerge.llm import jobs as J  # noqa: E402


class State:
    runner = None
    key = None


def get_runner(bank: str, T: int, device: str):
    import torch

    from condmerge.llm.runner import Runner

    if State.key != (bank, T):
        if State.runner is not None:
            del State.runner
            State.runner = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        State.runner = Runner(bank, T, device)
        State.key = (bank, T)
    return State.runner


def expert_refs(bank: str, T: int, split: str) -> list[float] | None:
    for r in J.read_results(bank, T, "experts"):
        if r["cfg"]["split"] == split and r["cfg"]["kind"] == "expert":
            return r["own_local"]
    return None


def save_preds(bank: str, T: int, split: str, key: str, pred: dict) -> str:
    d = J.PREDS / f"{bank}_T{T}" / split
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.npz"
    np.savez_compressed(p, **pred)
    return str(p.relative_to(BK.ROOT))


def summarize(met: dict, ref: list[float] | None) -> dict:
    out = {k: met[k] for k in ("joint_macro", "local_macro", "free_macro", "joint_per_task", "local_per_task",
                               "n", "seconds")}
    if ref is not None:
        loc = np.asarray(met["local_per_task"], np.float64)
        ok = np.isfinite(loc)
        norm = loc[ok] / np.asarray(ref, np.float64)[ok]
        out["normalized"] = float(norm.mean())
        out["normalized_per_task"] = [float(x) if np.isfinite(x) else None for x in loc / np.asarray(ref)]
        out["normalized_min"] = float(norm.min())
    return out


# ----------------------------------------------------------------------------------------------- job kinds
def run_merge(cfg: dict, key: str, device: str) -> dict:
    from condmerge.llm import rules

    r = get_runner(cfg["bank"], cfg["T"], device)
    b = float("inf") if cfg.get("b") in ("inf", float("inf")) else float(cfg.get("b", 0.0))
    lam, info = rules.build(cfg["bank"], cfg["T"], cfg["rule"], b, cfg.get("jitter_rel", 0.0), cfg.get("jitter_seed", 0))
    if cfg["rule"] != "uniform" and b != 0 and not cfg.get("jitter_rel") and info["dist_from_uniform"] < 1e-12:
        info["warning"] = "non-uniform rule produced exactly uniform coefficients"
    chk = r.set_merge(lam, float(cfg["alpha"]))
    r.set_extras(None)
    met, pred = r.evaluate(cfg["split"])
    return {"metrics": summarize(met, expert_refs(cfg["bank"], cfg["T"], cfg["split"])), "rule_info": info,
            "lambda": np.round(lam, 6).tolist(), "check": chk,
            "preds": save_preds(cfg["bank"], cfg["T"], cfg["split"], key, pred)}


def run_expert(cfg: dict, key: str, device: str) -> dict:
    r = get_runner(cfg["bank"], cfg["T"], device)
    nm, K = len(r.mods), r.K
    own, full = [], {}
    for k in range(K):
        lam = np.zeros((nm, K))
        lam[:, k] = 1.0
        r.set_merge(lam, 1.0)
        r.set_extras(None)
        if cfg.get("full_matrix"):
            met, pred = r.evaluate(cfg["split"])
            full[r.tasks[k]] = {"local_per_task": met["local_per_task"], "joint_per_task": met["joint_per_task"]}
            save_preds(cfg["bank"], cfg["T"], cfg["split"], f"{key}_expert{k}", pred)
            own.append(met["local_per_task"][k])
        else:
            met, pred = r.evaluate(cfg["split"], only_task=k)
            save_preds(cfg["bank"], cfg["T"], cfg["split"], f"{key}_expert{k}", pred)
            own.append(met["local_per_task"][k])
    r.clear()
    return {"own_local": own, "tasks": r.tasks, "full_matrix": full or None}


def run_base(cfg: dict, key: str, device: str) -> dict:
    r = get_runner(cfg["bank"], cfg["T"], device)
    r.clear()
    met, pred = r.evaluate(cfg["split"])
    return {"metrics": summarize(met, expert_refs(cfg["bank"], cfg["T"], cfg["split"])),
            "preds": save_preds(cfg["bank"], cfg["T"], cfg["split"], key, pred)}


def isoc_extras(r, alpha: float) -> dict:
    """Iso-C (official form, as the source pipelines): alpha * mean(sigma) U V^T of the SUM of the T updates,
    computed exactly from the concatenated factors (QR cores)."""
    import torch

    out = {}
    for m, w in r.wrappers.items():
        a = w.a.detach().double().cpu()                     # (K r, in)
        b = w.b.detach().double().cpu()                     # (out, K r)
        qb, rb = torch.linalg.qr(b)
        qa, ra = torch.linalg.qr(a.T)
        uc, s, vch = torch.linalg.svd(rb @ ra.T)
        u, v = qb @ uc, qa @ vch.T
        out[m] = [((u * (alpha * s.mean())).float().numpy(), v.T.float().numpy(), 1.0)]
    return out


def run_context(cfg: dict, key: str, device: str) -> dict:
    r = get_runner(cfg["bank"], cfg["T"], device)
    r.clear()
    if cfg["method"] == "isoc":
        r.set_extras(isoc_extras(r, float(cfg["alpha"])))
    else:
        raise ValueError(cfg["method"])
    met, pred = r.evaluate(cfg["split"])
    r.set_extras(None)
    return {"metrics": summarize(met, expert_refs(cfg["bank"], cfg["T"], cfg["split"])),
            "preds": save_preds(cfg["bank"], cfg["T"], cfg["split"], key, pred)}


def load_edit(path: str) -> dict:
    """{module: (U_scaled (out x k), V^T (k x in))} with E = U_scaled @ V^T, the edit of ONE expert's matrices."""
    z = np.load(BK.ROOT / path, allow_pickle=False)
    mods = [str(m) for m in z["modules"]]
    return {m: (z[f"U_{i}"], z[f"V_{i}"]) for i, m in enumerate(mods)}, int(z["expert"]), str(z["edit_tag"])


def root_stats(r, split: str, task: int, nmax: int = 250) -> dict:
    """Root (first answer token) log-probs over U's first tokens and last-prompt-token hidden states for the first
    nmax examples of one task (function-change measurements)."""
    import torch

    ids, labels, task_of, index = r.split(split)
    rows = np.flatnonzero(task_of == task)[:nmax]
    LL = r.LL
    sub = [ids[i] for i in rows]
    if r.spec.multi_token:
        first = np.unique(r.ev.cands.first)
    else:
        first = np.asarray(r.ev.token_ids)
    lp, hs = np.zeros((len(rows), len(first)), np.float32), np.zeros((len(rows), r.model.config.hidden_size), np.float16)
    with torch.no_grad():
        for rr, x, mask, lens in LL.batches(sub, max_tokens=8192, max_batch=32):
            h = LL.last_hidden(r.model, x, mask, lens)
            lg = (h.float() @ r.model.lm_head.weight.float().T).log_softmax(-1)
            lp[rr] = lg[:, torch.as_tensor(first, device=lg.device)].cpu().numpy()
            hs[rr] = h.float().cpu().numpy().astype(np.float16)
    return {"rows": rows, "first_tokens": first, "logp": lp, "hidden": hs}


def run_tail(cfg: dict, key: str, device: str) -> dict:
    """Tail-edit cell: matrices in {orig, edited} x coefficients in {orig, recomputed}; rule 'uniform' allowed."""
    from condmerge.llm import rules

    r = get_runner(cfg["bank"], cfg["T"], device)
    edit, e_idx, tag = load_edit(cfg["edit_file"])
    b = float("inf") if cfg.get("b") in ("inf", float("inf")) else float(cfg.get("b", 0.0))
    if cfg["coeffs"] == "orig":
        lam, info = rules.build(cfg["bank"], cfg["T"], cfg["rule"], b)
    else:  # recomputed from the edited expert's scores (exact: the edit sets sigma_min to tau), SAME beta
        assert cfg["rule"] in ("kappa", "logkappa", "kappa_res", "kappa_rev"), cfg["rule"]
        z = np.load(BK.ROOT / cfg["edit_file"])
        orig = rules.score_table(cfg["bank"], cfg["T"], cfg["rule"])
        table = orig.copy()
        edited_scores = json.loads(str(z["edited_scores"]))[cfg["rule"]]
        for j, m in enumerate(r.mods):
            if m in edited_scores:
                table[j, e_idx] = edited_scores[m]
        lam, info = rules.coefficients(table, b, scale=rules.scale_of(orig))
        info.update({"n_eff_mean": float(rules.n_eff(lam).mean()), "lambda_bar": lam.mean(0).tolist(),
                     "scale_edited_table": rules.scale_of(table), "note": "beta kept at the original table's scale"})
    alpha = float(cfg["alpha"])
    chk = r.set_merge(lam, alpha)
    extras = {}
    if cfg["matrices"] == "edited":
        for j, m in enumerate(r.mods):
            if m in edit:
                u, v = edit[m]
                extras[m] = [(u, v, alpha * float(lam[j, e_idx]))]
    r.set_extras(extras)
    met, pred = r.evaluate(cfg["split"])
    out = {"metrics": summarize(met, expert_refs(cfg["bank"], cfg["T"], cfg["split"])), "rule_info": info,
           "lambda_edited_expert": [float(x) for x in lam[:, e_idx]], "lambda": np.round(lam, 6).tolist(),
           "check": chk, "edit_tag": tag, "edited_expert": r.tasks[e_idx],
           "preds": save_preds(cfg["bank"], cfg["T"], cfg["split"], key, pred)}
    r.set_extras(None)
    return out


def run_expert_edit(cfg: dict, key: str, device: str) -> dict:
    """The edited expert alone vs the original expert: own-task accuracy (split) and function change on its task."""
    r = get_runner(cfg["bank"], cfg["T"], device)
    edit, e_idx, tag = load_edit(cfg["edit_file"])
    nm, K = len(r.mods), r.K
    lam = np.zeros((nm, K))
    lam[:, e_idx] = 1.0
    res = {}
    stats = {}
    for cond in ("orig", "edited"):
        r.set_merge(lam, 1.0)
        r.set_extras({m: [(u, v, 1.0)] for m, (u, v) in edit.items()} if cond == "edited" else None)
        met, pred = r.evaluate(cfg["split"], only_task=e_idx)
        save_preds(cfg["bank"], cfg["T"], cfg["split"], f"{key}_{cond}", pred)
        res[cond] = met["local_per_task"][e_idx]
        res[cond + "_joint"] = met["joint_per_task"][e_idx]
        stats[cond] = root_stats(r, cfg["split"], e_idx)
    r.set_extras(None)
    a, b2 = stats["orig"], stats["edited"]

    def renorm(lp):  # log-probs renormalised over U's first tokens (the answer-relevant distribution)
        lp = lp.astype(np.float64)
        mx = lp.max(1, keepdims=True)
        return lp - (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))

    la, lb = renorm(a["logp"]), renorm(b2["logp"])
    pb = b2["logp"].astype(np.float64)
    kl = np.sum(np.exp(la) * (la - lb), axis=1)  # KL(orig || edited) over U's first tokens
    dh = np.linalg.norm(a["hidden"].astype(np.float64) - b2["hidden"].astype(np.float64), axis=1) / \
        np.maximum(np.linalg.norm(a["hidden"].astype(np.float64), axis=1), 1e-12)
    d = J.PREDS / f"{cfg['bank']}_T{cfg['T']}" / "function_change"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{key}.npz", kl=kl, rel_dh=dh, logp_orig=a["logp"], logp_edit=b2["logp"])
    return {"own_local": res, "edit_tag": tag, "edited_expert": r.tasks[e_idx],
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "max_abs_dlogp": float(np.abs(a["logp"] - pb).max()),
            "rel_dhidden_mean": float(dh.mean()), "rel_dhidden_max": float(dh.max()),
            "argmax_first_token_changed": float(np.mean(a["logp"].argmax(1) != b2["logp"].argmax(1)))}


def run_gradpref(cfg: dict, key: str, device: str) -> dict:
    """g[p, i] = d (mean val loss) / d lambda_i[p] at the merge (rule, b, alpha); loss = gold-answer NLL
    (multi-token: teacher-forced sequence NLL as in expert training; single-token: CE over the task's answers)."""
    import torch
    import torch.nn.functional as F

    from condmerge.llm import rules

    r = get_runner(cfg["bank"], cfg["T"], device)
    b = float("inf") if cfg.get("b") in ("inf", float("inf")) else float(cfg.get("b", 0.0))
    lam, info = rules.build(cfg["bank"], cfg["T"], cfg["rule"], b)
    alpha = float(cfg["alpha"])
    r.set_merge(lam, alpha)
    r.set_extras(None)
    ws = list(r.wrappers.values())
    for w in ws:
        w.w.requires_grad_(True)
        w.w.grad = None
    ids, labels, task_of, _ = r.split(cfg["split"])
    rng = np.random.default_rng(0)
    nper = int(cfg.get("n_per_task", 64))
    rows = np.concatenate([rng.permutation(np.flatnonzero(task_of == k))[:nper] for k in range(r.K)])
    total = len(rows)
    D, LL = r.D, r.LL
    t0 = time.time()
    if r.spec.multi_token:
        words = {t: D.answers(t) for t in r.tasks}
        seqs = [D.answer_seq(words[r.tasks[task_of[i]]][labels[i]]) for i in rows]
        prompts = [ids[i] for i in rows]
        for chunk in LL.token_chunks({j: list(prompts[j]) + list(seqs[j][:-1]) for j in range(total)},
                                     list(range(total)), 2048):
            full = [list(prompts[j]) + list(seqs[j][:-1]) for j in chunk]
            _, x, mask, lens = LL.collate(full, list(range(len(full))))
            h = r.model.model(input_ids=x.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
            rr, pos, tgt = [], [], []
            for q, j in enumerate(chunk):
                for k2 in range(len(seqs[j])):
                    rr.append(q), pos.append(len(prompts[j]) - 1 + k2), tgt.append(seqs[j][k2])
            lg = LL.head(r.model, h[torch.as_tensor(rr, device=device), torch.as_tensor(pos, device=device)])
            nll = F.cross_entropy(lg, torch.as_tensor(tgt, device=device), reduction="sum") / total
            nll.backward()
    else:
        sub = [ids[i] for i in rows]
        for rr, x, mask, lens in LL.batches(sub, max_tokens=4096, max_batch=32):
            h = LL.last_hidden(r.model, x, mask, lens)
            lg = LL.union_logits(r.model, h, r.ev.token_ids)
            loss = 0.0
            for q, j in enumerate(rr):
                i = rows[j]
                cols = torch.as_tensor(r.ev.maps[r.tasks[task_of[i]]], device=lg.device)
                loss = loss + F.cross_entropy(lg[q, cols][None], torch.as_tensor([labels[i]], device=lg.device))
            (loss / total).backward()
    g = np.stack([alpha * w.w.grad.detach().double().cpu().numpy().reshape(r.K, -1).sum(1) for w in ws])
    for w in ws:
        w.w.requires_grad_(False)
        w.w.grad = None
    d = J.PREDS / f"{cfg['bank']}_T{cfg['T']}" / "gradpref"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{key}.npy", g)
    return {"grad_file": str((d / f"{key}.npy").relative_to(BK.ROOT)), "n_examples": int(total),
            "seconds": time.time() - t0, "rule_info": info}


KINDS = {"merge": run_merge, "expert": run_expert, "base": run_base, "context": run_context, "tail": run_tail,
         "expert_edit": run_expert_edit, "gradpref": run_gradpref}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--idle-exit", type=float, default=0.0, help="exit after this many idle seconds (0 = never)")
    ap.add_argument("--prefer-bank", default=None, help="take this bank's jobs first whatever their priority")
    ap.add_argument("--only-bank", default=None, help="never take other banks' jobs (memory budget on a shared card)")
    args = ap.parse_args()
    import threading

    import torch

    torch.set_num_threads(6)
    d = J.dirs()
    for name in J.requeue_own(args.name):
        print(f"requeued interrupted job {name}", flush=True)
    beat = {"state": "starting"}

    def beater():  # liveness for the orchestrator's stale-job check, also during long jobs
        while True:
            try:
                J.heartbeat(args.name, dict(beat))
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(target=beater, daemon=True).start()
    idle_since = None
    while True:
        if (J.QROOT / "STOP").exists():
            print("STOP file: exiting", flush=True)
            return
        pref = args.prefer_bank or (State.key[0] if State.key else None)
        got = J.claim(args.name, pref, strict=bool(args.prefer_bank), loaded=State.key, only_bank=args.only_bank)
        if got is None:
            if (J.QROOT / "ALL_DONE").exists():
                print("ALL_DONE: exiting", flush=True)
                return
            idle_since = idle_since or time.time()
            if args.idle_exit and time.time() - idle_since > args.idle_exit:
                return
            beat.clear()
            beat.update({"state": "idle", "loaded": list(State.key) if State.key else None})
            time.sleep(20)
            continue
        idle_since = None
        path, spec = got
        cfg, key = spec["cfg"], spec["key"]
        beat.clear()
        beat.update({"state": "running", "job": key, "cfg": cfg})
        J.heartbeat(args.name, dict(beat))
        t0 = time.time()
        attempt = 0
        while True:
            try:
                out = KINDS[cfg["kind"]](cfg, key, args.device)
                rec = {"key": key, "cfg": cfg, **out, "worker": args.name, "commit": J.commit(),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_seconds": time.time() - t0}
                J.append(J.result_dir(cfg["bank"], cfg["T"], cfg["group"]) / f"{args.name}.jsonl", rec)
                J.finish(path, True)
                m = out.get("metrics", {})
                print(f"{time.strftime('%H:%M:%S')} done {cfg['kind']} {cfg.get('rule', cfg.get('method', ''))} "
                      f"b={cfg.get('b')} a={cfg.get('alpha')} {cfg.get('split')} "
                      f"norm={m.get('normalized', float('nan')):.4f} J={m.get('joint_macro', float('nan')):.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                attempt += 1
                torch.cuda.empty_cache()
                if attempt <= 2:
                    print(f"OOM on {key}, retry {attempt}", flush=True)
                    time.sleep(30)
                    continue
                J.finish(path, False, "OOM\n" + traceback.format_exc())
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"FAILED {key}\n{tb}", flush=True)
                J.finish(path, False, tb)
                if "CUDA error" in tb or "device-side assert" in tb:
                    sys.exit(3)  # corrupted CUDA context: let the wrapper restart the process
                break


if __name__ == "__main__":
    main()
