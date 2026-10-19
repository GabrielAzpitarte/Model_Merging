"""Shared harness for the Llama-3.2-3B-Instruct text port (llama_pipeline.py, llama_train_lora.py).

Every model is the frozen BF16 backbone plus one static update per LoRA module (q/k/v/o of the 28 blocks),

    y = W0 x + U_B C V_A^T x,

computed on a separate FP32 side path (CoreLinear): adding a LoRA delta into BF16 weights would round it away.
C is (q_out x q_in) with q = 16 T (CoreSpace of the T-expert bank); rank(C) is the deployed rank.

Readout: the next-token logits at the last prompt token (the assistant header), restricted to the bank's answer
tokens U ("joint head"). Metrics per task (llama_data.py splits):
  joint  argmax over U == gold answer          (no task ID: all answers of the bank compete)
  local  argmax over H_t == gold answer        (task-restricted, as the vision local head)
  free   argmax over the whole 128k vocabulary == gold answer token
normalized = mean over tasks of local / (the task's own expert local accuracy on the same split), exactly
lowrank_lib.add_normalized.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llama_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = D.STORE / "outputs"
BANK = OUT / "lora"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def module_names(n_layers):
    return [f"model.layers.{i}.self_attn.{p}" for i in range(n_layers) for p in PROJ]


def resolve(model, name):
    mod = model
    for part in name.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def replace_module(model, name, value):
    parts = name.split(".")
    parent = resolve(model, ".".join(parts[:-1]))
    setattr(parent, parts[-1], value)


def load_model(device="cuda:0"):
    os.environ.setdefault("HF_HOME", str(D.HF_HOME))
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(D.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device).eval()
    model.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------------------------- CoreSpace
def load_core(T, tasks=None):
    """{module: CoreSpace} of the first T bank experts (float32 arrays), cached."""
    from icamerge.core_space import LoRAExpert, build_core_space

    tasks = D.TASKSETS[T] if tasks is None else tasks
    path = OUT / "cores" / f"T{T}.pkl"
    if path.exists():
        return pickle.load(path.open("rb"))
    blobs = {t: np.load(BANK / f"{t}.npz") for t in tasks}
    names = [k[2:] for k in blobs[tasks[0]].files if k.startswith("A_")]
    names.sort(key=lambda n: (int(n.split(".")[2]), PROJ.index(n.split(".")[-1])))
    core = {}
    for m in names:
        cs = build_core_space([LoRAExpert(t, blobs[t][f"A_{m}"], blobs[t][f"B_{m}"]) for t in tasks], module=m)
        cs.u_b, cs.v_a, cs.cores = (cs.u_b.astype(np.float32), cs.v_a.astype(np.float32), cs.cores.astype(np.float32))
        core[m] = cs
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    pickle.dump(core, tmp.open("wb"))
    os.replace(tmp, path)
    return core


class CoreLinear(nn.Module):
    """base(x) + U_B C V_A^T x on an FP32 side path. C is either fixed (set_core) or trainable:
    directional  C = C0 + sum_t C_t diag(phi_t)   (TARA-like, the stage-1 writer; lowrank_lib.CoreWriter)
    factored     C = P Q^T                          (a rank-r LoRA, stages 2-3)"""

    def __init__(self, base, cs, device):
        super().__init__()
        self.base = base
        self.register_buffer("u_b", torch.as_tensor(np.asarray(cs.u_b, np.float32), device=device))
        self.register_buffer("v_a", torch.as_tensor(np.asarray(cs.v_a, np.float32), device=device))
        self.fixed = None
        self.mode = None
        self.dense = None

    def set_core(self, c):
        self.mode = None
        self.dense = None
        self.fixed = None if c is None else torch.as_tensor(np.asarray(c, np.float32), device=self.u_b.device)

    def set_dense(self, dw):
        """A dense update Delta W (out x in) outside the CoreSpace (RegMean-family solutions leave span(V_A))."""
        self.mode, self.fixed = None, None
        if dw is None or isinstance(dw, torch.Tensor):
            self.dense = None if dw is None else dw.to(self.u_b.device, torch.float32)
        else:
            self.dense = torch.as_tensor(np.asarray(dw, np.float32), device=self.u_b.device)

    def make_rankdirs(self, p, q, init):
        """TARA-A: C = sum_t P_t diag(phi_t) Q_t^T over each expert's own rank-1 LoRA directions."""
        dev = self.u_b.device
        self.mode, self.fixed, self.dense = "rankdirs", None, None
        self.rp = torch.as_tensor(np.asarray(p, np.float32), device=dev)   # (T, q_out, r)
        self.rq = torch.as_tensor(np.asarray(q, np.float32), device=dev)   # (T, q_in, r)
        self.phi = nn.Parameter(torch.full((self.rp.shape[0], self.rp.shape[2]), float(init), device=dev))
        return [self.phi]

    def make_taskwise(self, cores_t, init):
        """AdaMerging (layer-wise): C = sum_t lambda_t C_t, one scalar per task for this module."""
        dev = self.u_b.device
        self.mode, self.fixed, self.dense = "taskwise", None, None
        self.g = torch.as_tensor(np.asarray(cores_t, np.float32), device=dev)
        self.lam = nn.Parameter(torch.full((self.g.shape[0],), float(init), device=dev))
        return [self.lam]

    def make_directional(self, cores_t, c0, init=0.0):
        dev = self.u_b.device
        self.mode, self.fixed, self.dense = "directional", None, None
        self.g = torch.as_tensor(np.asarray(cores_t, np.float32), device=dev)
        self.c0 = torch.as_tensor(np.asarray(c0, np.float32), device=dev)
        self.phi = nn.Parameter(torch.full((self.g.shape[0], self.g.shape[2]), float(init), device=dev))
        return [self.phi]

    def make_factored(self, p, q):
        dev = self.u_b.device
        self.mode, self.fixed = "factored", None
        self.p = nn.Parameter(torch.as_tensor(np.asarray(p, np.float32), device=dev))
        self.q = nn.Parameter(torch.as_tensor(np.asarray(q, np.float32), device=dev))
        return [self.p, self.q]

    def core(self):
        if self.mode == "directional":
            return self.c0 + torch.einsum("toi,ti->oi", self.g, self.phi)
        if self.mode == "factored":
            return self.p @ self.q.T
        if self.mode == "rankdirs":
            return torch.einsum("tor,tr,tir->oi", self.rp, self.phi, self.rq)
        if self.mode == "taskwise":
            return torch.einsum("toi,t->oi", self.g, self.lam)
        return self.fixed

    def forward(self, x):
        y = self.base(x)
        if self.dense is not None:
            return y + F.linear(x.float(), self.dense).to(y.dtype)
        c = self.core()
        if c is None:
            return y
        upd = F.linear(F.linear(F.linear(x.float(), self.v_a.T), c), self.u_b)
        return y + upd.to(y.dtype)


def install(model, core, device):
    wrappers = {}
    for m, cs in core.items():
        w = CoreLinear(resolve(model, m), cs, device)
        replace_module(model, m, w)
        wrappers[m] = w
    return wrappers


def set_cores(wrappers, cores):
    for m, w in wrappers.items():
        w.set_core(None if cores is None else cores[m])


def current_cores(wrappers):
    with torch.no_grad():
        return {m: w.core().detach().cpu().numpy().astype(np.float64) for m, w in wrappers.items()}


# ----------------------------------------------------------------------------------------------- batching
def encode(texts):
    tok = D.tokenizer()
    return [tok.encode(D.prompt(x), add_special_tokens=False) for x in texts]


def batches(ids, max_tokens=12288, max_batch=64):
    """Length-sorted batches: yields (row indices, input_ids, attention_mask, lengths) with right padding."""
    order = np.argsort([len(x) for x in ids], kind="stable")
    cur = []
    for i in order:
        if cur and (len(cur) + 1) * len(ids[i]) > max_tokens or len(cur) == max_batch:
            yield collate(ids, cur)
            cur = []
        cur.append(int(i))
    if cur:
        yield collate(ids, cur)


def token_chunks(ids, rows, max_tokens=3072):
    """Split rows (one optimisation step) into length-sorted micro-batches of at most max_tokens padded tokens."""
    rows = sorted(rows, key=lambda i: len(ids[i]))
    out, cur = [], []
    for i in rows:
        if cur and (len(cur) + 1) * len(ids[i]) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def collate(ids, rows):
    lens = [len(ids[i]) for i in rows]
    n = max(lens)
    pad = D.tokenizer().pad_token_id
    x = torch.full((len(rows), n), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for j, i in enumerate(rows):
        x[j, :lens[j]] = torch.as_tensor(ids[i])
        mask[j, :lens[j]] = 1
    return np.asarray(rows), x, mask, torch.as_tensor(lens)


def last_hidden(model, x, mask, lens, inputs_embeds=None):
    dev = model.lm_head.weight.device
    kw = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": x.to(dev)}
    h = model.model(attention_mask=mask.to(dev), use_cache=False, **kw).last_hidden_state
    return h[torch.arange(len(lens), device=dev), (lens - 1).to(dev)]


def union_logits(model, h, token_ids):
    w = model.lm_head.weight[torch.as_tensor(token_ids, device=h.device)]
    return h.float() @ w.float().T


# ----------------------------------------------------------------------------------------------- evaluation
class Evaluator:
    """Scores static cores on the val / test examples of the T tasks (joint over U, local over H_t, free)."""

    def __init__(self, model, wrappers, tasks):
        self.model, self.wrappers, self.tasks = model, wrappers, list(tasks)
        self.words, self.token_ids, self.maps = D.union(self.tasks)
        self.sets = {}

    def split(self, split):
        if split not in self.sets:
            texts, labels, task_of = [], [], []
            for k, t in enumerate(self.tasks):
                val, test = D.eval_indices(t)
                ex = D.examples(t, "eval")
                for i in (val if split == "val" else test):
                    texts.append(ex[i][0])
                    labels.append(ex[i][1])
                    task_of.append(k)
            self.sets[split] = (encode(texts), np.asarray(labels), np.asarray(task_of))
        return self.sets[split]

    @torch.no_grad()
    def outputs(self, ids, free=False):
        """(N, |U|) union logits and, if free, the full-vocabulary argmax token per prompt."""
        n = len(ids)
        lg = np.zeros((n, len(self.token_ids)), np.float32)
        top = np.zeros(n, np.int64)
        for rows, x, mask, lens in batches(ids):
            h = last_hidden(self.model, x, mask, lens)  # BF16 backbone, FP32 CoreLinear side paths (no autocast)
            lg[rows] = union_logits(self.model, h, self.token_ids).cpu().numpy()
            if free:
                top[rows] = self.model.lm_head(h).argmax(-1).cpu().numpy()
        return lg, top

    def score(self, cores, split, only=None):
        """only: a task index -> score that task's examples only (the other tasks' entries are NaN)."""
        set_cores(self.wrappers, cores)
        ids, labels, task_of = self.split(split)
        rows = np.arange(len(ids)) if only is None else np.flatnonzero(task_of == only)
        lg, top = self.outputs([ids[i] for i in rows], free=True)
        set_cores(self.wrappers, None)
        return self.metrics(lg, top, labels[rows], task_of[rows])

    def score_dense(self, deltas, split):
        for m, w in self.wrappers.items():
            w.set_dense(deltas[m])
        ids, labels, task_of = self.split(split)
        lg, top = self.outputs(ids, free=True)
        set_cores(self.wrappers, None)
        return self.metrics(lg, top, labels, task_of)

    def metrics(self, lg, top, labels, task_of):
        joint, local, free, n = [], [], [], []
        for k, t in enumerate(self.tasks):
            sel = task_of == k
            if not sel.any():
                joint.append(float("nan")), local.append(float("nan")), free.append(float("nan")), n.append(0)
                continue
            gold_u = self.maps[t][labels[sel]]
            joint.append(float((lg[sel].argmax(1) == gold_u).mean()))
            local.append(float((lg[sel][:, self.maps[t]].argmax(1) == labels[sel]).mean()))
            free.append(float((top[sel] == self.token_ids[gold_u]).mean()))
            n.append(int(sel.sum()))
        return {"tasks": self.tasks, "joint_per_task": joint, "local_per_task": local, "free_per_task": free, "n": n,
                "joint_macro": float(np.nanmean(joint)), "local_macro": float(np.nanmean(local)),
                "free_macro": float(np.nanmean(free))}


def add_normalized(res, ref):
    """ref: the experts' own local accuracy per task on the same split (lowrank_lib.add_normalized)."""
    res["normalized"] = float(np.mean(np.asarray(res["local_per_task"]) / np.asarray(ref)))
    return res


# ----------------------------------------------------------------------------------------------- core algebra
def flat(m):
    """Iso-C: the spectrum of m replaced by its mean (same singular vectors); rectangular-safe."""
    u, s, vt = np.linalg.svd(np.asarray(m, np.float64), full_matrices=False)
    return s.mean() * (u @ vt)


def balanced_pq(c, r):
    u, s, vt = np.linalg.svd(np.asarray(c, np.float64), full_matrices=False)
    rs = np.sqrt(s[:r])
    return u[:, :r] * rs, vt[:r].T * rs


def sqrtm_psd(g, eps=1e-12):
    w, v = np.linalg.eigh(g)
    return (v * np.sqrt(np.clip(w, eps, None))) @ v.T


def damp(g, lam):
    return g + lam * np.trace(g) / len(g) * np.eye(len(g))


def kfac_components(c, gram_in, gram_out, lam=1e-2):
    """lowrank_lib.kfac_components (rectangular): A = Omega^1/2 C G^1/2 = U S V^T; top-r = argmin of the K-FAC loss."""
    gi, go = sqrtm_psd(damp(gram_in, lam)), sqrtm_psd(damp(gram_out, lam))
    u, s, vt = np.linalg.svd(go @ c @ gi, full_matrices=False)
    return u, s, vt, np.linalg.inv(go), np.linalg.inv(gi)


def kfac_reconstruct(parts, keep):
    u, s, vt, igo, igi = parts
    idx = np.asarray(keep, dtype=int)
    return igo @ ((u[:, idx] * s[idx]) @ vt[idx]) @ igi


def save_cores(path, cores):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: torch.as_tensor(np.asarray(c, np.float32)) for n, c in cores.items()}, path)


def load_cores(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return {n: blob[n].double().numpy() for n in blob}
