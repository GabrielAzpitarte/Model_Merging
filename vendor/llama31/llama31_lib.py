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

import llama31_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("LLAMA31_OUT", D.STORE / "outputs"))  # smoke tests redirect all outputs
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


def load_model(device="cuda:0", dtype=torch.bfloat16):
    os.environ.setdefault("HF_HOME", str(D.HF_HOME))
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(D.MODEL, dtype=dtype, attn_implementation="sdpa")
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
    x = torch.full((len(rows), n), D.PAD, dtype=torch.long)
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


# ----------------------------------------------------------------------------------------------- multi-token scoring
class Cands:
    """A prefix-free set of complete answer sequences (tokens(label) + <|eot_id|>): the bank's union U or one task's
    H_t. score(y | x) = log p(y | x) = sum_i log p(y_i | x, y_<i) (pre-registered: SUM, protocol section 4)."""

    def __init__(self, seqs):
        self.seqs = [tuple(int(t) for t in s) for s in seqs]
        full = set(self.seqs)
        assert len(full) == len(self.seqs), "duplicate answer sequences"
        assert not any(s[:k] in full for s in self.seqs for k in range(1, len(s))), "answer set is not prefix-free"
        self.first = np.asarray([s[0] for s in self.seqs])

    def __len__(self):
        return len(self.seqs)


def prune(root, cand_idx, cands, delta=1e-3, kmin=4):
    """Candidates (indices into cands, a subset of cand_idx) whose first token is among the most probable first tokens
    of cand_idx covering >= 1 - delta of their first-token mass (at least kmin first tokens). root: (V,) log-probs.
    Every excluded candidate y has log p(y|x) <= log p(y_1|x) (all terms are <= 0): bound used by the certificates."""
    cand_idx = np.asarray(cand_idx)
    f = cands.first[cand_idx]
    uf = np.unique(f)
    lp = root[uf].astype(np.float64)
    p = np.exp(lp - lp.max())
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order]) / p.sum()
    n = min(len(uf), max(kmin, int(np.searchsorted(cum, 1 - delta)) + 1))
    return cand_idx[np.isin(f, uf[order[:n]])]


def _tree(seq_lists):
    """Trie nodes (distinct proper prefixes) of each example's candidate sequences: node tokens, depths, ancestor
    matrix anc[b, j, j'] (node j' is a prefix of node j, j itself included), validity, and the scoring terms
    (example, candidate slot, node, next token) of every non-first answer token."""
    B = len(seq_lists)
    nodes = []
    for seqs in seq_lists:
        nd = {}
        for s in seqs:
            for k in range(1, len(s)):
                nd.setdefault(s[:k], len(nd))
        nodes.append(nd)
    M = max([len(nd) for nd in nodes] + [1])
    tok = torch.full((B, M), D.PAD, dtype=torch.long)
    depth = torch.ones((B, M), dtype=torch.long)
    anc = torch.zeros((B, M, M), dtype=torch.bool)
    valid = torch.zeros((B, M), dtype=torch.bool)
    tb, ts, tj, tt = [], [], [], []
    for b, nd in enumerate(nodes):
        for p, j in nd.items():
            tok[b, j] = p[-1]
            depth[b, j] = len(p)
            valid[b, j] = True
            for k in range(1, len(p) + 1):
                anc[b, j, nd[p[:k]]] = True
        for i, s in enumerate(seq_lists[b]):
            for k in range(1, len(s)):
                tb.append(b), ts.append(i), tj.append(nd[s[:k]]), tt.append(s[k])
    terms = tuple(torch.as_tensor(v, dtype=torch.long) for v in (tb, ts, tj, tt))
    return tok, depth, anc, valid, terms, M


def head(model, h):
    """Output logits in FP32 (a frozen FP32 copy of lm_head): BF16 logits add ~0.1-nat rounding noise to the
    full-vocabulary log-softmax of every answer token."""
    w = getattr(model, "_head32", None)
    if w is None:
        w = model.lm_head.weight.detach().float()
        object.__setattr__(model, "_head32", w)
    return h.float() @ w.T


def _gather_logp(model, h, targets, chunk=2048):
    """log p(target | row) over the full vocabulary for hidden rows h (R, H), chunked over rows."""
    out = []
    for s in range(0, len(h), chunk):
        lg = head(model, h[s:s + chunk])
        out.append(lg.gather(1, targets[s:s + chunk, None]).squeeze(1) - torch.logsumexp(lg, 1))
    return torch.cat(out) if out else torch.zeros(0, device=h.device)


def _assemble(model, root, h_nodes, terms, seq_lists):
    """(B, K) sum scores (-inf padding) from the root log-probs (B, V) and the node hidden states (B, M, H)."""
    dev = root.device
    B, K = len(seq_lists), max([len(s) for s in seq_lists] + [1])
    first = torch.zeros((B, K), dtype=torch.long)
    ok = torch.zeros((B, K), dtype=torch.bool)
    for b, seqs in enumerate(seq_lists):
        for i, s in enumerate(seqs):
            first[b, i], ok[b, i] = s[0], True
    ok = ok.to(dev)
    scores = root.gather(1, first.to(dev))
    tb, ts, tj, tt = (t.to(dev) for t in terms)
    if len(tb):
        vals = _gather_logp(model, h_nodes[tb, tj], tt)
        scores = scores.index_put((tb, ts), vals, accumulate=True)
    return scores.masked_fill(~ok, float("-inf"))


def prompt_pass(model, x, mask, lens, cache=True):
    """Prompt forward: full-vocabulary log-probs of the first answer token (B, V) and the KV cache."""
    dev = model.lm_head.weight.device
    out = model.model(input_ids=x.to(dev), attention_mask=mask.to(dev), use_cache=cache)
    h = out.last_hidden_state[torch.arange(len(lens), device=dev), (lens - 1).to(dev)]
    return head(model, h).log_softmax(-1), out.past_key_values


def score_cached(model, cache, lens, n_prompt, root, seq_lists):
    """No-grad sum scores of seq_lists (per example lists of answer sequences) on top of the prompts' KV cache: all
    trie nodes in one pass under a tree mask (node -> its prompt and its ancestors). The cache is cropped back."""
    tok, depth, anc, valid, terms, M = _tree(seq_lists)
    dev = root.device
    h = None
    if valid.any():
        B = len(seq_lists)
        pos = lens[:, None] + depth - 1
        m = torch.zeros((B, 1, M, n_prompt + M), dtype=torch.bool)
        m[:, 0, :, :n_prompt] = (torch.arange(n_prompt)[None, :] < lens[:, None])[:, None, :]
        m[:, 0, :, n_prompt:] = anc
        h = model.model(input_ids=tok.to(dev), attention_mask=m.to(dev), position_ids=pos.to(dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        cache.crop(-M)  # drop the node entries, keep the prompt KV
    return _assemble(model, root, h, terms, seq_lists)


def score_packed(model, ids, rows, seq_lists, embeds_grad=False):
    """Differentiable sum scores: ONE forward over [prompt (right-padded) | trie nodes] with a 4D tree mask.
    Returns (B, K) scores and the (B, S) validity of every position (prompt tokens and nodes; for K-FAC hooks)."""
    _, x, mask2, lens = collate(ids, list(rows))
    tok, depth, anc, valid, terms, M = _tree(seq_lists)
    B, L = x.shape
    dev = model.lm_head.weight.device
    pv = mask2.bool()
    m = torch.zeros((B, 1, L + M, L + M), dtype=torch.bool)
    m[:, 0, :L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))[None] & pv[:, None, :]
    m[:, 0, L:, :L] = pv[:, None, :]
    m[:, 0, L:, L:] = anc
    pos = torch.cat([torch.arange(L)[None].expand(B, -1), lens[:, None] + depth - 1], 1)
    inp = torch.cat([x, tok], 1).to(dev)
    # embeds_grad: a leaf requiring grad at the input, so that output gradients exist with frozen weights (K-FAC)
    kw = {"inputs_embeds": model.model.embed_tokens(inp).detach().requires_grad_(True)} if embeds_grad else \
        {"input_ids": inp}
    h = model.model(attention_mask=m.to(dev), position_ids=pos.to(dev), use_cache=False, **kw).last_hidden_state
    root = head(model, h[torch.arange(B, device=dev), (lens - 1).to(dev)]).log_softmax(-1)
    return _assemble(model, root, h[:, L:], terms, seq_lists), torch.cat([pv, valid], 1)


@torch.no_grad()
def root_logprobs(model, ids, rows):
    _, x, mask, lens = collate(ids, list(rows))
    return prompt_pass(model, x, mask, lens, cache=False)[0].cpu().numpy()


@torch.no_grad()
def score_rows(model, ids, cands, choose, max_tokens=8192, max_batch=32):
    """Dense (N, |cands|) sum scores, -inf where not scored. choose(row, root (V,) log-probs) -> candidate indices."""
    out = np.full((len(ids), len(cands)), -np.inf, np.float32)
    for rows, x, mask, lens in batches(ids, max_tokens, max_batch):
        root, cache = prompt_pass(model, x, mask, lens)
        rn = root.cpu().numpy()
        sel = [np.asarray(choose(int(r), rn[b])) for b, r in enumerate(rows)]
        sc = score_cached(model, cache, lens, x.shape[1], root, [[cands.seqs[i] for i in s] for s in sel])
        sc = sc.cpu().numpy()
        for b, r in enumerate(rows):
            out[r, sel[b]] = sc[b, :len(sel[b])]
        del cache
    return out


# ----------------------------------------------------------------------------------------------- evaluation
class Evaluator:
    """Scores static cores on the val / test examples of the T tasks. Answers are multi-token; per example:
      joint  EXACT argmax over U of the sum score (no task ID: all answers of the bank compete)
      local  EXACT argmax over H_t
    Exactness: U and H_t are first pruned by first-token mass; then a certificate adds every unscored candidate whose
    first-token log-prob (an upper bound of its score) exceeds the best scored score -- over U for joint, over H_t for
    local -- until none remains, so both argmaxes equal those of the full candidate sets.
    free (the 3B whole-vocabulary metric) = the greedy first token is the gold answer's first token."""

    def __init__(self, model, wrappers, tasks, delta=1e-3):
        self.model, self.wrappers, self.tasks = model, wrappers, list(tasks)
        self.words, seqs, self.maps = D.union(self.tasks)
        self.cands = Cands(seqs)
        self.delta = delta
        self.sets = {}
        self.stats = {"scored": 0, "rows": 0, "rounds": 0}

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
    def predict(self, ids, task_of):
        """(joint argmax U-index, local argmax index in H_t, greedy first token) per prompt, exact."""
        n, nu = len(ids), len(self.cands)
        pj, pl, top = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
        all_idx = np.arange(nu)
        for rows, x, mask, lens in batches(ids, max_tokens=8192, max_batch=32):  # 64: OOM on flat (zero-shot) models
            root, cache = prompt_pass(self.model, x, mask, lens)
            rn = root.cpu().numpy()
            B = len(rows)
            S = np.full((B, nu), -np.inf)
            done = np.zeros((B, nu), bool)
            local = [self.maps[self.tasks[task_of[r]]] for r in rows]
            todo = [np.union1d(prune(rn[b], local[b], self.cands, self.delta, kmin=1),
                               prune(rn[b], all_idx, self.cands, self.delta, kmin=1)) for b in range(B)]
            bound = rn[:, self.cands.first]
            while any(len(t) for t in todo):
                sc = score_cached(self.model, cache, lens, x.shape[1], root,
                                  [[self.cands.seqs[i] for i in t] for t in todo]).cpu().numpy()
                for b, t in enumerate(todo):
                    S[b, t], done[b, t] = sc[b, :len(t)], True
                    self.stats["scored"] += len(t)
                nxt = []
                for b in range(B):
                    need = ~done[b] & (bound[b] > S[b].max())
                    loc = local[b][~done[b, local[b]]]
                    nxt.append(np.union1d(np.flatnonzero(need), loc[bound[b, loc] > S[b, local[b]].max()]))
                todo = nxt
                self.stats["rounds"] += 1
            self.stats["rows"] += B
            for b, r in enumerate(rows):
                pj[r] = int(S[b].argmax())
                pl[r] = int(S[b, self.maps[self.tasks[task_of[r]]]].argmax())
                top[r] = int(rn[b].argmax())
            del cache
        return pj, pl, top

    def score(self, cores, split, only=None):
        """only: a task index -> score that task's examples only (the other tasks' entries are NaN)."""
        set_cores(self.wrappers, cores)
        ids, labels, task_of = self.split(split)
        rows = np.arange(len(ids)) if only is None else np.flatnonzero(task_of == only)
        pj, pl, top = self.predict([ids[i] for i in rows], task_of[rows])
        set_cores(self.wrappers, None)
        return self.metrics(pj, pl, top, labels[rows], task_of[rows])

    def score_dense(self, deltas, split):
        for m, w in self.wrappers.items():
            w.set_dense(deltas[m])
        ids, labels, task_of = self.split(split)
        pj, pl, top = self.predict(ids, task_of)
        set_cores(self.wrappers, None)
        return self.metrics(pj, pl, top, labels, task_of)

    def metrics(self, pj, pl, top, labels, task_of):
        joint, local, free, n = [], [], [], []
        for k, t in enumerate(self.tasks):
            sel = task_of == k
            if not sel.any():
                joint.append(float("nan")), local.append(float("nan")), free.append(float("nan")), n.append(0)
                continue
            gold_u = self.maps[t][labels[sel]]
            joint.append(float((pj[sel] == gold_u).mean()))
            local.append(float((pl[sel] == labels[sel]).mean()))
            free.append(float((top[sel] == self.cands.first[gold_u]).mean()))
            n.append(int(sel.sum()))
        return {"tasks": self.tasks, "joint_per_task": joint, "local_per_task": local, "free_per_task": free, "n": n,
                "joint_macro": float(np.nanmean(joint)), "local_macro": float(np.nanmean(local)),
                "free_macro": float(np.nanmean(free))}

    def task_scores(self, ids, task):
        """(N, |H_t|) exact sum scores over one task's answers (the expert posteriors of the router target)."""
        cols = self.maps[task]
        return score_rows(self.model, ids, self.cands, lambda r, root: cols)[:, cols]

    def posterior(self, ids, delta=1e-4):
        """(N, |U|) softmax over U of the sum scores; candidates pruned by first-token mass (< delta of U's first-token
        mass) get probability 0."""
        all_idx = np.arange(len(self.cands))
        s = score_rows(self.model, ids, self.cands, lambda r, root: prune(root, all_idx, self.cands, delta))
        return torch.as_tensor(s).softmax(-1).numpy()


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
