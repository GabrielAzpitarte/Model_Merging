"""Exact merged LoRA models on one GPU, and their evaluation with the projects' own (vendored) evaluators.

Merge of K experts with per-map coefficients lambda[p] (summing to 1) and global scale alpha:

    W_merge[p] = W0[p] + alpha * sum_i lambda_i[p] * B_i[p] A_i[p]

is applied EXACTLY by LoRA concatenation (never by averaging A and B separately):

    y = W0 x + B_cat diag(w) A_cat x,   A_cat = [A_1; ...; A_K] (K r x in),  B_cat = [B_1 ... B_K] (out x K r),
    w = alpha * (lambda_1 1_r, ..., lambda_K 1_r),

on an FP32 side path added to the BF16 base output (the evaluators' own convention: adding a LoRA delta into
BF16 weights would round it away). An expert is w = one-hot block. Extra low-rank terms c * U (V x) carry
weight edits (tail-edit intervention) without densifying.

Every evaluation records, besides metrics and per-example predictions:
  * the coefficient matrix lambda (maps x K) actually requested and alpha;
  * per-map merged-update Frobenius norms  ||B diag(w) A||_F  (exact, from R x R Gram products);
  * a fingerprint: sum over maps of u_p^T (merged update) v_p for fixed random probes (seeded per map), computed
    twice -- from the requested coefficients on CPU and from the GPU buffers actually installed -- and asserted
    equal, so a mislabelled or silently uniform merge cannot pass as another method.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import banks as BK

R = 16


class ConcatLoRA(nn.Module):
    def __init__(self, base: nn.Module, a_cat: np.ndarray, b_cat: np.ndarray, device: str):
        super().__init__()
        self.base = base
        self.register_buffer("a", torch.as_tensor(a_cat, dtype=torch.float32, device=device))
        self.register_buffer("b", torch.as_tensor(b_cat, dtype=torch.float32, device=device))
        self.register_buffer("w", torch.zeros(a_cat.shape[0], dtype=torch.float32, device=device))
        self.extra: list[tuple[torch.Tensor, torch.Tensor, float]] = []  # (U (out x k), V (k x in), coef)
        self.active = False

    def set_w(self, w: np.ndarray) -> None:
        self.w.copy_(torch.as_tensor(np.asarray(w, np.float32), device=self.w.device))
        self.active = bool(np.any(np.asarray(w) != 0))

    def forward(self, x):
        y = self.base(x)
        if not self.active and not self.extra:
            return y
        xf = x.float()
        upd = None
        if self.active:
            upd = F.linear(F.linear(xf, self.a) * self.w, self.b)
        for u, v, c in self.extra:
            if c == 0.0:
                continue
            e = c * F.linear(F.linear(xf, v), u)
            upd = e if upd is None else upd + e
        return y if upd is None else y + upd.to(y.dtype)


class Runner:
    """One bank (first T experts), model loaded once; set_merge() then evaluate(split)."""

    def __init__(self, bank: str, T: int, device: str = "cuda:0"):
        t0 = time.time()
        self.bank, self.T, self.device = bank, T, device
        self.spec = BK.BANKS[bank]
        self.D, self.LL = BK.modules(bank)
        self.tasks = BK.tasks(bank, T)
        self.K = len(self.tasks)
        self.mods = BK.module_names(bank)
        if self.spec.multi_token:
            self.model = self.LL.load_model(device, dtype=torch.bfloat16)
        else:
            self.model = self.LL.load_model(device)
        factors = {t: BK.load_lora(bank, t) for t in self.tasks}
        self.wrappers: dict[str, ConcatLoRA] = {}
        self.H, self.bu, self.av = {}, {}, {}
        for j, m in enumerate(self.mods):
            a_cat = np.concatenate([factors[t][m][0] for t in self.tasks], 0)        # (K r, in)
            b_cat = np.concatenate([factors[t][m][1] for t in self.tasks], 1)        # (out, K r)
            base = self.LL.resolve(self.model, m)
            assert isinstance(base, nn.Linear) and base.weight.shape == (b_cat.shape[0], a_cat.shape[1]), m
            w = ConcatLoRA(base, a_cat, b_cat, device)
            self.LL.replace_module(self.model, m, w)
            self.wrappers[m] = w
            a64, b64 = a_cat.astype(np.float64), b_cat.astype(np.float64)
            self.H[m] = (b64.T @ b64) * (a64 @ a64.T)                               # ||B W A||^2 = w^T H w
            rng = np.random.default_rng(1000 + j)
            u, v = rng.standard_normal(b64.shape[0]), rng.standard_normal(a64.shape[1])
            self.bu[m], self.av[m] = b64.T @ u, a64 @ v                              # u^T B W A v = sum bu w av
        self.ev = self.LL.Evaluator(self.model, {}, self.tasks)
        self.splits: dict = {}
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------------------------------- merge state
    def weights(self, lam: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
        lam = np.asarray(lam, np.float64)
        assert lam.shape == (len(self.mods), self.K), lam.shape
        assert np.all(np.isfinite(lam)) and np.all(lam >= -1e-12), "coefficients must be finite, non-negative"
        return {m: alpha * np.repeat(lam[j], R) for j, m in enumerate(self.mods)}

    def set_merge(self, lam: np.ndarray, alpha: float) -> dict:
        ws = self.weights(lam, alpha)
        for m, w in ws.items():
            self.wrappers[m].set_w(w)
        return self.verify(ws)

    def set_extras(self, extras: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] | None) -> None:
        for m, w in self.wrappers.items():
            w.extra = []
        for m, lst in (extras or {}).items():
            self.wrappers[m].extra = [(torch.as_tensor(u, dtype=torch.float32, device=self.device),
                                       torch.as_tensor(v, dtype=torch.float32, device=self.device), float(c))
                                      for u, v, c in lst]

    def clear(self) -> None:
        for w in self.wrappers.values():
            w.set_w(np.zeros(w.w.shape[0]))
            w.extra = []

    def verify(self, ws: dict[str, np.ndarray]) -> dict:
        """Norms + fingerprint from the requested weights and from the installed GPU buffers (must agree)."""
        norms, fp_req, fp_gpu = {}, 0.0, 0.0
        for m, w in ws.items():
            norms[m] = float(np.sqrt(max(w @ self.H[m] @ w, 0.0)))
            fp_req += float(np.sum(self.bu[m] * w * self.av[m]))
            wg = self.wrappers[m].w.detach().double().cpu().numpy()
            fp_gpu += float(np.sum(self.bu[m] * wg * self.av[m]))
        assert abs(fp_req - fp_gpu) <= 1e-5 * max(1.0, abs(fp_req)), ("installed state differs", fp_req, fp_gpu)
        return {"fingerprint": fp_req, "fingerprint_gpu": fp_gpu, "update_norm": norms,
                "update_norm_total": float(np.sqrt(sum(v * v for v in norms.values())))}

    # ------------------------------------------------------------------------------------------- data
    def split(self, split: str):
        if split not in self.splits:
            texts, labels, task_of, index = [], [], [], []
            for k, t in enumerate(self.tasks):
                ex = self.D.examples(t, "eval")
                for i in BK.split_indices(self.bank, t, split):
                    texts.append(ex[i][0])
                    labels.append(ex[i][1])
                    task_of.append(k)
                    index.append(int(i))
            self.splits[split] = (self.LL.encode(texts), np.asarray(labels), np.asarray(task_of), np.asarray(index))
        return self.splits[split]

    # ------------------------------------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, split: str, only_task: int | None = None) -> tuple[dict, dict]:
        """(metrics, predictions). only_task: evaluate one task's examples (expert references)."""
        ids, labels, task_of, index = self.split(split)
        rows = np.arange(len(ids)) if only_task is None else np.flatnonzero(task_of == only_task)
        sub_ids = [ids[i] for i in rows]
        t0 = time.time()
        if self.spec.multi_token:
            pj, pl, top = self.ev.predict(sub_ids, task_of[rows])
            met = self.ev.metrics(pj, pl, top, labels[rows], task_of[rows])
            pred = {"joint": pj.astype(np.int32), "local": pl.astype(np.int32)}
        else:
            lg, top = self.ev.outputs(sub_ids, free=True)
            met = self.ev.metrics(lg, top, labels[rows], task_of[rows])
            pred = {"joint": lg.argmax(1).astype(np.int32),
                    "local": np.asarray([int(np.argmax(lg[r][self.ev.maps[self.tasks[task_of[rows][r]]]]))
                                         for r in range(len(rows))], np.int32),
                    "logits": lg.astype(np.float16)}
        met["seconds"] = time.time() - t0
        pred.update({"labels": labels[rows].astype(np.int32), "task_of": task_of[rows].astype(np.int16),
                     "index": index[rows].astype(np.int32)})
        return met, pred


def normalized(met: dict, ref: list[float]) -> float:
    loc = np.asarray(met["local_per_task"], np.float64)
    ok = np.isfinite(loc)
    return float(np.mean(loc[ok] / np.asarray(ref, np.float64)[ok]))
