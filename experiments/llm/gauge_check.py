"""Function-preserving reparametrisations of Llama attention change per-matrix kappa (numerical check, layer 5).
(1) v/o: per KV head h, W_V[h] -> M_h W_V[h], and W_O columns of every query head in group h -> W_O[:, h] M_h^-1.
    Exact: attention output per head is softmax(.) V, linear in V; o_proj maps the concatenated heads.
(2) q/k with RoPE: per head and frequency pair j, scale rows (2j, 2j+1) of W_Q by c and of W_K by 1/c
    (a scaled identity on each 2-D pair commutes with the RoPE rotation, so q.k is unchanged for every position).
Checks the exact invariance on random inputs (attention logits / head outputs) and reports kappa before/after."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/home/azpitarte/condmerge/src")
from condmerge.llm import banks as BK
torch.manual_seed(0)
bank = "l31_8b"; L = 5
wq = BK.base_weight(bank, f"model.layers.{L}.self_attn.q_proj").double()
wk = BK.base_weight(bank, f"model.layers.{L}.self_attn.k_proj").double()
wv = BK.base_weight(bank, f"model.layers.{L}.self_attn.v_proj").double()
wo = BK.base_weight(bank, f"model.layers.{L}.self_attn.o_proj").double()
H, KV, d = 32, 8, 128
kap = lambda w: (lambda s: float(s[0] / s[-1]))(torch.linalg.svdvals(w))
# (1) v/o gauge
rng = np.random.default_rng(0)
M = [torch.from_numpy(np.eye(d) + 0.5 * rng.standard_normal((d, d))).double() for _ in range(KV)]
wv2 = wv.clone(); wo2 = wo.clone()
for h in range(KV):
    wv2[h * d:(h + 1) * d] = M[h] @ wv[h * d:(h + 1) * d]
    Minv = torch.linalg.inv(M[h])
    for qh in range(h * (H // KV), (h + 1) * (H // KV)):   # query heads sharing KV head h (GQA)
        wo2[:, qh * d:(qh + 1) * d] = wo[:, qh * d:(qh + 1) * d] @ Minv
x = torch.randn(7, 4096, dtype=torch.float64)
A = torch.softmax(torch.randn(H, 7, 7, dtype=torch.float64), -1)  # any attention pattern
def attn_out(wv_, wo_):
    v = (x @ wv_.T).reshape(7, KV, d)
    heads = torch.stack([A[qh] @ v[:, qh // (H // KV)] for qh in range(H)], 1).reshape(7, H * d)
    return heads @ wo_.T
dv = float((attn_out(wv, wo) - attn_out(wv2, wo2)).norm() / attn_out(wv, wo).norm())
print(f"v/o reparam: relative output change {dv:.1e}; kappa(W_V) {kap(wv):.4g} -> {kap(wv2):.4g}; kappa(W_O) {kap(wo):.4g} -> {kap(wo2):.4g}")
# (2) q/k gauge with RoPE: per (head, pair) scale c
c = torch.from_numpy(np.exp(rng.uniform(-1.5, 1.5, size=(H, d // 2)))).double()
wq2 = wq.clone().reshape(H, d, 4096); wk2 = wk.clone().reshape(KV, d, 4096)
# rotate_half convention: pairs are (i, i + d/2); scale both members of each pair by the same c
for qh in range(H):
    cc = torch.cat([c[qh], c[qh]])
    wq2[qh] = wq2[qh] * cc[:, None]
# K must be scaled by 1/c for every query head sharing it -> requires c identical within a GQA group
c_group = c.reshape(KV, H // KV, d // 2)[:, 0]          # use the group's first head's c for all its heads
wq2 = wq.clone().reshape(H, d, 4096)
for qh in range(H):
    cc = torch.cat([c_group[qh // (H // KV)], c_group[qh // (H // KV)]])
    wq2[qh] = wq2[qh] * cc[:, None]
for h in range(KV):
    cc = torch.cat([c_group[h], c_group[h]])
    wk2[h] = wk2[h] / cc[:, None]
wq2 = wq2.reshape(H * d, 4096); wk2 = wk2.reshape(KV * d, 4096)
def rope(t, pos):  # t (n, heads, d), rotate_half convention with Llama-3 style frequencies (any base works)
    inv = 1.0 / (500000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = pos[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    rot = torch.cat([-t[..., d // 2:], t[..., :d // 2]], -1)
    return t * cos[:, None] + rot * sin[:, None]
pos = torch.arange(7, dtype=torch.float64)
def logits(wq_, wk_):
    q = rope((x @ wq_.T).reshape(7, H, d), pos); k = rope((x @ wk_.T).reshape(7, KV, d), pos)
    return torch.stack([q[:, qh] @ k[:, qh // (H // KV)].T for qh in range(H)])
dl = float((logits(wq, wk) - logits(wq2, wk2)).norm() / logits(wq, wk).norm())
print(f"q/k reparam (RoPE-commuting): relative attention-logit change {dl:.1e}; kappa(W_Q) {kap(wq):.4g} -> {kap(wq2):.4g}; kappa(W_K) {kap(wk):.4g} -> {kap(wk2):.4g}")
import json, pathlib
out = pathlib.Path("/data/home/azpitarte/condmerge/results/llm/spectral/gauge_check.json")
out.write_text(json.dumps({"bank": bank, "layer": L,
    "vo": {"rel_output_change": dv, "kappa_WV": [kap(wv), kap(wv2)], "kappa_WO": [kap(wo), kap(wo2)]},
    "qk": {"rel_logit_change": dl, "kappa_WQ": [kap(wq), kap(wq2)], "kappa_WK": [kap(wk), kap(wk2)]}}, indent=1))
