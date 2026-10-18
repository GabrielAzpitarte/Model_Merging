"""LoRA experts (KnOTS rank-16 CLIP ViT-B/32 bank): exact products, QR-core spectra, gauge, merging.

Each expert updates the attention q/k/v/out projections of all 12 blocks with
``Delta = gamma * B @ A`` (B: 768x16, A: 16x768, gamma = lora_alpha / r = 1).
The adapters were trained on the vision tower with frozen zero-shot heads, so
they are evaluated exactly like the full-FT experts.

Spectral conventions (review 3.7):
* ordinary kappa2(Delta) is infinite for r < 768 (not a usable expert score);
* the nonzero spectrum of Delta is computed exactly from the r x r core
  ``R_B R_A^T`` of reduced QR factorisations ``B = Q_B R_B``, ``A^T = Q_A R_A``;
* ``kappa_plus(Delta; tau)`` keeps singular values above ``tau * sigma_1``;
* factor condition numbers kappa(A), kappa(B) are gauge dependent:
  ``(B R)(R^-1 A) = B A`` for any invertible R.
Full-weight statistics use ``W_i = W0 + Delta_i`` (finite kappa2).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

KNOTS_ROOT = Path("/data/home/azpitarte/lora_merging_ica/checkpoints/vitb32")


def load_knots_lora(task: str, root: Path = KNOTS_ROOT) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """{unprefixed weight name: (B, A, gamma)} for one expert."""
    cfg = json.loads((root / task / "adapter_config.json").read_text())
    if cfg.get("use_rslora"):
        gamma = float(cfg["lora_alpha"]) / float(cfg["r"]) ** 0.5
    else:
        gamma = float(cfg["lora_alpha"]) / float(cfg["r"])
    sd = torch.load(root / task / "adapter_model.bin", map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if not k.endswith("lora_A.weight"):
            continue
        stem = k.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        b = sd[k.replace("lora_A", "lora_B")]
        out[stem + ".weight"] = (b.float(), v.float(), gamma)
    return out


def delta(b: torch.Tensor, a: torch.Tensor, gamma: float) -> torch.Tensor:
    return gamma * (b @ a)


def core_svals(b: torch.Tensor, a: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact nonzero singular values of gamma*B@A from the r x r QR core (float64)."""
    b, a = b.double(), a.double()
    _, rb = torch.linalg.qr(b)
    _, ra = torch.linalg.qr(a.T)
    return abs(gamma) * torch.linalg.svdvals(rb @ ra.T)


def gauge(b: torch.Tensor, a: torch.Tensor, r_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return b @ r_mat, torch.linalg.solve(r_mat, a)


def kappa(m: torch.Tensor) -> float:
    s = torch.linalg.svdvals(m.double())
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


class LoRABank:
    """Base + dense per-expert updates for the targeted matrices only (others are exactly the base)."""

    def __init__(self, tasks: list[str], device: str, base: dict[str, torch.Tensor] | None = None):
        from .aa_common import load_base_unprefixed

        base = base or load_base_unprefixed()
        self.tasks = list(tasks)
        self.names = list(base)
        self.base = {k: v.to(device) for k, v in base.items()}
        self.factors = {t: load_knots_lora(t) for t in tasks}
        targets = sorted(set().union(*[set(f) for f in self.factors.values()]))
        for t in tasks:
            assert set(self.factors[t]) == set(targets), f"{t} targets differ"
        self.linear = targets  # the matrices that carry an update
        self.deltas = {n: torch.stack([delta(*self.factors[t][n]).to(device) for t in tasks]) for n in targets}

    @property
    def k(self) -> int:
        return len(self.tasks)

    def merged(self, lam_lin: dict[str, torch.Tensor], alpha, default: torch.Tensor | None = None,
               compress_rank: int | None = None) -> dict:
        """W0 + alpha * sum_i lam_i Delta_i on targeted matrices (exact, rank <= K*r), optionally truncated."""
        dev = next(iter(self.base.values())).device
        default = torch.full((self.k,), 1.0 / self.k, device=dev) if default is None else default
        out = dict(self.base)
        for n in self.linear:
            lam = lam_lin.get(n, default)
            upd = alpha * torch.tensordot(lam, self.deltas[n], dims=1)
            if compress_rank is not None:
                u, s, vh = torch.linalg.svd(upd, full_matrices=False)
                upd = (u[:, :compress_rank] * s[:compress_rank]) @ vh[:compress_rank]
            out[n] = self.base[n] + upd
        return out
