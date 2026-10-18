"""Shared helpers for the aa/side track (new file; master-owned modules are not edited).

Conventions match master (peer session): unprefixed transformers-5 parameter
names; 73 linear maps = 72 attention/MLP weights + patch embedding reshaped to
(768, 3072); position embedding, LayerNorm and class embedding get the prior;
a linear layer's bias is tied to its weight's lambda.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .data import CLIP_MEAN, CLIP_STD, load_cache
from .paths import BASE_MODEL, fusionbench_dir


def is_linear_map(name: str, t: torch.Tensor) -> bool:
    if name == "embeddings.patch_embedding.weight":
        return True
    return t.ndim == 2 and name.endswith(".weight") and "position_embedding" not in name


def bias_owner(name: str) -> str | None:
    if name.endswith(".bias") and "layer_norm" not in name and "layernorm" not in name:
        return name[: -len(".bias")] + ".weight"
    return None


def layer_index(name: str) -> int:
    if name.startswith("encoder.layers."):
        return int(name.split(".")[2])
    if name.startswith("embeddings") or name.startswith("pre_layrnorm"):
        return -1
    return 12


def module_type(name: str) -> str:
    for key in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding", "position_embedding",
                "class_embedding", "layer_norm", "layrnorm", "layernorm"):
        if key in name:
            return key
    return "other"


def load_base_unprefixed() -> dict[str, torch.Tensor]:
    from transformers import CLIPVisionModel

    m = CLIPVisionModel.from_pretrained(BASE_MODEL)
    return {k: v.detach().clone().float() for k, v in m.state_dict().items()}


def load_expert_unprefixed(task: str) -> dict[str, torch.Tensor]:
    sd = load_file(str(fusionbench_dir(task) / "model.safetensors"))
    return {k.removeprefix("vision_model."): v.float() for k, v in sd.items()}


def new_model(device: str):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    return CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).to(device).eval()


class Bank:
    """Base weights + stacked task updates (K, *shape) on one device, unprefixed names."""

    def __init__(self, tasks: list[str], device: str, names: list[str] | None = None):
        base = load_base_unprefixed()
        self.names = names or list(base)
        self.tasks = list(tasks)
        self.base = {k: base[k].to(device) for k in self.names}
        experts = {t: load_expert_unprefixed(t) for t in tasks}
        self.deltas = {k: torch.stack([experts[t][k].to(device) - self.base[k] for t in tasks]) for k in self.names}
        self.linear = [k for k in self.names if is_linear_map(k, self.base[k])]

    @property
    def k(self) -> int:
        return len(self.tasks)

    def lam_for(self, name: str, lam_lin: dict[str, torch.Tensor], default: torch.Tensor) -> torch.Tensor:
        if name in lam_lin:
            return lam_lin[name]
        owner = bias_owner(name)
        if owner is not None and owner in lam_lin:
            return lam_lin[owner]
        return default

    def merged(self, lam_lin: dict[str, torch.Tensor], alpha, default: torch.Tensor | None = None) -> dict:
        """W0 + alpha * sum_i lam_i Delta_i (differentiable in lam_lin / alpha if they require grad)."""
        dev = next(iter(self.base.values())).device
        default = torch.full((self.k,), 1.0 / self.k, device=dev) if default is None else default
        out = {}
        for n in self.names:
            lam = self.lam_for(n, lam_lin, default)
            out[n] = self.base[n] + alpha * torch.tensordot(lam, self.deltas[n], dims=1)
        return out


class TaskData:
    """GPU-resident uint8 pixels + labels for several tasks/splits, zero-shot heads, projection."""

    def __init__(self, tasks: list[str], splits: tuple[str, ...], device: str):
        from .models import build_heads, visual_projection_weight

        self.device = device
        self.tasks = list(tasks)
        self.heads = {t: h.to(device) for t, h in build_heads(self.tasks, device).items()}
        self.proj = visual_projection_weight().to(device)
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.data = {}
        for s in splits:
            for t in tasks:
                x, y = load_cache(t, s, mmap=True)
                self.data[(t, s)] = (torch.from_numpy(np.array(x)).to(device), torch.from_numpy(np.array(y)).to(device))

    def pixels(self, xb: torch.Tensor) -> torch.Tensor:
        return (xb.float() / 255.0 - self.mean) / self.std

    def features(self, model, xb: torch.Tensor, params: dict | None = None, amp: str | None = None) -> torch.Tensor:
        px = self.pixels(xb)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
            if params is None:
                pooled = model(pixel_values=px).pooler_output
            else:
                from torch.func import functional_call

                pooled = functional_call(model, params, (px,)).pooler_output
        f = pooled.float() @ self.proj.T
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def evaluate(self, model, split: str, params: dict | None = None, batch: int = 1024, amp: str | None = "fp16",
                 tasks: list[str] | None = None) -> dict:
        acc, ce = {}, {}
        for t in tasks or self.tasks:
            x, y = self.data[(t, split)]
            correct, loss = 0, 0.0
            for i in range(0, x.shape[0], batch):
                logits = self.features(model, x[i:i + batch], params, amp) @ self.heads[t].T
                correct += int((logits.argmax(-1) == y[i:i + batch]).sum())
                loss += float(F.cross_entropy(100.0 * logits, y[i:i + batch], reduction="sum"))
            acc[t] = correct / y.numel()
            ce[t] = loss / y.numel()
        return {"acc": acc, "ce": ce, "avg_acc": float(np.mean(list(acc.values()))),
                "avg_ce": float(np.mean(list(ce.values())))}
