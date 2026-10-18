"""GPU-resident zero-shot evaluation of a vision state dict.

The pixel caches (uint8, already resized/cropped) are moved to the GPU once;
each evaluation then copies a candidate state dict into a single resident
``CLIPVisionModel`` and runs a forward pass per task. Normalisation happens on
the GPU. Forward passes use fp16 autocast by default (checked against fp32 in
``scripts/sanity_eval.py``).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import CLIP_MEAN, CLIP_STD, load_cache
from .models import build_heads, new_vision_model, visual_projection_weight


class Evaluator:
    def __init__(self, tasks: list[str], splits: tuple[str, ...] = ("dev",), device: str = "cuda",
                 batch: int = 1024, amp: bool = True):
        self.tasks = list(tasks)
        self.device = device
        self.batch = batch
        self.amp = amp
        self.model = new_vision_model(device)
        self.proj = visual_projection_weight().to(device)
        self.heads = {t: h.to(device) for t, h in build_heads(self.tasks, device).items()}
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.data: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for split in splits:
            for t in self.tasks:
                self._load(t, split)

    def _load(self, task: str, split: str) -> None:
        if (task, split) in self.data:
            return
        x, y = load_cache(task, split, mmap=True)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        yt = torch.from_numpy(y).to(self.device)
        self.data[(task, split)] = (xt, yt)

    def n(self, task: str, split: str) -> int:
        self._load(task, split)
        return int(self.data[(task, split)][1].numel())

    @torch.no_grad()
    def load_state(self, sd: dict[str, torch.Tensor]) -> None:
        """Copy a FusionBench-keyed state dict (``vision_model.*``) into the resident model."""
        params = dict(self.model.named_parameters())
        # transformers 5 drops the ``vision_model.`` prefix from CLIPVisionModel parameter names.
        def src(k: str) -> str:
            return k if k in sd else f"vision_model.{k}"
        missing = [k for k in params if src(k) not in sd]
        if missing:
            raise KeyError(f"state dict missing {missing[:5]}")
        if len(sd) != len(params):
            raise KeyError(f"state dict has {len(sd)} tensors, model has {len(params)}")
        for k, p in params.items():
            p.copy_(sd[src(k)].to(p.device, p.dtype))

    def _features(self, xb: torch.Tensor) -> torch.Tensor:
        px = (xb.float() / 255.0 - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            pooled = self.model(pixel_values=px).pooler_output
        f = pooled.float() @ self.proj.T
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def evaluate(self, tasks: list[str] | None = None, split: str = "dev", return_preds: bool = False,
                 return_loss: bool = False, logit_scale: float = 100.0) -> dict:
        """Top-1 accuracy per task (and optional per-example predictions / CE at CLIP's logit scale)."""
        tasks = self.tasks if tasks is None else tasks
        out: dict = {"acc": {}, "n": {}}
        if return_preds:
            out["pred"] = {}
        if return_loss:
            out["ce"] = {}
        t0 = time.time()
        for t in tasks:
            self._load(t, split)
            x, y = self.data[(t, split)]
            head = self.heads[t]
            preds, ce = [], 0.0
            for i in range(0, x.shape[0], self.batch):
                logits = self._features(x[i:i + self.batch]) @ head.T
                preds.append(logits.argmax(-1))
                if return_loss:
                    ce += float(F.cross_entropy(logit_scale * logits, y[i:i + self.batch], reduction="sum"))
            p = torch.cat(preds)
            out["acc"][t] = float((p == y).float().mean())
            out["n"][t] = int(y.numel())
            if return_preds:
                out["pred"][t] = p.to(torch.int16).cpu().numpy()
            if return_loss:
                out["ce"][t] = ce / y.numel()
        out["avg_acc"] = float(np.mean([out["acc"][t] for t in tasks]))
        out["seconds"] = time.time() - t0
        return out


def normalized(acc: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    return {t: acc[t] / ref[t] for t in acc}
