"""CLIP ViT-B/32 state dicts, zero-shot heads, and a fast cached evaluator.

Forward pass (task-vectors / FusionBench protocol):
``visual_projection(vision_model(pixels).pooler_output)`` -> L2-normalise ->
cosine against a frozen zero-shot text head built from the *base* CLIP text
tower. Only the vision tower is merged; the projection and heads are shared.

Heads follow the ICA project's validated recipe (base zero-shot reproduced the
published per-task numbers to 0.1 pt on 7/8 tasks): per class, embed every
template, L2-normalise, average, renormalise; class order is the HF dataset's.
``logit_scale`` is omitted (cannot change an argmax).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .data import CLIP_MEAN, CLIP_STD, load_cache
from .paths import BASE_MODEL, CACHE, TASKS8, fusionbench_dir

_BASE_CACHE: dict = {}


def load_base_state() -> dict[str, torch.Tensor]:
    """Base ``CLIPVisionModel`` state dict (keys ``vision_model.*``), float32, CPU."""
    if "sd" not in _BASE_CACHE:
        from transformers import CLIPVisionModel

        m = CLIPVisionModel.from_pretrained(BASE_MODEL)
        _BASE_CACHE["sd"] = {k: v.detach().clone().float() for k, v in m.state_dict().items()}
    return _BASE_CACHE["sd"]


def load_expert_state(task: str) -> dict[str, torch.Tensor]:
    """FusionBench expert, keys normalised to the base layout (transformers 5 drops ``vision_model.``)."""
    sd = load_file(str(fusionbench_dir(task) / "model.safetensors"))
    sd = {k.removeprefix("vision_model."): v for k, v in sd.items()}
    base = load_base_state()
    if set(sd) != set(base):
        missing, extra = set(base) - set(sd), set(sd) - set(base)
        raise KeyError(f"{task}: key mismatch, missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    return {k: v.float() for k, v in sd.items()}


def load_experts(tasks=TASKS8) -> dict[str, dict[str, torch.Tensor]]:
    return {t: load_expert_state(t) for t in tasks}


def build_heads(device: str = "cuda") -> dict:
    """Zero-shot heads + visual projection; cached to ``artifacts/cache/heads_vitb32.pt``."""
    path = CACHE / "heads_vitb32.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    from transformers import CLIPModel, CLIPTokenizer

    from .data import classnames
    from .templates import get_templates

    model = CLIPModel.from_pretrained(BASE_MODEL).eval().to(device)
    tok = CLIPTokenizer.from_pretrained(BASE_MODEL)
    heads = {}
    with torch.no_grad():
        for task in TASKS8:
            rows = []
            for name in classnames(task):
                enc = tok([t(name) for t in get_templates(task)], padding=True, return_tensors="pt").to(device)
                emb = model.text_projection(model.text_model(**enc)[1])
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.mean(dim=0)
                rows.append(emb / emb.norm())
            heads[task] = torch.stack(rows).cpu()
    out = {"heads": heads, "visual_projection": model.visual_projection.weight.detach().cpu().clone()}
    torch.save(out, path)
    return out


class Evaluator:
    """Holds uint8 pixels on the GPU and evaluates arbitrary vision state dicts.

    ``evaluate(sd, split)`` returns ``{task: accuracy}``. Precision is fp16
    autocast throughout (the same for every method; the fp32 gap is reported
    once in the expert table).
    """

    def __init__(self, device: str = "cuda", splits=("dev",), tasks=TASKS8, batch: int = 1000,
                 amp: bool = True, max_per_task: int | None = None):
        from transformers import CLIPVisionConfig, CLIPVisionModel

        self.device = device
        self.tasks = list(tasks)
        self.batch = batch
        self.amp = amp
        hp = build_heads(device)
        self.heads = {t: hp["heads"][t].to(device) for t in self.tasks}
        self.proj = hp["visual_projection"].to(device)
        cfg = CLIPVisionConfig.from_pretrained(BASE_MODEL)
        self.model = CLIPVisionModel(cfg).eval().to(device)
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.data: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for split in splits:
            for t in self.tasks:
                x, y = load_cache(t, split, mmap=True)
                if max_per_task is not None:
                    x, y = x[:max_per_task], y[:max_per_task]
                self.data[(t, split)] = (torch.from_numpy(np.ascontiguousarray(x)).to(device),
                                         torch.from_numpy(y).to(device))

    def load(self, sd: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in sd.items()}, strict=True)

    @torch.no_grad()
    def features(self, x_u8: torch.Tensor) -> torch.Tensor:
        x = (x_u8.float() / 255.0 - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            pooled = self.model(pixel_values=x).pooler_output
        f = pooled.float() @ self.proj.T
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def correct(self, task: str, split: str) -> np.ndarray:
        """Per-example correctness (bool), for paired bootstrap / McNemar comparisons."""
        x, y = self.data[(task, split)]
        out = []
        for i in range(0, x.shape[0], self.batch):
            f = self.features(x[i:i + self.batch])
            out.append((f @ self.heads[task].T).argmax(-1).eq(y[i:i + self.batch]))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def accuracy(self, task: str, split: str) -> float:
        return float(self.correct(task, split).mean())

    def evaluate(self, sd: dict[str, torch.Tensor] | None, split: str = "dev", tasks=None) -> dict[str, float]:
        if sd is not None:
            self.load(sd)
        return {t: self.accuracy(t, split) for t in (tasks or self.tasks)}


def summarize(acc: dict[str, float], expert_acc: dict[str, float]) -> dict[str, float]:
    """Mean absolute accuracy, mean normalised accuracy (acc / own-expert acc), and the minimum."""
    norm = {t: acc[t] / expert_acc[t] for t in acc}
    return {"mean_acc": float(np.mean(list(acc.values()))),
            "mean_norm": float(np.mean(list(norm.values()))),
            "min_norm": float(np.min(list(norm.values())))}


def load_expert_table(path: Path | None = None) -> dict:
    from .paths import RESULTS

    path = path or RESULTS / "experts" / "expert_accuracy.json"
    return json.loads(path.read_text())


def timer() -> float:
    torch.cuda.synchronize()
    return time.time()
