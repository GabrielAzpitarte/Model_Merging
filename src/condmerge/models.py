"""Base CLIP, FusionBench experts, and zero-shot heads.

FusionBench full fine-tuned experts (``tanganke/clip-vit-base-patch32_<task>``)
are ``CLIPVisionModel`` state dicts: the whole vision tower was fine-tuned with
the text-derived classification head frozen (task-vectors protocol). The
visual projection is *not* in the checkpoints; evaluation uses the pretrained
projection and pretrained text tower, exactly as FusionBench does:
``normalize(visual_projection(vision_model(x).pooler_output)) @ head.T``.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from safetensors.torch import load_file

from .paths import BASE_MODEL, CACHE, fusionbench_dir
from .templates import get_templates


def load_base_clip(device: str = "cpu"):
    from transformers import CLIPModel

    return CLIPModel.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(device).eval()


def base_vision_state(clip) -> "OrderedDict[str, torch.Tensor]":
    """Base vision tower with the same key names as the FusionBench checkpoints."""
    sd = clip.vision_model.state_dict()
    return OrderedDict((f"vision_model.{k}", v.detach().clone()) for k, v in sd.items())


def load_expert_state(task: str) -> "OrderedDict[str, torch.Tensor]":
    sd = load_file(str(fusionbench_dir(task) / "model.safetensors"))
    return OrderedDict(sorted(sd.items()))


def new_vision_model(device: str):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    cfg = CLIPVisionConfig.from_pretrained(BASE_MODEL)
    return CLIPVisionModel(cfg).to(device).eval()


def build_heads(tasks: list[str], device: str = "cuda", force: bool = False) -> dict[str, torch.Tensor]:
    """Zero-shot heads (n_classes, 512): mean of L2-normalised template embeddings, renormalised.

    Class order is each HF dataset's own label order. Cached on disk.
    """
    from transformers import CLIPTokenizer

    from .data import classnames

    path = CACHE / "heads_clip-vit-base-patch32.pt"
    heads = torch.load(path, weights_only=True) if path.exists() and not force else {}
    missing = [t for t in tasks if t not in heads]
    if missing:
        clip = load_base_clip(device)
        tok = CLIPTokenizer.from_pretrained(BASE_MODEL)
        with torch.no_grad():
            for task in missing:
                rows = []
                for name in classnames(task):
                    toks = tok([t(name) for t in get_templates(task)], padding=True, return_tensors="pt").to(device)
                    emb = clip.text_projection(clip.text_model(**toks)[1])
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb = emb.mean(dim=0)
                    rows.append(emb / emb.norm())
                heads[task] = torch.stack(rows).cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(heads, path)
    return {t: heads[t] for t in tasks}


def visual_projection_weight() -> torch.Tensor:
    path = CACHE / "visual_projection_clip-vit-base-patch32.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    clip = load_base_clip("cpu")
    w = clip.visual_projection.weight.detach().clone()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, path)
    return w
