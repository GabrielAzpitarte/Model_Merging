"""In-memory collection of per-linear-map input second moments E[x x^T] for one model.

Same convention as scripts/compute_input_grams.py: uncentred, averaged over all tokens
(patches: per-image patch count; q/k/v share ``attn_in``), accumulated in float64.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import CLIP_MEAN, CLIP_STD


def input_name(param: str) -> str:
    if param == "embeddings.patch_embedding.weight":
        return "patches"
    layer = param.split(".")[2]
    if any(k in param for k in ("q_proj", "k_proj", "v_proj")):
        return f"L{layer}.attn_in"
    if "out_proj" in param:
        return f"L{layer}.out_in"
    if "fc1" in param:
        return f"L{layer}.fc1_in"
    if "fc2" in param:
        return f"L{layer}.fc2_in"
    raise KeyError(param)


@torch.no_grad()
def collect_grams(model, pixels_u8: list[np.ndarray], names: list[str], device: str, batch: int = 256) -> dict:
    """``model``: a CLIPVisionModel already holding the weights; ``pixels_u8``: list of uint8 arrays (N,3,224,224)."""
    grams: dict[str, torch.Tensor] = {}
    mods = dict(model.named_modules())
    hooks, seen = [], set()

    def make_hook(key: str):
        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            grams[key] = grams.get(key, 0) + x.T @ x
        return hook

    for p in names:
        key = input_name(p)
        if key in seen or key == "patches":
            continue
        seen.add(key)
        mod_name = p[: -len(".weight")]
        mod_name = mod_name if mod_name in mods else "vision_model." + mod_name
        hooks.append(mods[mod_name].register_forward_pre_hook(make_hook(key)))
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    ps = model.config.patch_size
    n_img = 0
    try:
        for arr in pixels_u8:
            for i in range(0, arr.shape[0], batch):
                px = (torch.from_numpy(np.ascontiguousarray(arr[i:i + batch])).to(device).float() / 255 - mean) / std
                pt = px.unfold(2, ps, ps).unfold(3, ps, ps).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * ps * ps).double()
                grams["patches"] = grams.get("patches", 0) + pt.T @ pt
                model(pixel_values=px)
                n_img += px.shape[0]
    finally:
        for h in hooks:
            h.remove()
    n_patches = (224 // ps) ** 2
    out = {}
    for k, g in grams.items():
        n = n_img * n_patches if k == "patches" else n_img * (n_patches + 1)
        out[k] = (g / n).float()
    return {"grams": out, "n_images": n_img, "inputs_of": {p: input_name(p) for p in names}}
