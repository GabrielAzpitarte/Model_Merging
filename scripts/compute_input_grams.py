"""Uncentred input second moments C = E[x x^T] for every linear map, on the pooled calibration set.

Information budget (B): the pool is the union of the 8 tasks' ``calib`` splits (512 train
images each) WITHOUT task identities and without labels. Activations come from the model
named by ``--model`` (default: the base W0). q/k/v share one input (``attn_in``).
Saved as float32 to artifacts/cache/grams/<model>.pt:
``{"grams": {input_name: (d, d)}, "n_tokens": int, "inputs_of": {param: input_name}}``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE_ARCH, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import load_base_state, load_expert_state  # noqa: E402
from condmerge.data import CLIP_MEAN, CLIP_STD, load_cache  # noqa: E402


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
def main() -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    from condmerge.paths import BASE_MODEL
    from condmerge.spectra import linear_map_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base", help="'base' or an expert task name")
    ap.add_argument("--data", default="pool", help="'pool' (8 calib splits, no task ids; budget B) or "
                    "'task:<name>' (one task's calib split; budget C)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    sd = load_base_state() if args.model == "base" else load_expert_state(args.model)
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    names = linear_map_names(sd)
    inputs_of = {p: input_name(p) for p in names}
    grams: dict[str, torch.Tensor] = {}
    count = {"tokens": 0}
    hooks = []
    mods = dict(model.named_modules())

    def make_hook(key: str):
        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            grams[key] = grams.get(key, 0) + x.T @ x
        return hook

    seen = set()
    for p in names:
        key = inputs_of[p]
        if key in seen or key == "patches":
            continue
        seen.add(key)
        mod_name = p[: -len(".weight")]
        mod_name = mod_name if mod_name in mods else "vision_model." + mod_name
        hooks.append(mods[mod_name].register_forward_pre_hook(make_hook(key)))
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
    n_img = 0
    data_tasks = TASKS8 if args.data == "pool" else [args.data.split(":", 1)[1]]
    for task in data_tasks:
        x, _ = load_cache(task, "calib", mmap=True)  # labels deliberately unused
        for i in range(0, x.shape[0], args.batch):
            px = (torch.from_numpy(np.ascontiguousarray(x[i:i + args.batch])).to(dev).float() / 255 - mean) / std
            # patch-embedding input = non-overlapping 32x32 patches flattened (3*32*32), conv weight order
            ps = model.config.patch_size
            pt = px.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, 3, g, g, ps, ps)
            pt = pt.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * ps * ps).double()
            grams["patches"] = grams.get("patches", 0) + pt.T @ pt
            model(pixel_values=px)
            n_img += px.shape[0]
    for h in hooks:
        h.remove()
    n_patches = (224 // model.config.patch_size) ** 2
    n_tokens = n_img * (n_patches + 1)
    out = {"grams": {}, "n_images": n_img, "n_tokens_per_input": {}, "inputs_of": inputs_of, "model": args.model,
           "data": args.data,
           "budget": "B: pooled unlabelled calib, no task ids" if args.data == "pool" else "C: one task's unlabelled calib"}
    for k, g in grams.items():
        n = n_img * n_patches if k == "patches" else n_tokens
        out["grams"][k] = (g / n).float().cpu()
        out["n_tokens_per_input"][k] = n
    d = CACHE_ARCH / "grams"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}.pt" if (args.model == "base" and args.data == "pool") else f"{args.model}__{args.data.replace(':', '-')}.pt"
    torch.save(out, d / name)
    print(f"saved {len(out['grams'])} grams for {args.model} from {n_img} images in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
