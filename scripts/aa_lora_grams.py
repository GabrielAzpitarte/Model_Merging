"""Input Gram matrices of the LoRA experts' attention maps, for RegMean-type merging.

Convention (matches master's compute_input_grams.py): uncentred second moment E[x x^T], mean over all 50 tokens of
every image, accumulated in fp64 and stored as fp32. q/k/v_proj share the input "L{l}.attn_in" (the LN1 output);
out_proj has its own input "L{l}.out_in" (the attention output).
Budgets:
  C   task-separated: expert i (W0 + B_i A_i) on task i's calib split (512 train images, unlabeled)
  Bp  pooled, label-free and task-ID-free: the union of the 8 calib splits (4096 images) through EACH expert
  B0  pooled through the BASE model (one Gram shared by all experts; RegMean must then reduce to averaging)
Writes artifacts/cache/aa/grams_lora_<bank>_<budget>.pt:
  {"grams": {input_name: (K, d, d) float32}, "experts": [...], "budget": ..., "n_tokens": {expert: int}}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import CACHE, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402
from condmerge.data import load_cache  # noqa: E402


def collect(model, data: TaskData, images: torch.Tensor, batch: int = 256) -> tuple[dict[str, torch.Tensor], int]:
    """Mean uncentred Grams of every attention input over all tokens of `images` (uint8, on GPU)."""
    acc: dict[str, torch.Tensor] = {}
    count = {"n": 0}
    hooks = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn.q_proj") or name.endswith("self_attn.out_proj"):
            layer = int(name.split("layers.")[1].split(".")[0])
            key = f"L{layer}.attn_in" if name.endswith("q_proj") else f"L{layer}.out_in"

            def hook(m, inp, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                acc[key] = acc.get(key, 0) + x.T @ x
            hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            px = data.pixels(images[i:i + batch])
            model(pixel_values=px)
            count["n"] += px.shape[0] * 50
    for h in hooks:
        h.remove()
    return {k: (v / count["n"]).float() for k, v in acc.items()}, count["n"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="knots_release")
    ap.add_argument("--budgets", default="C,Bp,B0")
    ap.add_argument("--n-per-task", type=int, default=512, help="calibration images per task (subsample: first N)")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    bank = load_bank(args.bank)
    data = TaskData(TASKS8[:1], ("dev",), dev)   # only for pixels() normalisation constants
    model = new_model(dev)
    params = dict(model.named_parameters())
    calib = {t: torch.from_numpy(np.array(load_cache(t, "calib")[0][: args.n_per_task])).to(dev) for t in TASKS8}
    pooled = torch.cat([calib[t] for t in TASKS8])

    def load_expert(t: str | None):
        with torch.no_grad():
            for n, p in params.items():
                w = base[n].to(dev)
                if t is not None and n in bank[t]:
                    w = w + bank[t][n].to(dev)
                p.copy_(w)

    out_dir = CACHE / "aa"
    for budget in args.budgets.split(","):
        grams: dict[str, list] = {}
        ntok = {}
        experts = ["base"] if budget == "B0" else TASKS8
        for t in experts:
            load_expert(None if t == "base" else t)
            imgs = calib[t] if budget == "C" else pooled
            g, n = collect(model, data, imgs)
            ntok[t] = n
            for k, v in g.items():
                grams.setdefault(k, []).append(v.cpu())
            print(f"{args.bank} {budget} {t}: {n} tokens ({time.time() - t0:.0f}s)", flush=True)
        blob = {"grams": {k: torch.stack(v) for k, v in sorted(grams.items())}, "experts": experts, "budget": budget,
                "bank": args.bank, "n_tokens": ntok,
                "convention": "uncentred E[xx^T], mean over all 50 tokens/image; q/k/v share attn_in"}
        suffix = "" if args.n_per_task == 512 else f"_n{args.n_per_task}"
        blob["n_per_task"] = args.n_per_task
        torch.save(blob, out_dir / f"grams_lora_{args.bank}_{budget}{suffix}.pt")
        print(f"saved {budget}: {len(blob['grams'])} inputs", flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
