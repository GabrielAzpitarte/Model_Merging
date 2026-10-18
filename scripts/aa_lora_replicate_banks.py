"""Replicate the LoRA 'argmin-kappa beats random per-map selection' observation on independent 8-task LoRA banks.

Banks (Alexandre's, read-only, /shared/ssd_30T/molohio_vit20; rank 16, alpha 16 -> gamma 1):
  lora_c5a (KnOTS recipe, seed 0), lora_c5a_seed1 (same recipe, seed 1), lora (vit20 bank, 2000 steps, seed 0);
plus the KnOTS release for reference. Per bank (dev, budget A rules):
  uniform at alpha {0.6, 0.8, 1.0}; kappa argmin and kappa softmax b=128 at alpha {0.8, 1.0};
  20 free random per-map selections at alpha {0.8, 1.0} (the shuffled-kappa null at b=inf).
kappa = sigma_1/sigma_q of W0 + B A per adapted map, float64 on CPU.
Writes results/aa/lora/replicate_banks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402

MOLOHIO = Path("/shared/ssd_30T/molohio_vit20")


def load_bank(name: str) -> dict[str, dict[str, torch.Tensor]]:
    """{task: {weight name: dense Delta (float32)}}"""
    out = {}
    if name == "knots_release":
        from condmerge.lora import delta, load_knots_lora
        for t in TASKS8:
            out[t] = {n: delta(*f) for n, f in load_knots_lora(t).items()}
        return out
    for t in TASKS8:
        z = np.load(MOLOHIO / name / f"{t}.npz")
        stems = sorted({k[2:] for k in z.files if k.startswith("A_")})
        out[t] = {s + ".weight": torch.from_numpy(z[f"B_{s}"] @ z[f"A_{s}"]).float() for s in stems}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="lora_c5a,lora_c5a_seed1,lora")
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    f = (RESULTS / "aa" / "lora" / "replicate_banks.jsonl").open("a")
    eye = np.eye(8)
    for bname in args.banks.split(","):
        bank = load_bank(bname)
        names = sorted(bank[TASKS8[0]])
        deltas = {n: torch.stack([bank[t][n] for t in TASKS8]).to(dev) for n in names}
        K = np.zeros((len(names), 8))
        for pi, n in enumerate(names):
            w0 = base[n].double().numpy()
            for i, t in enumerate(TASKS8):
                s = np.linalg.svd(w0 + bank[t][n].double().numpy(), compute_uv=False)
                K[pi, i] = s[0] / s[-1]
        sc = float(np.median([np.std(K[p]) for p in range(len(names)) if np.std(K[p]) > 0]))

        def ev(label, lam_rows, a, seed=None):
            params = dict(base_dev)
            for pi, n in enumerate(names):
                lam = torch.as_tensor(lam_rows[pi], dtype=torch.float32, device=dev)
                params[n] = base_dev[n] + a * torch.tensordot(lam, deltas[n], dims=1)
            r = data.evaluate(model, "dev", params=params)
            f.write(json.dumps({"bank": bname, "label": label, "alpha": a, "seed": seed, "avg_acc": r["avg_acc"], "acc": r["acc"]}) + "\n")
            f.flush()
            return r["avg_acc"]

        uni = np.full((len(names), 8), 1 / 8)
        for a in (0.6, 0.8, 1.0):
            print(f"{bname} uniform a={a}: {ev('uniform', uni, a):.4f}", flush=True)
        amin = eye[K.argmin(1)]
        z = -128 / sc * (K - K.min(1, keepdims=True))
        soft = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        for a in (0.8, 1.0):
            ka = ev("kappa_argmin", amin, a)
            ks = ev("kappa_b128", soft, a)
            nulls = [ev("free_random_selection", eye[np.random.default_rng(2000 + s).integers(0, 8, len(names))], a, s)
                     for s in range(args.n_random)]
            nulls = np.array(nulls)
            print(f"{bname} a={a}: kappa_argmin {ka:.4f} kappa_b128 {ks:.4f} | random mean {nulls.mean():.4f} sd {nulls.std():.4f} "
                  f"max {nulls.max():.4f} | argmin percentile {np.mean(nulls < ka) * 100:.0f} ({time.time() - t0:.0f}s)", flush=True)
        del deltas
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
