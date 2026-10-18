"""Sanity check: pretrained zero-shot and each expert on every task (8x8), dev split.

Also checks fp16 autocast against fp32 for the base model and one expert, and
records FusionBench's published individual-expert accuracies for comparison
(FusionBench ViT-B/32 table; measured on the *full* official test split, so a
few tenths of a point of difference from our 20 % dev subset is expected).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import torch  # noqa: E402

from condmerge.evaluate import Evaluator  # noqa: E402
from condmerge.models import base_vision_state, load_base_clip, load_expert_state  # noqa: E402

FUSIONBENCH_PUBLISHED = {  # individual fine-tuned experts, ViT-B/32 (FusionBench docs / task-vectors line)
    "sun397": 75.3, "cars": 77.7, "resisc45": 96.1, "eurosat": 99.7,
    "svhn": 97.5, "gtsrb": 98.7, "mnist": 99.7, "dtd": 79.4,
}
PRETRAINED_PUBLISHED = {
    "sun397": 63.2, "cars": 59.2, "resisc45": 60.2, "eurosat": 45.0,
    "svhn": 31.6, "gtsrb": 32.6, "mnist": 48.3, "dtd": 44.4,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=str(RESULTS / "sanity"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(TASKS8, splits=(args.split,))
    base = base_vision_state(load_base_clip("cpu"))
    res = {"split": args.split, "n": {t: ev.n(t, args.split) for t in TASKS8}}

    ev.load_state(base)
    r16 = ev.evaluate(split=args.split)
    ev.amp = False
    r32 = ev.evaluate(split=args.split)
    ev.amp = True
    res["pretrained"] = r16["acc"]
    res["pretrained_fp32"] = r32["acc"]
    res["seconds_per_full_eval_fp16"] = r16["seconds"]
    res["seconds_per_full_eval_fp32"] = r32["seconds"]
    print("pretrained fp16", {k: round(100 * v, 2) for k, v in r16["acc"].items()}, f"{r16['seconds']:.1f}s")
    print("pretrained fp32", {k: round(100 * v, 2) for k, v in r32["acc"].items()}, f"{r32['seconds']:.1f}s")

    matrix = {}
    for e in TASKS8:
        ev.load_state(load_expert_state(e))
        r = ev.evaluate(split=args.split)
        matrix[e] = r["acc"]
        print(f"expert {e:9s}", " ".join(f"{100 * r['acc'][t]:5.1f}" for t in TASKS8), f"{r['seconds']:.1f}s", flush=True)
        if e == "dtd":
            ev.amp = False
            r32 = ev.evaluate(split=args.split)
            ev.amp = True
            res["dtd_expert_fp32"] = r32["acc"]
    res["expert_matrix"] = matrix  # matrix[expert][task]
    res["individual"] = {t: matrix[t][t] for t in TASKS8}
    res["published_individual"] = FUSIONBENCH_PUBLISHED
    res["published_pretrained"] = PRETRAINED_PUBLISHED
    (out / f"expert_matrix_{args.split}.json").write_text(json.dumps(res, indent=1))
    print("individual", {t: round(100 * matrix[t][t], 2) for t in TASKS8})
    print("avg individual", 100 * sum(matrix[t][t] for t in TASKS8) / 8)
    print("max |fp16-fp32| pretrained", max(abs(res["pretrained"][t] - res["pretrained_fp32"][t]) for t in TASKS8))
    print("max |fp16-fp32| dtd expert", max(abs(matrix["dtd"][t] - res["dtd_expert_fp32"][t]) for t in TASKS8))
    print("cuda mem GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
