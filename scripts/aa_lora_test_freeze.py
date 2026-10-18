"""KnOTS LoRA bank: frozen configurations (selected on dev) evaluated once on the untouched TEST split.

Frozen from the dev sweep (results/aa/lora/rule_sweep_dev_shard*.jsonl):
  uniform alpha 0.8 (dev-best uniform); kappa b=128 alpha 1.0 (dev-best literal rule); kappa argmin alpha 1.0;
  shuf{0,1,2}:kappa at b=128 alpha 1.0 (null at the literal rule's exact configuration);
  best single expert (eurosat, alpha 0.6; dev-best single); MetaGPT alpha 1.0 (dev-best).
Fixed final-rank budget: rank-16 truncation of the merged update per map for uniform and kappa-b128
(exact merges have rank up to 128). Writes results/aa/lora/test_freeze.jsonl.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from aa_lora_rule_sweep import lora_score_tables, scale_of, softmax_coeffs  # noqa: E402
from condmerge.aa_common import TaskData, new_model  # noqa: E402
from condmerge.lora import LoRABank  # noqa: E402


def main() -> None:
    dev = "cuda"
    bank = LoRABank(TASKS8, dev)
    names = bank.linear
    tables = lora_score_tables(RESULTS / "aa" / "lora" / "lora_stats.csv", names)
    data = TaskData(TASKS8, ("test",), dev)
    model = new_model(dev)
    eye = np.eye(8)

    def co(rule, b):
        beta = b / scale_of(tables[rule]) if math.isfinite(b) else math.inf
        return {p: softmax_coeffs(tables[rule][p], beta) for p in names}

    uni = {p: np.full(8, 1 / 8) for p in names}
    sq = np.zeros(8)
    for p in names:
        sq += bank.deltas[p].double().pow(2).flatten(1).sum(1).cpu().numpy()
    configs = [("uniform", uni, 0.8, None), ("kappa_b128", co("kappa", 128), 1.0, None),
               ("kappa_argmin", co("kappa", math.inf), 1.0, None),
               ("shuf0:kappa_b128", co("shuf0:kappa", 128), 1.0, None), ("shuf1:kappa_b128", co("shuf1:kappa", 128), 1.0, None),
               ("shuf2:kappa_b128", co("shuf2:kappa", 128), 1.0, None),
               ("single:eurosat", {p: eye[TASKS8.index("eurosat")] for p in names}, 0.6, None),
               ("metagpt", {p: sq / sq.sum() for p in names}, 1.0, None),
               ("uniform_rank16", uni, 0.8, 16), ("kappa_b128_rank16", co("kappa", 128), 1.0, 16)]
    with (RESULTS / "aa" / "lora" / "test_freeze.jsonl").open("w") as f:
        for label, c, a, comp in configs:
            params = bank.merged({p: torch.as_tensor(v, dtype=torch.float32, device=dev) for p, v in c.items()}, a, compress_rank=comp)
            r = data.evaluate(model, "test", params=params)
            f.write(json.dumps({"label": label, "alpha": a, "compress_rank": comp, "split": "test",
                                "avg_acc": r["avg_acc"], "acc": r["acc"]}) + "\n")
            f.flush()
            print(f"{label:20s} a={a} rank={comp} TEST avg={r['avg_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
