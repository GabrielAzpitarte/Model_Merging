"""Frozen TEST confirmation for LoRA RegMean (task-separated Grams, 512 calib images per task, full solve).

Configurations fixed from dev (results/aa/lora/regmean_<bank>_dev.jsonl): for each bank the dev-best alpha of
each config is reused unchanged. Configs: uniform; RegMean shrink 0.95 (RegMean's fixed off-diagonal
shrinkage); RegMean kappa-cap 1e4 with ridge to zero (conditioning-guided). Per-example correctness is kept
so the kappa-cap minus shrink difference gets a paired bootstrap CI (2000 resamples within each task, mean over
tasks). Writes results/aa/lora/regmean_test.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

from aa_lora_regmean import input_name  # noqa: E402
from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402
from condmerge.regmean_lora import regmean_full  # noqa: E402

CONFIGS = {"uniform": None, "shrink0.95": dict(a=0.95), "kcap1e4_zero": dict(a=1.0, kappa_target=1e4, ridge_to="zero")}


def dev_best_alpha(bank: str, label: str) -> float:
    rows = [json.loads(l) for l in (RESULTS / "aa" / "lora" / f"regmean_{bank}_dev.jsonl").read_text().splitlines() if l.strip()]
    if label == "uniform":
        rr = [r for r in rows if r["method"] == "uniform"]
    else:
        kw = CONFIGS[label]
        rr = [r for r in rows if r["method"] == "regmean" and r["budget"] == "C" and r["solve"] == "full"
              and r["shrink"] == kw["a"] and r["kappa_target"] == kw.get("kappa_target")
              and (r.get("ridge_to") == kw.get("ridge_to") or kw.get("kappa_target") is None)]
    return float(max(rr, key=lambda r: r["avg_acc"])["alpha"])


def correctness(model, data: TaskData, params: dict) -> dict[str, np.ndarray]:
    out = {}
    with torch.no_grad():
        for t in TASKS8:
            x, y = data.data[(t, "test")]
            preds = []
            for i in range(0, x.shape[0], 1024):
                preds.append((data.features(model, x[i:i + 1024], params, "fp16") @ data.heads[t].T).argmax(-1))
            out[t] = (torch.cat(preds) == y).cpu().numpy()
    return out


def main() -> None:
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("test",), dev)
    model = new_model(dev)
    rng = np.random.default_rng(0)
    report = {}
    for bank in ("knots_release", "lora_c5a", "lora"):
        b = load_bank(bank)
        names = sorted(b[TASKS8[0]])
        g = torch.load(CACHE / "aa" / f"grams_lora_{bank}_C.pt", weights_only=False)["grams"]
        corr = {}
        report[bank] = {}
        for label, kw in CONFIGS.items():
            a = dev_best_alpha(bank, label)
            params = dict(base_dev)
            for n in names:
                d = torch.stack([b[t][n] for t in TASKS8]).double()
                if kw is None:
                    m = d.mean(0).float()
                else:
                    m, _ = regmean_full(d, g[input_name(n)].double(), **kw)
                params[n] = base_dev[n] + a * m.to(dev)
            c = correctness(model, data, params)
            corr[label] = c
            acc = {t: float(c[t].mean()) for t in TASKS8}
            report[bank][label] = {"alpha": a, "avg_acc": float(np.mean(list(acc.values()))), "acc": acc}
            print(f"{bank} {label:13s} alpha {a}: TEST {100 * report[bank][label]['avg_acc']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        # paired bootstrap of kcap - shrink (mean over tasks of per-task accuracy difference)
        diffs = []
        for _ in range(2000):
            ds = []
            for t in TASKS8:
                n_t = corr["kcap1e4_zero"][t].shape[0]
                idx = rng.integers(0, n_t, n_t)
                ds.append(corr["kcap1e4_zero"][t][idx].mean() - corr["shrink0.95"][t][idx].mean())
            diffs.append(np.mean(ds))
        diffs = np.array(diffs)
        point = report[bank]["kcap1e4_zero"]["avg_acc"] - report[bank]["shrink0.95"]["avg_acc"]
        report[bank]["kcap_minus_shrink"] = {"point": float(point), "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}
        print(f"{bank}: kcap - shrink = {100 * point:+.2f} [{100 * np.quantile(diffs, 0.025):+.2f}, {100 * np.quantile(diffs, 0.975):+.2f}]", flush=True)
    (RESULTS / "aa" / "lora" / "regmean_test.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
