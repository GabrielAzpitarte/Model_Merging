"""LoRA: is the argmin-kappa (= argmax sigma_min) per-map selection just 'pick the functionally gentlest expert'?

Functional energy of each adapted map's update on pooled calibration inputs (budget B, master grams of the
same base CLIP): e_ip = tr(Delta_ip C_p Delta_ip^T). Reports within-map Spearman(sigma_min(W_i), e_i) and the
agreement of argmax-sigma_min with argmin-e; then evaluates hard selections on dev (alpha 0.8, 1.0):
argmin-kappa, argmin-energy (gentlest), argmax-energy (strongest).
Writes results/aa/lora/gentle.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

torch.set_num_threads(4)

from condmerge.aa_common import TaskData, new_model  # noqa: E402
from condmerge.lora import LoRABank  # noqa: E402


def main() -> None:
    dev = "cuda"
    rows = list(csv.DictReader((RESULTS / "aa" / "lora" / "lora_stats.csv").open()))
    tab = {(r["param"], r["expert"]): r for r in rows}
    bank = LoRABank(TASKS8, dev)
    names = bank.linear
    g = torch.load(CACHE / "grams" / "base.pt", map_location="cpu", weights_only=False)
    E = np.zeros((len(names), 8))
    for pi, n in enumerate(names):
        c = g["grams"][g["inputs_of"][n]].double().to(dev)
        for i in range(8):
            d = bank.deltas[n][i].double()
            E[pi, i] = float(torch.trace(d @ c @ d.T))
    SN = np.array([[float(tab[(n, t)]["smin_W"]) for t in TASKS8] for n in names])
    K = np.array([[float(tab[(n, t)]["kappa2_W"]) for t in TASKS8] for n in names])
    rho = float(np.mean([spearmanr(SN[i], E[i]).statistic for i in range(len(names))]))
    agree_gentle = float(np.mean(K.argmin(1) == E.argmin(1)))
    agree_strong = float(np.mean(K.argmin(1) == E.argmax(1)))
    print(f"within-map Spearman(sigma_min(W_i), functional energy) = {rho:+.3f}; argmin-kappa == argmin-energy {agree_gentle:.2f}, "
          f"== argmax-energy {agree_strong:.2f}", flush=True)
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    eye = np.eye(8)
    res = {"spearman_smin_energy": rho, "agree_argmin_kappa_argmin_energy": agree_gentle,
           "agree_argmin_kappa_argmax_energy": agree_strong, "eval": []}
    sels = {"argmin_kappa": K.argmin(1), "argmin_energy(gentlest)": E.argmin(1), "argmax_energy(strongest)": E.argmax(1)}
    for a in (0.8, 1.0):
        for lab, w in sels.items():
            params = bank.merged({n: torch.as_tensor(eye[w[i]], dtype=torch.float32, device=dev) for i, n in enumerate(names)}, a)
            r = data.evaluate(model, "dev", params=params)
            res["eval"].append({"label": lab, "alpha": a, "avg_acc": r["avg_acc"], "acc": r["acc"],
                                "counts": np.bincount(w, minlength=8).tolist()})
            print(f"{lab:26s} a={a} avg={r['avg_acc']:.4f} counts={np.bincount(w, minlength=8).tolist()}", flush=True)
    (RESULTS / "aa" / "lora" / "gentle.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
