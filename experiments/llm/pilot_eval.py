"""Pilot: (1) reproduce recorded numbers through the concatenation path (TA@alpha val, one expert val), (2) time."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from condmerge.llm import banks as BK
from condmerge.llm.runner import Runner, normalized
bank, T = sys.argv[1], int(sys.argv[2])
import torch
torch.set_num_threads(8)
r = Runner(bank, T, "cuda:0")
print("load", round(r.load_seconds, 1), "s; tasks", r.tasks, flush=True)
K, nm = r.K, len(r.mods)
store = Path(BK.BANKS[bank].lora_dir).parent
rec_ta = json.loads((store / f"T{T}" / "eval_base_TA.json").read_text())
rec_ex = json.loads((store / f"T{T}" / "eval_base_zeroshot_experts.json").read_text())
uni = np.full((nm, K), 1.0 / K)
out = {}
for ta_alpha in ([0.2] if T == 8 else [rec_ta["TA|alpha"]]):
    alpha = ta_alpha * K  # pipeline TA = ta_alpha * sum_t C_t = (ta_alpha K) * mean
    chk = r.set_merge(uni, alpha)
    met, pred = r.evaluate("val")
    k = f"TA@{ta_alpha}|val"
    print(k, "mine J%.2f L%.2f" % (100 * met["joint_macro"], 100 * met["local_macro"]),
          "recorded J%.2f L%.2f" % (100 * rec_ta[k]["joint_macro"], 100 * rec_ta[k]["local_macro"]),
          "secs %.0f" % met["seconds"], "fp", round(chk["fingerprint"], 6), flush=True)
    out[k] = {"mine": met, "recorded": rec_ta[k]}
t0 = r.tasks[0]
lam = np.zeros((nm, K)); lam[:, 0] = 1.0
r.set_merge(lam, 1.0)
met, _ = r.evaluate("val", only_task=0)
key = f"expert_{t0}|val"
print(key, "mine L%.4f" % met["local_per_task"][0], "recorded L%.4f" % rec_ex[key]["local_per_task"][0], "secs %.0f" % met["seconds"], flush=True)
print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)
