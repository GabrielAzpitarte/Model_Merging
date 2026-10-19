"""Cross-check: evaluate TA@alpha with the ORIGINAL CoreSpace path (vendored lib + icamerge from the source repo)
in today's environment, against the recorded number and the concatenation path."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from condmerge.llm import banks as BK
bank, T, ta = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
src = {"l31_8b": "/shared/ssd_30T/azpitarte_llama31/repo/src", "l32_3b": "/data/home/azpitarte/lora_merging_ica_margin/src"}[bank]
sys.path.insert(0, src)
import torch
torch.set_num_threads(8)
D, LL = BK.modules(bank)
core = LL.load_core(T)  # cached pickle of the original pipeline
model = LL.load_model("cuda:0") if bank == "l32_3b" else LL.load_model("cuda:0", dtype=torch.bfloat16)
wr = LL.install(model, core, "cuda:0")
ev = LL.Evaluator(model, wr, D.TASKSETS[T])
cores = {m: ta * cs.cores.astype(np.float64).sum(0) for m, cs in core.items()}
res = ev.score(cores, "val")
print(f"CoreSpace path TA@{ta}|val J{100*res['joint_macro']:.2f} L{100*res['local_macro']:.2f}", flush=True)
print(json.dumps({"joint": res["joint_per_task"], "local": res["local_per_task"]}))
