import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from condmerge.llm.runner import Runner
torch.set_num_threads(8)
r = Runner("l31_8b", 8, "cuda:0")
K, nm = r.K, len(r.mods)
r.set_merge(np.full((nm, K), 1.0 / K), 1.6)
ids, labels, task_of, _ = r.split("val")
LL = r.LL
# prompt pass only
t0 = time.time(); n = 0
with torch.no_grad():
    for rows, x, mask, lens in LL.batches(ids, max_tokens=8192, max_batch=32):
        root, cache = LL.prompt_pass(r.model, x, mask, lens); n += x.numel(); del cache
torch.cuda.synchronize(); tp = time.time() - t0
print(f"prompt pass only: {tp:.1f}s, padded tokens {n}, real tokens {sum(len(i) for i in ids)}", flush=True)
for k in range(K):
    rows = np.flatnonzero(task_of == k)
    t0 = time.time(); r.ev.predict([ids[i] for i in rows], task_of[rows]); dt = time.time() - t0
    print(r.tasks[k], len(rows), f"{dt:.1f}s", f"mean len {np.mean([len(ids[i]) for i in rows]):.0f}", r.ev.stats, flush=True)
# bigger batches
t0 = time.time()
orig = LL.batches
def big(ids_, max_tokens=8192, max_batch=32): return orig(ids_, max_tokens=16384, max_batch=64)
LL.batches = big
r.ev.predict(ids, task_of); torch.cuda.synchronize()
print(f"full predict with 16384/64 batches: {time.time()-t0:.1f}s maxmem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
