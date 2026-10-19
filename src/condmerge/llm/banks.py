"""The two Llama LoRA banks used by the LLM extension, read-only.

Both banks were trained by earlier projects with the same recipe: PEFT LoRA r = 16, lora_alpha = 16 (scale 1,
absorbed into B by the training scripts), dropout 0, no bias, on q/k/v/o_proj of every block, BF16 backbone and
FP32 adapters; nothing else was trained. Evaluation code is a vendored, byte-identical snapshot of each project's
evaluator (vendor/llama31 from repo commit ce0d375, vendor/llama32 from latent_merge commit 301362c), so upstream
branch changes cannot alter running jobs.

  l31_8b  meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f2 (official gated repo; safetensors sha256 in STORE/sha.txt),
          20 experts, multi-token answers scored by the exact sum log-probability (vendor/llama31/llama31_lib.py)
  l32_3b  unsloth/Llama-3.2-3B-Instruct @ 006f5dcd (byte-identical weights to the meta-llama repo per its
          project), 20 experts, single-token answers (vendor/llama32/llama_lib.py)

Effective expert matrix of adapted map p (exact, assembled in float64):
    W_i[p] = W0[p] + Delta_i[p],   Delta_i[p] = B_i[p] A_i[p]   (scale s_i = 1 already inside B_i)
W0 is stored in BF16 (the released checkpoint precision); A_i, B_i in FP32. The deployed expert is exactly this
matrix: the evaluators add the FP32 side path B (A x) to the BF16 base output.

Splits (identical to the projects'): evaluation source permuted with seed 0, first 20 % = val (cap 250 / task),
next = test (cap 1000 / task). NEW here: ``fresh`` = the examples after the test cap (cap 1000 / task), never
used by any earlier run of either project (only tasks with a large enough evaluation source have them).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "vendor"
FRESH_CAP = 1000
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class BankSpec:
    name: str
    vendor: str
    data_mod: str
    lib_mod: str
    model_dir: str
    lora_dir: str
    hf_home: str
    n_layers: int
    multi_token: bool
    model_id: str
    revision: str


BANKS = {
    "l31_8b": BankSpec(
        name="l31_8b", vendor="llama31", data_mod="llama31_data", lib_mod="llama31_lib",
        model_dir="/shared/ssd_30T/azpitarte_llama31/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
                  "0e9e39f249a16976918f6564b8830bc894c89659",
        lora_dir="/shared/ssd_30T/azpitarte_llama31/outputs/lora", hf_home="/shared/ssd_30T/azpitarte_llama31/hf",
        n_layers=32, multi_token=True, model_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="0e9e39f249a16976918f6564b8830bc894c89659"),
    "l32_3b": BankSpec(
        name="l32_3b", vendor="llama32", data_mod="llama_data", lib_mod="llama_lib",
        model_dir="/shared/ssd_30T/molohio_llama32/hf/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/"
                  "006f5dcd1393c3add266de40994ba96225e9689d",
        lora_dir="/shared/ssd_30T/molohio_llama32/outputs/lora", hf_home="/shared/ssd_30T/molohio_llama32/hf",
        n_layers=28, multi_token=False, model_id="unsloth/Llama-3.2-3B-Instruct",
        revision="006f5dcd1393c3add266de40994ba96225e9689d"),
}


def offline_env(spec: BankSpec) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", spec.hf_home)
    os.environ.pop("LLAMA31_VAL_CAP", None)
    os.environ.pop("LLAMA31_TEST_CAP", None)
    os.environ.pop("LLAMA31_OUT", None)


@cache
def modules(bank: str):
    """(data module, lib module) of the vendored evaluator, with MODEL pointed at the local snapshot directory
    (no hub access; the tokenizer and weights are read from the pinned revision)."""
    spec = BANKS[bank]
    offline_env(spec)
    path = str(VENDOR / spec.vendor)
    if path not in sys.path:
        sys.path.insert(0, path)
    D = importlib.import_module(spec.data_mod)
    D.MODEL = spec.model_dir
    LL = importlib.import_module(spec.lib_mod)
    assert D.VAL_CAP == 250 and D.TEST_CAP == 1000, (D.VAL_CAP, D.TEST_CAP)
    return D, LL


def tasks(bank: str, T: int) -> list[str]:
    D, _ = modules(bank)
    return list(D.TASKSETS[T])


def module_names(bank: str) -> list[str]:
    n = BANKS[bank].n_layers
    return [f"model.layers.{i}.self_attn.{p}" for i in range(n) for p in PROJ]


def kind_of(module: str) -> str:
    return module.rsplit(".", 1)[-1]


def layer_of(module: str) -> int:
    return int(module.split(".")[2])


# ----------------------------------------------------------------------------------------------- weights
@cache
def _weight_index(bank: str) -> dict:
    spec = BANKS[bank]
    return json.loads((Path(spec.model_dir) / "model.safetensors.index.json").read_text())["weight_map"]


def base_weight(bank: str, module: str):
    """W0 of an adapted map as stored (torch BF16 tensor, out x in), straight from the safetensors shard."""
    from safetensors import safe_open

    spec = BANKS[bank]
    key = module + ".weight"
    shard = _weight_index(bank)[key]
    with safe_open(str(Path(spec.model_dir) / shard), framework="pt") as fh:
        return fh.get_tensor(key)


def load_lora(bank: str, task: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{module: (A (r x in), B (out x r))} float32, PEFT scale already in B."""
    spec = BANKS[bank]
    z = np.load(Path(spec.lora_dir) / f"{task}.npz")
    out = {}
    for m in module_names(bank):
        a, b = z[f"A_{m}"], z[f"B_{m}"]
        assert a.dtype == np.float32 and b.dtype == np.float32, (m, a.dtype, b.dtype)
        assert a.shape[0] == 16 and b.shape[1] == 16, (m, a.shape, b.shape)
        out[m] = (a, b)
    extra = [k for k in z.files if not (k.startswith("A_") or k.startswith("B_"))]
    assert not extra and len(z.files) == 2 * len(out), ("unexpected arrays in bank file", task, extra)
    return out


def load_lora_module(bank: str, task: str, module: str) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of one module only (np.load of an .npz reads arrays lazily)."""
    spec = BANKS[bank]
    with np.load(Path(spec.lora_dir) / f"{task}.npz") as z:
        a, b = z[f"A_{module}"], z[f"B_{module}"]
    assert a.dtype == np.float32 and b.dtype == np.float32 and a.shape[0] == 16 and b.shape[1] == 16, module
    return a, b


def expert_info(bank: str, task: str) -> dict:
    spec = BANKS[bank]
    return json.loads((Path(spec.lora_dir) / f"{task}.json").read_text())


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest(bank: str, T: int, hash_weights: bool = True) -> dict:
    """Checkpoint identity record: base revision + shard hashes, expert files + hashes + training metadata,
    vendored evaluator hashes, LoRA convention checks."""
    spec = BANKS[bank]
    D, LL = modules(bank)
    out = {"bank": bank, "model_id": spec.model_id, "revision": spec.revision, "model_dir": spec.model_dir,
           "lora_dir": spec.lora_dir, "T": T, "tasks": tasks(bank, T), "multi_token": spec.multi_token,
           "vendor": {p.name: sha256_file(p) for p in sorted((VENDOR / spec.vendor).glob("*")) if p.is_file()}}
    cfg = json.loads((Path(spec.model_dir) / "config.json").read_text())
    out["config"] = {k: cfg.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                             "num_key_value_heads", "torch_dtype", "intermediate_size")}
    if hash_weights:
        out["shards"] = {p.name: sha256_file(p) for p in sorted(Path(spec.model_dir).glob("*.safetensors"))}
    ex = {}
    for t in tasks(bank, T):
        info = expert_info(bank, t)
        assert info["r"] == 16 and info["alpha"] == 16 and list(info["modules"]) == list(PROJ), (t, info)
        assert info["model"] in (spec.model_id, "meta-llama/Llama-3.1-8B-Instruct"), (t, info["model"])
        ex[t] = {"npz_sha256": sha256_file(Path(spec.lora_dir) / f"{t}.npz"), "json": info}
    out["experts"] = ex
    out["lora_convention"] = ("PEFT r=16, lora_alpha=16 -> scale 1 absorbed into B; targets q/k/v/o_proj; "
                              "no bias, no modules_to_save; base frozen BF16")
    return out


# ----------------------------------------------------------------------------------------------- splits
def split_indices(bank: str, task: str, split: str) -> np.ndarray:
    """Indices into D.examples(task, 'eval') for val / test (the projects' own) and fresh (new, unused)."""
    D, _ = modules(bank)
    n = len(D.examples(task, "eval"))
    perm = np.random.default_rng(0).permutation(n)
    n_val = int(round(0.2 * n))
    if split == "val":
        idx = perm[:n_val][:D.VAL_CAP]
    elif split == "test":
        idx = perm[n_val:][:D.TEST_CAP]
    elif split == "fresh":
        idx = perm[n_val + D.TEST_CAP:][:FRESH_CAP]
    else:
        raise ValueError(split)
    if split in ("val", "test"):
        ref = D.eval_indices(task)[0 if split == "val" else 1]
        assert np.array_equal(idx, ref), (bank, task, split)
    return idx


def split_sizes(bank: str, T: int) -> dict:
    return {s: {t: int(len(split_indices(bank, t, s))) for t in tasks(bank, T)} for s in ("val", "test", "fresh")}
