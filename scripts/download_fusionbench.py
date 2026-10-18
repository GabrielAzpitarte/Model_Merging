"""Download FusionBench full fine-tuned CLIP vision experts (vision tower only).

Each repo holds a ``CLIPVisionModel`` (no visual projection, no text tower)
fine-tuned from ``openai/clip-vit-base-patch32`` with the text head frozen,
following the task-vectors protocol. Files go to
``artifacts/checkpoints/fusionbench_<arch>/<task>/`` (outside git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

TASKS8 = ["sun397", "stanford-cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
TASKS20_EXTRA = [
    "oxford_flowers102", "pcam", "fer2013", "oxford-iiit-pet", "stl10", "cifar100",
    "cifar10", "food101", "fashion_mnist", "emnist_letters", "kmnist", "rendered-sst2",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="clip-vit-base-patch32")
    ap.add_argument("--tasks", default="8", help="'8', '20', or comma list")
    ap.add_argument("--root", default="artifacts/checkpoints")
    args = ap.parse_args()
    tasks = TASKS8 if args.tasks == "8" else TASKS8 + TASKS20_EXTRA if args.tasks == "20" else args.tasks.split(",")
    root = Path(args.root) / f"fusionbench_{args.arch.replace('clip-vit-', '').replace('-', '')}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for t in tasks:
        repo = f"tanganke/{args.arch}_{t}"
        local = root / t
        path = snapshot_download(repo, local_dir=str(local))
        st = local / "model.safetensors"
        manifest[t] = {"repo": repo, "local": str(local), "bytes": st.stat().st_size, "sha256": sha256(st)}
        print("ok", repo, manifest[t]["bytes"], manifest[t]["sha256"][:12], flush=True)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", root)


if __name__ == "__main__":
    main()
