"""Project paths and the task registry.

Task order follows the task-vectors / FusionBench convention
(SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"  # symlink to /data/home/azpitarte/condmerge_data (outside git)
CACHE = ARTIFACTS / "cache"
CKPT = ARTIFACTS / "checkpoints"
RESULTS = ROOT / "results"

BASE_MODEL = "openai/clip-vit-base-patch32"

TASKS8 = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]

# FusionBench repo suffix for each canonical task name.
FUSIONBENCH_NAME = {
    "sun397": "sun397",
    "cars": "stanford-cars",
    "resisc45": "resisc45",
    "eurosat": "eurosat",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "mnist": "mnist",
    "dtd": "dtd",
}

# Hugging Face dataset specs (identical to the ICA project's configs/vision/vitb32.yaml,
# so dev/test splits are the same images as in that project).
DATASETS = {
    "cars": {"path": "tanganke/stanford_cars"},
    "dtd": {"path": "tanganke/dtd"},
    "eurosat": {"path": "tanganke/eurosat"},
    "gtsrb": {"path": "tanganke/gtsrb"},
    "mnist": {"path": "ylecun/mnist"},
    "resisc45": {"path": "tanganke/resisc45"},
    "sun397": {"path": "tanganke/sun397"},
    "svhn": {"path": "ufldl-stanford/svhn", "name": "cropped_digits"},
}

# KnOTS rank-16 LoRA experts (same 8 tasks), from the earlier project.
KNOTS_LORA_ROOT = Path("/data/home/azpitarte/lora_merging_ica/checkpoints/vitb32")


def fusionbench_dir(task: str, arch: str = "basepatch32") -> Path:
    return CKPT / f"fusionbench_{arch}" / FUSIONBENCH_NAME[task]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
