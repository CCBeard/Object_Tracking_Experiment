"""Shared helpers: config loading, environment capture, and speed measurement."""

import platform
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=None):
    """Load a YAML experiment config and apply `key=value` overrides to its `train` section."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        cfg["train"][key] = yaml.safe_load(value)  # parses "5" -> 5, "true" -> True, "0.1" -> 0.1
    return cfg


def env_info():
    """Hardware and package versions, recorded alongside every run for reproducibility."""
    import torch
    import ultralytics

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "none"

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "git_commit": commit,
    }


def measure_fps(model, data, imgsz, n=200, warmup=20):
    """Batch-1 predict speed on val images: (FPS, mean ms per image), pre + inference + postprocess."""
    from ultralytics.data.utils import check_det_dataset

    images = sorted(Path(check_det_dataset(data)["val"]).glob("*.jpg"))[: n + warmup]
    ms = []
    for i, img in enumerate(images):
        speed = model.predict(img, imgsz=imgsz, verbose=False)[0].speed
        if i >= warmup:
            ms.append(sum(speed.values()))
    mean_ms = sum(ms) / len(ms)
    return 1000 / mean_ms, mean_ms
