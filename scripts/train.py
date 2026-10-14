"""Fine-tune a YOLO model from a YAML config, then log final val metrics to results/.

Usage:
    python scripts/train.py --config configs/y11n_640.yaml
    python scripts/train.py --config configs/y11n_640.yaml --set epochs=1 fraction=0.05   # smoke test

Outputs:
    runs/visdrone/<name>/          Ultralytics run dir (weights, curves, confusion matrix, results.csv)
    results/runs.csv               one row per run: headline metrics + hardware/versions
    results/per_class/<name>.csv   per-class AP50 and AP50-95 on val
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.utils import RESULTS_DIR, REPO_ROOT, append_csv, env_info, load_config, timestamp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment YAML")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override train args, e.g. --set epochs=1 batch=8")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    name, train_args = cfg["name"], cfg["train"]
    env = env_info()
    print(f"[{name}] env: {env}")

    # --- train ---
    model = YOLO(cfg["model"])
    t0 = time.time()
    model.train(data=cfg["data"], project=str(REPO_ROOT / "runs" / "visdrone"), name=name, **train_args)
    train_hours = (time.time() - t0) / 3600
    save_dir = Path(model.trainer.save_dir)  # may be name2, name3... if the dir already existed

    # Snapshot exactly what was run next to the weights.
    with open(save_dir / "experiment.json", "w") as f:
        json.dump({"config": cfg, "env": env}, f, indent=2)

    # --- evaluate best checkpoint on val ---
    best = YOLO(model.trainer.best)
    val_batch = train_args["batch"] if train_args.get("batch", 16) > 0 else 16
    metrics = best.val(data=cfg["data"], imgsz=train_args["imgsz"], batch=val_batch,
                       max_det=train_args.get("max_det", 300),
                       split="val", project=str(save_dir), name="val_best")
    box = metrics.box

    append_csv(RESULTS_DIR / "runs.csv", {
        "timestamp": timestamp(),
        "name": name,
        "run_dir": save_dir.relative_to(REPO_ROOT).as_posix(),
        "model": cfg["model"],
        "imgsz": train_args["imgsz"],
        "epochs": train_args["epochs"],
        "batch": train_args["batch"],
        "seed": train_args.get("seed"),
        "precision": round(box.mp, 4),
        "recall": round(box.mr, 4),
        "mAP50": round(box.map50, 4),
        "mAP50-95": round(box.map, 4),
        "train_hours": round(train_hours, 2),
        **env,
    })

    for i, cls_idx in enumerate(box.ap_class_index):
        append_csv(RESULTS_DIR / "per_class" / f"{save_dir.name}.csv", {
            "class": metrics.names[int(cls_idx)],
            "AP50": round(float(box.ap50[i]), 4),
            "AP50-95": round(float(box.ap[i]), 4),
        })

    print(f"[{name}] done in {train_hours:.2f} h  mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}")
    print(f"[{name}] logged to {RESULTS_DIR / 'runs.csv'}")


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
