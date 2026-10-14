"""Fine-tune a YOLO model

Usage:
    python scripts/train.py --config configs/yolo11n_640.yaml
    python scripts/train.py --config configs/yolo11n_640.yaml --set epochs=1 fraction=0.05   # smoke test

Outputs:
    runs/visdrone/<name>/          Ultralytics run dir (weights, curves, confusion matrix, results.csv)
    ClearML task                   project CLEARML_PROJECT; config, env, curves, plots, best.pt,
                                   val_best metrics and per-class AP
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from clearml import Task
from ultralytics import YOLO

from src.utils import REPO_ROOT, env_info, load_config

CLEARML_PROJECT = "Object_Tracking/VisDrone"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment YAML")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override train args, e.g. --set epochs=1 batch=8")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)

    # Created before YOLO so the Ultralytics ClearML callback logs into this task instead of making its own.
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=cfg["name"],
        tags=[cfg["model"], f"imgsz{cfg['train']['imgsz']}"],
        output_uri=True,  # upload best.pt to the ClearML file server
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},  # the callback logs these itself
    )
    task.connect(cfg, name="Experiment")
    logger = task.get_logger()

    name, train_args = cfg["name"], cfg["train"]
    env = env_info()
    task.connect(env, name="Environment")
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

    per_class = pd.DataFrame({
        "class": [metrics.names[int(c)] for c in box.ap_class_index],
        "AP50": box.ap50.round(4),
        "AP50-95": box.ap.round(4),
    }).set_index("class")

    for k, v in {"precision": box.mp, "recall": box.mr, "mAP50": box.map50, "mAP50-95": box.map,
                 "train_hours": train_hours}.items():
        logger.report_single_value(f"val_best/{k}", round(float(v), 4))
    logger.report_table("Per-class AP (val_best)", "val", iteration=0, table_plot=per_class)
    task.close()

    print(f"[{name}] done in {train_hours:.2f} h  mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}")


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
