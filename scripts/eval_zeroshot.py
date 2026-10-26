"""Zero-shot baseline: evaluate a COCO-pretrained YOLO on VisDrone val, on the classes the two share.

VisDrone GT is relabelled into COCO class IDs (see src/visdrone.py), so the unmodified 80-class model
can be scored with the standard Ultralytics evaluator. AP is only computed for classes present in GT,
so predictions of unrelated COCO classes (dog, chair, ...) don't affect the result.

Usage:
    python scripts/eval_zeroshot.py
    python scripts/eval_zeroshot.py --models yolo11n.pt yolo11s.pt --imgsz 640 1280

Caveats (state these next to the numbers in the README):
  - VisDrone "ignored regions" are dropped rather than masked, so detections inside them count as
    false positives. The fine-tuned model is evaluated the same way, so the comparison is consistent.
  - Tricycles / awning-tricycles have no COCO class and are dropped; if the model calls them
    motorcycles, those are false positives for "motorcycle".
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Task
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.utils import REPO_ROOT, env_info
from src.visdrone import VISDRONE_TO_COCO

CLEARML_PROJECT = "Object_Tracking/VisDrone"
SRC_DIR = REPO_ROOT / "data" / "VisDrone2019-DET-val"
OUT_DIR = REPO_ROOT / "datasets" / "VisDrone_val_coco_labels"


def build_dataset(coco_names):
    """Write YOLO-format labels with COCO class IDs; hardlink images alongside. Returns dataset YAML path."""
    img_out, lbl_out = OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # cheap to rebuild; avoids stale labels/cache
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    n_boxes = 0
    for img_path in sorted((SRC_DIR / "images").glob("*.jpg")):
        try:
            os.link(img_path, img_out / img_path.name)  # no extra disk space
        except OSError:
            shutil.copy2(img_path, img_out / img_path.name)

        W, H = Image.open(img_path).size
        ann = np.loadtxt(SRC_DIR / "annotations" / f"{img_path.stem}.txt",
                         delimiter=",", ndmin=2, usecols=range(8)).astype(int)
        lines = []
        for x, y, w, h, _score, cls, _trunc, _occ in ann:
            if cls not in VISDRONE_TO_COCO or w <= 0 or h <= 0:
                continue
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            cx, cy, bw, bh = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
            lines.append(f"{VISDRONE_TO_COCO[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)

    yaml_path = OUT_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in coco_names.items())
    yaml_path.write_text(f"path: {OUT_DIR.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names}\n")
    print(f"built {OUT_DIR}: {len(list(img_out.iterdir()))} images, {n_boxes} boxes")
    return yaml_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["yolo11n.pt"])
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    args = p.parse_args()

    env = env_info()
    data_yaml = build_dataset(YOLO(args.models[0]).names)

    for weights in args.models:
        model = YOLO(weights)
        for imgsz in args.imgsz:
            run_name = f"{Path(weights).stem}_{imgsz}"
            # One task per model/imgsz so each is comparable to fine-tuned runs in the UI.
            task = Task.init(
                project_name=CLEARML_PROJECT,
                task_name=f"zeroshot_{run_name}",
                task_type=Task.TaskTypes.testing,
                tags=["zeroshot", Path(weights).name, f"imgsz{imgsz}"],
                reuse_last_task_id=False,
                auto_connect_frameworks=False,
            )
            params = {"model": Path(weights).name, "imgsz": imgsz, "batch": 8, "max_det": 500}
            task.connect(params, name="Eval")
            task.connect(env, name="Environment")

            metrics = model.val(data=str(data_yaml), imgsz=imgsz, batch=params["batch"],
                                max_det=params["max_det"], split="val",
                                project=str(REPO_ROOT / "runs" / "zeroshot"), name=run_name, exist_ok=True)
            box = metrics.box

            rows = [("all", box.map50, box.map)] + [
                (metrics.names[int(c)], box.ap50[i], box.ap[i]) for i, c in enumerate(box.ap_class_index)
            ]
            logger = task.get_logger()
            logger.report_single_value("zeroshot/mAP50", round(float(box.map50), 4))
            logger.report_single_value("zeroshot/mAP50-95", round(float(box.map), 4))
            logger.report_table("Per-class AP (zero-shot)", "val", iteration=0, table_plot=pd.DataFrame(
                [(c, round(float(a50), 4), round(float(a), 4)) for c, a50, a in rows],
                columns=["class", "AP50", "AP50-95"]).set_index("class"))
            task.close()

            print(f"\n=== {run_name} (zero-shot, COCO-shared classes) ===")
            print(f"{'class':<12}{'AP50':>8}{'AP50-95':>10}")
            for cls_name, ap50, ap in rows:
                print(f"{cls_name:<12}{ap50:>8.3f}{ap:>10.3f}")


if __name__ == "__main__":
    main()
