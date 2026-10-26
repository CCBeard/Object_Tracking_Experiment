"""Show ground truth and model predictions side by side for one VisDrone val image.

Weights are pulled from the ClearML model registry (Models tab, or the training task's Artifacts > Output Model).
imgsz defaults to the value the model was trained at.

Usage:
    python scripts/compare_pred_gt.py --model-id <id> --index 42
    python scripts/compare_pred_gt.py --model-id <id>                      # random image
    python scripts/compare_pred_gt.py --model-id <id> --index 42 --conf 0.4 --imgsz 640
    python scripts/compare_pred_gt.py --model-id <id> --index 42 --model-view   # draw on the image the model sees
    python scripts/compare_pred_gt.py --model-id <id> --index 42 --upload  # attach figure to the training task

In a notebook:
    from clearml import InputModel
    from ultralytics import YOLO
    from src.viz import compare
    model = YOLO(InputModel("<id>").get_local_copy())
    compare(model, index=42, model_view=True)
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearml import InputModel, Task
from ultralytics import YOLO

from src.viz import compare, list_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="ClearML model ID")
    p.add_argument("--index", type=int, default=None, help="image index in sorted val set (default: random)")
    p.add_argument("--imgsz", type=int, default=None, help="default: the model's training imgsz")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--model-view", action="store_true",
                   help="draw on the downsampled, letterboxed image the network actually sees")
    p.add_argument("--upload", action="store_true", help="attach the figure to the model's training task")
    args = p.parse_args()

    n = len(list_images())
    if args.index is None:
        args.index = random.randrange(n)
    elif not 0 <= args.index < n:
        p.error(f"--index must be in [0, {n - 1}]")

    clearml_model = InputModel(model_id=args.model_id)
    train_task = Task.get_task(task_id=clearml_model.task)
    if args.imgsz is None:
        args.imgsz = int(train_task.get_parameter("General/imgsz"))  # connected by the Ultralytics callback
    model = YOLO(clearml_model.get_local_copy())
    print(f"model {clearml_model.name} ({args.model_id}) from task {train_task.name}, imgsz={args.imgsz}")

    fig = compare(model, args.index, args.imgsz, args.conf, model_view=args.model_view)
    if args.upload:
        series = f"idx{args.index}_imgsz{args.imgsz}_conf{args.conf}{'_modelview' if args.model_view else ''}"
        train_task.get_logger().report_matplotlib_figure("Predictions vs GT", series, fig, iteration=0,
                                                         report_image=True)
        train_task.flush(wait_for_uploads=True)
        print(f"uploaded {series} to ClearML task {train_task.id}")
    plt.show()


if __name__ == "__main__":
    main()
