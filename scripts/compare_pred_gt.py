"""Show ground truth and model predictions side by side for one VisDrone val image.

Usage:
    python scripts/compare_pred_gt.py --index 42
    python scripts/compare_pred_gt.py                      # random image
    python scripts/compare_pred_gt.py --index 42 --imgsz 1024 --conf 0.4 --weights runs/visdrone/y11n_1024/weights/best.pt
    python scripts/compare_pred_gt.py --index 42 --model-view   # draw on the 640px image the model sees
    python scripts/compare_pred_gt.py --index 42 --save    # also write results/figures/compare/

In a notebook:
    from ultralytics import YOLO
    from src.viz import compare
    model = YOLO("runs/visdrone/y11n_640-2/weights/best.pt")
    compare(model, index=42, model_view=True)
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.utils import RESULTS_DIR
from src.viz import compare, list_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None, help="image index in sorted val set (default: random)")
    p.add_argument("--weights", default="runs/visdrone/y11n_640-2/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--model-view", action="store_true",
                   help="draw on the downsampled, letterboxed image the network actually sees")
    p.add_argument("--save", action="store_true", help="save PNG to results/figures/compare/")
    args = p.parse_args()

    n = len(list_images())
    if args.index is None:
        args.index = random.randrange(n)
    elif not 0 <= args.index < n:
        p.error(f"--index must be in [0, {n - 1}]")

    model = YOLO(args.weights)
    save_to = None
    if args.save:
        tag = Path(args.weights).parents[1].name
        save_to = RESULTS_DIR / "figures" / "compare" / f"{tag}_idx{args.index}{'_modelview' if args.model_view else ''}.png"
    compare(model, args.index, args.imgsz, args.conf, save_to=save_to, model_view=args.model_view)
    if save_to:
        print(f"wrote {save_to}")
    plt.show()


if __name__ == "__main__":
    main()
