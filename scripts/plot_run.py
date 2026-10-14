"""
Usage:
    python scripts/plot_run.py runs/visdrone/y11n_640-2
    python scripts/plot_run.py runs/visdrone/y11n_640-2 --no-cm     # curves only (skips the ~30 s val pass)

Writes to results/figures/<run_name>_curves.png and <run_name>_confusion.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import RESULTS_DIR

TRAIN_COLOR, VAL_COLOR = "teal", "tan"   # categorical slots 1-2 (blue, orange)
INK, MUTED, GRID = "black", "gray", "white"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def plot_curves(run_dir, out_path):
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    ep = df["epoch"]
    best_ep = int(ep[df["metrics/mAP50-95(B)"].idxmax()])
    n_ep = int(ep.max())
    close_mosaic = 10
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        import yaml
        close_mosaic = yaml.safe_load(args_file.read_text()).get("close_mosaic", 10)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    losses = [("box_loss", "Box regression loss (CIoU)"),
              ("cls_loss", "Classification loss (BCE)"),
              ("dfl_loss", "Distribution focal loss (box edges)")]

    for ax, (key, title) in zip(axes.flat, losses):
        ax.plot(ep, df[f"train/{key}"], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(ep, df[f"val/{key}"], color=VAL_COLOR, lw=2, label="val")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right")

    ax = axes.flat[3]
    ax.plot(ep, df["metrics/mAP50(B)"], color=TRAIN_COLOR, lw=2, label="mAP50")
    ax.plot(ep, df["metrics/mAP50-95(B)"], color=VAL_COLOR, lw=2, label="mAP50-95")
    ax.axvline(best_ep, color=MUTED, lw=1, ls="--")
    ax.annotate(f"best epoch {best_ep}\nmAP50-95 = {df['metrics/mAP50-95(B)'].max():.3f}",
                xy=(best_ep, df["metrics/mAP50-95(B)"].max()), xytext=(-8, -42),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_title("Validation mAP", loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")

    # Ultralytics turns off mosaic augmentation for the final epochs; train loss drops visibly there.
    for a in axes.flat:
        a.axvspan(n_ep - close_mosaic + 0.5, n_ep + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)

    fig.suptitle(f"Training curves - {run_dir.name}   (shaded: final {close_mosaic} epochs, mosaic augmentation off)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_confusion(run_dir, out_path):
    from ultralytics import YOLO

    exp = json.loads((run_dir / "experiment.json").read_text())
    train = exp["config"]["train"]
    metrics = YOLO(run_dir / "weights" / "best.pt").val(
        data=exp["config"]["data"], imgsz=train["imgsz"], batch=16, max_det=train.get("max_det", 300),
        split="val", plots=True, verbose=False,  # plots=False skips filling the confusion matrix
        conf=0.25,  # the CM uses the val conf threshold
        project=str(run_dir), name="val_cm", exist_ok=True)

    # Ultralytics layout: matrix[predicted, true], last row/col = "background".
    cm = metrics.confusion_matrix.matrix
    names = [metrics.names[i] for i in range(len(metrics.names))] + ["background"]
    col_sums = cm.sum(axis=0, keepdims=True)
    norm = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums > 0)
    norm[:, -1] = 0  # background column (false positives) has no meaningful denominator; shown as counts below

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = len(names)
    for i in range(n):
        for j in range(n):
            if j == n - 1:
                if cm[i, j] > 0:
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7, color=MUTED)
            elif norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("true class")
    ax.set_ylabel("predicted class")
    ax.grid(False)
    ax.set_title("Confusion matrix @ conf 0.25, IoU 0.45\n"
                 "columns normalized by true class; background column = false-positive counts",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.75, label="fraction of true-class instances")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--no-cm", action="store_true", help="skip the confusion matrix (needs a val pass)")
    args = p.parse_args()

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(args.run_dir, out_dir / f"{args.run_dir.name}_curves.png")
    if not args.no_cm:
        plot_confusion(args.run_dir, out_dir / f"{args.run_dir.name}_confusion.png")


if __name__ == "__main__":
    main()
