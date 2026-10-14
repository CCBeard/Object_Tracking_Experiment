"""Side-by-side ground truth vs. model predictions on a VisDrone image."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from src.utils import REPO_ROOT
from src.visdrone import CLASS_COLORS, CLASS_NAMES, YOLO_TO_RAW

DEFAULT_SPLIT = REPO_ROOT / "data" / "VisDrone2019-DET-val"


def list_images(split_dir=DEFAULT_SPLIT):
    return sorted((Path(split_dir) / "images").glob("*.jpg"))


def load_gt(img_path):
    """Raw VisDrone annotations for an image -> (N, 5) array of [x0, y0, x1, y1, raw_class]."""
    ann_path = img_path.parent.parent / "annotations" / f"{img_path.stem}.txt"
    a = np.loadtxt(ann_path, delimiter=",", ndmin=2, usecols=range(8)).astype(int)
    return np.c_[a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], a[:, 5]]


def predict(model, img_path, imgsz=640, conf=0.25):
    """Model predictions -> (M, 6) array of [x0, y0, x1, y1, raw_class, confidence]."""
    r = model.predict(str(img_path), imgsz=imgsz, conf=conf, max_det=500, verbose=False)[0]
    b = r.boxes
    raw_cls = np.array([YOLO_TO_RAW[int(c)] for c in b.cls.cpu().numpy()])
    return np.c_[b.xyxy.cpu().numpy(), raw_cls, b.conf.cpu().numpy()] if len(b) else np.zeros((0, 6))


def letterbox(img, imgsz):
    """Reproduce Ultralytics' predict-time preprocessing: resize long side to imgsz, pad to a multiple of 32.

    Returns the letterboxed image plus (scale, pad_x, pad_y) for mapping full-res box coords into it.
    """
    from ultralytics.data.augment import LetterBox

    h, w = img.shape[:2]
    out = LetterBox(new_shape=imgsz, auto=True, stride=32)(image=img)
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    pad_x, pad_y = round((out.shape[1] - new_w) / 2 - 0.1), round((out.shape[0] - new_h) / 2 - 0.1)
    return out, r, pad_x, pad_y


def to_model_view(boxes, r, pad_x, pad_y):
    """Map full-resolution [x0, y0, x1, y1, ...] boxes into letterboxed coordinates."""
    boxes = boxes.astype(float).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + pad_y
    return boxes


def _draw(ax, img, boxes, title):
    ax.imshow(img, interpolation="nearest")  # show real pixels; default smoothing hides the downsampling
    for x0, y0, x1, y1, cls, *_ in boxes:
        cls = int(cls)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1,
                               edgecolor=CLASS_COLORS[cls], ls="--" if cls == 0 else "-"))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.axis("off")


def compare(model, index=None, imgsz=640, conf=0.25, split_dir=DEFAULT_SPLIT, seed=None, save_to=None,
            model_view=False):
    """Plot GT (left) vs predictions (right) for image `index` (random if None). Returns the figure.

    model_view=True draws both panels on the letterboxed image the network actually receives.
    """
    images = list_images(split_dir)
    if index is None:
        index = int(np.random.default_rng(seed).integers(len(images)))
    img_path = images[index]
    img = np.asarray(Image.open(img_path).convert("RGB"))

    gt = load_gt(img_path)
    pred = predict(model, img_path, imgsz, conf)
    n_gt_scored = int(np.isin(gt[:, 4], list(YOLO_TO_RAW.values())).sum())

    orig_hw = img.shape[:2]
    if model_view:
        img, r, pad_x, pad_y = letterbox(img, imgsz)
        gt, pred = to_model_view(gt, r, pad_x, pad_y), to_model_view(pred, r, pad_x, pad_y)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)
    _draw(axes[0], img, gt, f"Ground truth: {n_gt_scored} objects  (dashed gray = ignored region)")
    _draw(axes[1], img, pred, f"Prediction: {len(pred)} boxes  (conf >= {conf}, imgsz {imgsz})")

    present = sorted(set(gt[:, 4].astype(int)) | set(pred[:, 4].astype(int)))
    fig.legend(handles=[Patch(facecolor="none", edgecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in present],
               loc="lower center", ncol=len(present), frameon=False, bbox_to_anchor=(0.5, -0.04))
    view = (f"model view: {img.shape[1]}x{img.shape[0]} letterboxed from {orig_hw[1]}x{orig_hw[0]}"
            if model_view else f"full resolution {orig_hw[1]}x{orig_hw[0]}")
    fig.suptitle(f"[{index}] {img_path.name}   |   {view}", x=0.01, ha="left", fontsize=10, color="#6b6a63")

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig
