"""Measure batch-1 FPS for a ClearML model and log it on its training task as val_best/fps and val_best/latency_ms.

For backfilling runs trained before train.py measured FPS. Close other GPU work first; the numbers are only
comparable across runs measured on the same idle machine.

Usage:
    python scripts/measure_fps.py --model-id <id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearml import InputModel, Task
from ultralytics import YOLO

from src.utils import measure_fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="ClearML model ID")
    args = p.parse_args()

    clearml_model = InputModel(model_id=args.model_id)
    task = Task.get_task(task_id=clearml_model.task)
    imgsz = int(task.get_parameter("General/imgsz"))
    data = task.get_parameter("General/data")

    fps, latency_ms = measure_fps(YOLO(clearml_model.get_local_copy()), data, imgsz)
    logger = task.get_logger()
    logger.report_single_value("val_best/fps", round(fps, 4))
    logger.report_single_value("val_best/latency_ms", round(latency_ms, 4))
    task.flush(wait_for_uploads=True)
    print(f"{task.name} @ {imgsz}: {fps:.1f} FPS ({latency_ms:.2f} ms/img), logged to task {task.id}")


if __name__ == "__main__":
    main()
