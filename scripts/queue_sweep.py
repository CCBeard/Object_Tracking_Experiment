"""Create one ClearML training task per sweep config and enqueue them for clearml-agent.

Each task runs `scripts/train.py --config <yaml>` from this repo at the current commit (plus any diff against the
remote). Commit the configs first: untracked files are not captured, so the agent would not find them.

Usage:
    python scripts/queue_sweep.py configs/sweep/yolo11s_640.yaml configs/sweep/yolo11n_1280.yaml configs/sweep/yolo11s_1024.yaml
    python scripts/queue_sweep.py configs/sweep/*.yaml --queue sweep

Tasks run in the order given, one at a time, by a clearml-agent listening on the same queue.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearml import Task

from src.utils import REPO_ROOT, load_config

CLEARML_PROJECT = "Object_Tracking/VisDrone"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", type=Path, help="experiment YAMLs, queued in this order")
    p.add_argument("--queue", default="sweep")
    args = p.parse_args()

    for cfg_path in args.configs:
        name = load_config(cfg_path)["name"]
        task = Task.create(
            project_name=CLEARML_PROJECT,
            task_name=name,
            task_type=Task.TaskTypes.training,
            script=str(REPO_ROOT / "scripts" / "train.py"),
            argparse_args=[("config", cfg_path.resolve().relative_to(REPO_ROOT).as_posix())],
            requirements_file=str(REPO_ROOT / "requirements.txt"),
            add_task_init_call=False,  # train.py calls Task.init itself
        )
        task.add_tags(["sweep"])
        Task.enqueue(task, queue_name=args.queue)
        print(f"queued {name} ({task.id}) on '{args.queue}'")


if __name__ == "__main__":
    main()
