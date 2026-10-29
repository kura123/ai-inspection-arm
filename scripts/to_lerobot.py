"""Step2のお手本データ（npz）をLeRobotDataset形式に変換する。

使い方: uv run python scripts/to_lerobot.py [--config configs/step2.yaml]   （事前に uv sync --extra train）
出力: 設定の data.lerobot_root（例 data/step2/lerobot/。ローカルのみで、Hubには送らない）
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from inspection_arm.conveyor_task import state_names
from inspection_arm.sort_task import ACTION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    names = state_names(cfg)
    raw = Path(cfg["data"]["raw_dir"])
    root = Path(cfg["data"]["lerobot_root"])
    if root.exists():
        shutil.rmtree(root)  # 生成し直したnpzと食い違わないよう、毎回作り直す

    h, w = cfg["camera"]["policy"]["res"]
    image = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.front": image,
        "observation.images.wrist": image,
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create(repo_id=cfg["data"]["repo_id"], fps=cfg["sim"]["control_hz"],
                               features=features, root=root, robot_type="franka",
                               use_videos=True)

    files = [r["file"] for r in json.loads((raw / "summary.json").read_text()) if "file" in r]
    for f in files:
        ep = np.load(raw / f)
        for t in range(len(ep["action"])):
            ds.add_frame({
                "observation.images.front": ep["front"][t],
                "observation.images.wrist": ep["wrist"][t],
                "observation.state": ep["state"][t],
                "action": ep["action"][t],
                "task": str(ep["task"]),
            })
        ds.save_episode()
        print(f"{f}: {len(ep['action'])} frames")
    ds.finalize()
    print(f"episodes: {len(files)}, saved to {root}")


if __name__ == "__main__":
    main()
