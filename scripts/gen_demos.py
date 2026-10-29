"""Step2のお手本データ生成：スクリプトで仕分けを行い、20Hzで観測と行動を記録する。

使い方: uv run python scripts/gen_demos.py [--config configs/step2.yaml] [--episodes 100] [--seed 0]
        （流れ続けるベルト: --config configs/step2_conveyor.yaml）
出力: 設定の data.raw_dir（例 data/step2/raw/）ep_XXXX.npz（成功したエピソードのみ）、summary.json、preview.mp4（最初の数エピソード）
"""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from inspection_arm.cell import Cell
from inspection_arm.parts import VARIANTS, export_part, sample_defect
from inspection_arm.conveyor_task import make_task
from inspection_arm.sort_task import task_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=3, help="動画に残すエピソード数")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["data"]["raw_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 見た目の違う部品を種類ごとに2個ずつ用意し、エピソードごとに1個を使う
    defects = [sample_defect(v, rng) for v in VARIANTS for _ in range(2)]
    files = [export_part(out / "parts" / f"{i:02d}.glb", cfg["part"]["size"], d, rng)
             for i, d in enumerate(defects)]
    cell = Cell(cfg, files, None)
    task = make_task(cell, cfg)
    preview = imageio.get_writer(out / "preview.mp4", fps=cfg["sim"]["control_hz"], codec="libx264")

    summary = []
    n_saved = 0
    for ep in range(args.episodes):
        t0 = time.time()
        part = int(rng.integers(len(files)))
        bin_name = str(rng.choice(["OK", "NG"]))
        task.reset(part, bin_name, rng)
        frames = {"front": [], "wrist": [], "state": [], "action": []}
        for target, grip in task.expert_plan():
            obs = task.observe()
            action = task.expert_action(target, grip)
            task.apply_action(action)
            for k in ("front", "wrist", "state"):
                frames[k].append(obs[k])
            frames["action"].append(action)
            if ep < args.preview:
                preview.append_data(np.concatenate([obs["front"], obs["wrist"]], axis=1))
        res = task.result()
        row = {"episode": ep, "part": part, "variant": defects[part].variant, "bin": bin_name,
               "frames": len(frames["action"]), "speed": getattr(task, "speed", 0.0), **res}
        if res["success"]:
            np.savez_compressed(out / f"ep_{n_saved:04d}.npz", task=task_text(bin_name),
                                **{k: np.stack(v) for k, v in frames.items()})
            row["file"] = f"ep_{n_saved:04d}.npz"
            n_saved += 1
        summary.append(row)
        print(f"ep {ep:03d} {bin_name} {defects[part].variant:14s} speed={row['speed']:.3f} success={res['success']} "
              f"({time.time() - t0:.1f}s)")

    preview.close()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"expert success: {n_saved}/{args.episodes}, saved to {out}")


if __name__ == "__main__":
    main()
