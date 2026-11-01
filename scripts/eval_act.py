"""Step2の評価：学習したACT（またはお手本のスクリプト）で仕分けを行い、成功率を測る。

使い方:
  uv run python scripts/eval_act.py --mode act    [--episodes 50] [--seed 1000] [--video 5]
  uv run python scripts/eval_act.py --mode script [--episodes 50] [--seed 1000]
  uv run python scripts/eval_act.py --mode act --disturb 0.03   # 接近中に部品を3 cmずらす
流れ続けるベルト（--config configs/step2_conveyor.yaml）:
  --mode act / script（決められた速度を前提に推定）/ oracle（真の位置と速度を使う。お手本と同じ）
  --speeds 0.02 0.03 0.04 0.05 0.06 で、エピソードごとに速度を順に切り替える
出力: outputs/step2/eval_<mode>_seed<seed>/ に results.jsonl と動画（最初の --video エピソード）
お手本とは別のseedで部品の見た目・位置・向き・開始位置・入れる箱をばらつかせる。同じseedならACTとスクリプトは同じ条件になる。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from inspection_arm.cell import Cell
from inspection_arm.parts import VARIANTS, export_part, sample_defect
from inspection_arm.policy_io import act_step, load_policy
from inspection_arm.conveyor_task import ConveyorTask, make_task
from inspection_arm.sort_task import task_text

MODE_NAMES = {"act": "ACT（模倣学習）", "script": "スクリプト制御（お手本）",
              "oracle": "スクリプト制御（真の速度を使用）"}
CONVEYOR_MODE_NAMES = {**MODE_NAMES, "script": "スクリプト制御（速度4cm/sを前提）"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2.yaml")
    parser.add_argument("--mode", choices=["act", "script", "oracle"], default="act")
    parser.add_argument("--policy", default="outputs/train/act/checkpoints/last/pretrained_model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=160, help="1エピソードの上限（20Hzの周期数）")
    parser.add_argument("--video", type=int, default=5, help="動画に残すエピソード数")
    parser.add_argument("--speeds", type=float, nargs="*", help="ベルトの速度 [m/s]（順に使う）")
    parser.add_argument("--n-action-steps", type=int, help="ACTが予測し直すまでに実行する数")
    parser.add_argument("--disturb", type=float, default=0.0,
                        help="接近中に部品をずらす距離 [m]（向きはランダム、0ならずらさない）")
    parser.add_argument("--disturb-step", type=int, default=16,
                        help="部品をずらす周期（お手本では部品の真上に着くのが24周期目）")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = f"eval_{args.mode}_seed{args.seed}"
    if args.n_action_steps:
        name += f"_nas{args.n_action_steps}"
    if args.disturb > 0:
        name += f"_disturb{round(args.disturb * 100)}cm"
    out = Path("outputs") / Path(args.config).stem / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    defects = [sample_defect(v, rng) for v in VARIANTS for _ in range(2)]
    files = [export_part(out / "parts" / f"{i:02d}.glb", cfg["part"]["size"], d, rng)
             for i, d in enumerate(defects)]
    cell = Cell(cfg, files, out / "eval.mp4" if args.video > 0 else None)
    writer = cell.writer
    task = make_task(cell, cfg)
    conveyor = isinstance(task, ConveyorTask)
    names = CONVEYOR_MODE_NAMES if conveyor else MODE_NAMES
    if args.mode == "act":
        policy, pre, post = load_policy(args.policy, args.n_action_steps)

    results = []
    for ep in range(args.episodes):
        cell.writer = writer if ep < args.video else None
        part = int(rng.integers(len(files)))
        bin_name = str(rng.choice(["OK", "NG"]))
        if conveyor:
            speed = args.speeds[ep % len(args.speeds)] if args.speeds else None
            task.reset(part, bin_name, rng, speed=speed)
        else:
            task.reset(part, bin_name, rng)
        title = [names[args.mode], f"エピソード {ep + 1}/{args.episodes}: {bin_name}の箱へ入れる"]
        if conveyor:
            title[1] += f"（ベルト {task.speed * 100:.1f}cm/s）"
        max_z = 0.0
        angle = rng.uniform(0, 2 * np.pi)
        shift = args.disturb * np.array([np.cos(angle), np.sin(angle), 0.0])

        def disturb(t: int):
            if args.disturb > 0 and t == args.disturb_step:
                p = cell.parts[part].get_pos().cpu().numpy()
                cell.parts[part].set_pos(p + shift, zero_velocity=True)
                title.append(f"部品を{args.disturb * 100:.0f}cmずらした")
        if args.mode == "act":
            policy.reset()
            for t in range(args.max_steps):
                disturb(t)
                obs = task.observe()
                cell.overlay = {"lines": title, "inset": Image.fromarray(obs["wrist"]),
                                "inset_label": "手首カメラ"}
                task.apply_action(act_step(policy, pre, post, obs, task_text(bin_name)))
                max_z = max(max_z, float(cell.parts[part].get_pos()[2]))
        else:
            estimate = "nominal" if args.mode == "script" else "oracle"
            for t, (target, grip) in enumerate(task.expert_plan(estimate)):
                disturb(t)
                obs = task.observe()
                cell.overlay = {"lines": title, "inset": Image.fromarray(obs["wrist"]),
                                "inset_label": "手首カメラ"}
                task.apply_action(task.expert_action(target, grip))
                max_z = max(max_z, float(cell.parts[part].get_pos()[2]))
        res = task.result()
        picked = max_z > cell.part_z + 0.1
        verdict = "成功" if res["success"] else "失敗"
        cell.overlay = {"lines": [*title, f"結果: {verdict}（入った箱: {res['landed'] or 'なし'}）"]}
        cell.hold(1.0)
        row = {"episode": ep, "variant": defects[part].variant, "bin": bin_name,
               "speed": getattr(task, "speed", 0.0), "picked": picked, **res}
        results.append(row)
        print(f"ep {ep:03d} speed={row['speed']:.3f} target={bin_name} landed={res['landed']} picked={picked} "
              f"success={res['success']}")

    cell.writer = writer
    cell.close()
    with open(out / "results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(results)
    wrong = sum(r["landed"] not in (None, r["bin"]) for r in results)
    print(f"[{args.mode}] success: {sum(r['success'] for r in results)}/{n}, "
          f"picked: {sum(r['picked'] for r in results)}/{n}, wrong bin: {wrong}/{n}")
    for b in ("OK", "NG"):
        rows = [r for r in results if r["bin"] == b]
        print(f"  target {b}: {sum(r['success'] for r in rows)}/{len(rows)}")
    if conveyor:
        for v in sorted({round(r["speed"], 3) for r in results}) if args.speeds else []:
            rows = [r for r in results if round(r["speed"], 3) == v]
            print(f"  speed {v * 100:.1f}cm/s: {sum(r['success'] for r in rows)}/{len(rows)}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
