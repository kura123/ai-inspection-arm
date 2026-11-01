"""Step3の評価：流れ続けるベルトで、基準書に従った仕分けができるかを測る。

- vla: SmolVLAが、基準書の指示文と画像（検査・前方・手首）から判定も動作も行う
- claude_act: ClaudeがStep1と同じく基準書を読んで検査画像を判定し、その結果（入れる箱）でStep2のACTが動く

使い方:
  uv run python scripts/eval_vla.py --mode vla [--standards standard strict scratch_only]
  uv run python scripts/eval_vla.py --mode claude_act --standards scratch_only    # API費用 約0.4ドル/50個
出力: outputs/step3_vla/eval_<mode>_seed<seed>/ に results.jsonl と動画（最初の --video エピソード）
基準書ごとに5種類の部品を --per-variant 個ずつ流す。成功は「その基準書での正解の箱に入ったこと」。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from inspection_arm.cell import Cell
from inspection_arm.conveyor_task import ConveyorTask
from inspection_arm.parts import VARIANTS, export_part, sample_defect
from inspection_arm.policy_io import act_step, load_policy
from inspection_arm.sort_task import task_text

MODE_NAMES = {"vla": "VLA（SmolVLA：判定も動作も）", "claude_act": "Claude（判定）＋ACT（動作）"}
STANDARD_NAMES = {"standard": "標準", "strict": "厳格", "scratch_only": "傷のみ（学習にない基準書）"}
DEFAULT_POLICY = {"vla": "outputs/train/smolvla/checkpoints/last/pretrained_model",
                  "claude_act": "outputs/train/act_conveyor/checkpoints/last/pretrained_model"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step3_vla.yaml")
    parser.add_argument("--mode", choices=["vla", "claude_act"], default="vla")
    parser.add_argument("--policy", help="学習済みモデル（省略時はモードごとの既定）")
    parser.add_argument("--standards", nargs="*", default=["standard", "strict", "scratch_only"])
    parser.add_argument("--per-variant", type=int, default=10, help="基準書ごと・種類ごとのエピソード数")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--speeds", type=float, nargs="*", default=[0.02, 0.03, 0.04, 0.05, 0.06])
    parser.add_argument("--max-steps", type=int, default=160, help="1エピソードの上限（20Hzの周期数）")
    parser.add_argument("--n-action-steps", type=int, help="予測し直すまでに実行する数")
    parser.add_argument("--video", type=int, default=5, help="基準書ごとに動画に残すエピソード数")
    parser.add_argument("--tag", default="", help="出力フォルダ名の末尾に付ける文字列")
    parser.add_argument("--variants", nargs="*", help="流す部品の種類を絞る（動画用。省略時は5種類すべて）")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    vla_cfg = cfg["task"]["vla"]
    standards = {**vla_cfg["standards"], **vla_cfg["unseen"]}
    out = Path("outputs") / Path(args.config).stem / f"eval_{args.mode}_seed{args.seed}{args.tag}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 評価用の部品は学習とは別のseedで作る（見た目が学習データと違う）
    n = vla_cfg["parts_per_variant"]
    defects = [sample_defect(v, rng) for v in VARIANTS for _ in range(n)]
    files = [export_part(out / "parts" / f"{i:02d}.glb", cfg["part"]["size"], d, rng)
             for i, d in enumerate(defects)]
    cell = Cell(cfg, files, out / "eval.mp4" if args.video > 0 else None)
    writer = cell.writer
    task = ConveyorTask(cell, cfg)
    task.with_target_bin = args.mode == "claude_act"  # ACTは入れる箱を状態で受け取る
    policy, pre, post = load_policy(args.policy or DEFAULT_POLICY[args.mode], args.n_action_steps)
    if args.mode == "claude_act":
        from inspection_arm.inspector import Inspector
        inspector = Inspector(cfg.get("llm", {}).get("model", "claude-opus-5-5"),
                              cfg.get("llm", {}).get("effort", "medium"))

    results = []
    for name in args.standards:
        std = standards[name]
        if args.mode == "claude_act":
            rules = inspector.read_standard(Path(std["file"]).read_text())
            (out / f"rules_{name}.json").write_text(
                json.dumps(rules.model_dump(), ensure_ascii=False, indent=2))
        variants = [v for v in (args.variants or VARIANTS) for _ in range(args.per_variant)]
        rng.shuffle(variants)
        for k, variant in enumerate(variants):
            cell.writer = writer if k < args.video else None
            part = list(VARIANTS).index(variant) * n + int(rng.integers(n))
            truth = "NG" if defects[part].is_ng(std["thresholds"]) else "OK"
            speed = args.speeds[k % len(args.speeds)]
            task.reset(part, truth, rng, speed=speed)
            title = [MODE_NAMES[args.mode], f"基準書: {STANDARD_NAMES.get(name, name)}",
                     f"部品 {k + 1}/{len(variants)}（ベルト {speed * 100:.0f}cm/s）"]
            verdict, judgment = None, {}
            Image.fromarray(task.inspect_image).save(out / "images" / f"{name}_{k:02d}.png")
            if args.mode == "claude_act":
                j = inspector.judge(Image.fromarray(task.inspect_image), rules)
                verdict, judgment = j.verdict, j.model_dump()
                task.bin_name = verdict  # ACTにはClaudeの判定を入れる箱として渡す
                title.append(f"Claudeの判定: {verdict}")
                text = task_text(verdict)
            else:
                text = std["instruction"]
            inset = Image.fromarray(task.inspect_image)
            max_z = 0.0
            policy.reset()
            for _ in range(args.max_steps):
                obs = task.observe()
                cell.overlay = {"lines": title, "inset": inset, "inset_label": "検査カメラ"}
                task.apply_action(act_step(policy, pre, post, obs, text))
                max_z = max(max_z, float(cell.parts[part].get_pos()[2]))
            landed = task.landed_bin()
            row = {"standard": name, "variant": variant, "speed": speed, "truth": truth,
                   "verdict": verdict, "landed": landed, "success": landed == truth,
                   "picked": max_z > cell.part_z + 0.1, "image": f"images/{name}_{k:02d}.png",
                   "judgment": judgment}
            results.append(row)
            cell.overlay = {"lines": [*title, f"正解: {truth} / 入った箱: {landed or 'なし'} → "
                                              f"{'成功' if row['success'] else '失敗'}"],
                            "inset": inset, "inset_label": "検査カメラ"}
            cell.hold(1.0)
            print(f"{name:12s} {variant:14s} speed={speed:.2f} truth={truth} verdict={verdict} "
                  f"landed={landed} picked={row['picked']} success={row['success']}")

    cell.writer = writer
    cell.close()
    with open(out / "results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 基準書×部品の種類ごとの成功数
    table = defaultdict(lambda: [0, 0])
    for r in results:
        for key in [(r["standard"], r["variant"]), (r["standard"], "合計")]:
            table[key][0] += r["success"]
            table[key][1] += 1
    print(f"[{args.mode}]")
    for name in args.standards:
        cells = "  ".join(f"{v}={table[(name, v)][0]}/{table[(name, v)][1]}" for v in VARIANTS)
        tot = table[(name, "合計")]
        picked = sum(r["picked"] for r in results if r["standard"] == name)
        print(f"  {name:12s} success={tot[0]}/{tot[1]} picked={picked}/{tot[1]}  {cells}")
    if args.mode == "claude_act":
        u = inspector.usage
        print(f"  tokens: in={u['input_tokens']} out={u['output_tokens']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
