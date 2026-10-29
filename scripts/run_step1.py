"""Step1：基準書をLLMが読み、流れてくる部品をVLMが判定し、Frankaが良品箱／NG箱に仕分ける。

使い方: uv run python scripts/run_step1.py --standard standard [--per-variant 1] [--seed 0]
出力: outputs/step1/<standard>_seed<seed>/ に動画（run.mp4）、判定結果（results.jsonl）、ルール、検査画像
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from inspection_arm.cell import Cell
from inspection_arm.inspector import Inspector, Judgment, Rules
from inspection_arm.parts import VARIANTS, export_part, sample_defect

# 動画の字幕に使う日本語名
STANDARD_NAMES = {"standard": "標準", "strict": "厳格"}
DEFECT_NAMES = {"none": "なし", "scratch": "傷", "stain": "色ムラ", "chip": "欠け", "other": "その他"}
VERDICT_NAMES = {"OK": "良品（OK）→ 青の箱へ", "NG": "不良（NG）→ 橙の箱へ"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", default="standard")
    parser.add_argument("--per-variant", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--reuse-judgments", action="store_true",
                        help="前回のresults.jsonlの判定を使い、APIを呼ばずに仕分けだけやり直す")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("configs/step1.yaml").read_text())
    std = cfg["standards"][args.standard]
    out = Path(cfg["output_dir"]) / f"{args.standard}_seed{args.seed}"
    (out / "images").mkdir(parents=True, exist_ok=True)

    # 部品を作る（種類の順番はシャッフルし、同じseedなら基準書が違っても同じ部品の列になる）
    rng = np.random.default_rng(args.seed)
    variants = [v for v in VARIANTS for _ in range(args.per_variant)]
    rng.shuffle(variants)
    defects = [sample_defect(v, rng) for v in variants]
    files = [export_part(out / "parts" / f"{i:02d}.glb", cfg["part"]["size"], d, rng)
             for i, d in enumerate(defects)]

    inspector = Inspector(cfg["llm"]["model"], cfg["llm"]["effort"])
    if args.reuse_judgments:
        rules = Rules.model_validate_json((out / "rules.json").read_text())
        rows = [json.loads(line) for line in open(out / "results.jsonl")]
        saved = [Judgment.model_validate({k: r[k] for k in Judgment.model_fields}) for r in rows]
    else:
        standard_text = Path(std["file"]).read_text()
        rules = inspector.read_standard(standard_text)
        (out / "rules.json").write_text(json.dumps(rules.model_dump(), ensure_ascii=False, indent=2))

    cell = Cell(cfg, files, None if args.no_video else out / "run.mp4")
    title = f"基準書: {STANDARD_NAMES.get(args.standard, args.standard)}"
    cell.overlay = {"lines": [title, "LLMが基準書を読み、判定ルールに変換"]}
    cell.go_home()
    cell.hold(1.5)

    results = []
    for i, d in enumerate(defects):
        cell.overlay = {"lines": [title, f"部品 {i + 1}/{len(defects)}: 検査位置へ"]}
        cell.feed(i)
        image = cell.capture()
        image.save(out / "images" / f"{i:02d}.png")
        cell.overlay = {"lines": [title, f"部品 {i + 1}/{len(defects)}: VLMが判定中…"], "inset": image}
        cell.hold(0.8)
        # 判定を待つ間、シミュレーションは止まっている
        j = saved[i] if args.reuse_judgments else inspector.judge(image, rules)
        truth = "NG" if d.is_ng(std["thresholds"]) else "OK"
        cell.overlay = {
            "lines": [title, f"部品 {i + 1}/{len(defects)}: {VERDICT_NAMES[j.verdict]}",
                      f"欠陥: {DEFECT_NAMES[j.defect_type]}  大きさ: {j.estimated_ratio:.2f}"],
            "inset": image, "verdict": j.verdict,
        }
        cell.hold(1.2)
        sort = cell.pick_and_place(i, j.verdict)
        row = {"id": i, "variant": d.variant, "defect": d.to_dict(), "truth": truth,
               "judge_correct": j.verdict == truth, **sort, **j.model_dump()}
        results.append(row)
        print(f"{i:02d} {d.variant:14s} truth={truth} pred={j.verdict} picked={sort['picked']} "
              f"in_bin={sort['in_bin']}  {j.reason}")

    cell.overlay = {"lines": [title, "すべて仕分け完了"]}
    cell.hold(1.0)
    cell.close()

    with open(out / "results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(results)
    judge_ok = sum(r["judge_correct"] for r in results)
    sorted_ok = sum(r["judge_correct"] and r["in_bin"] for r in results)
    print(f"judgment accuracy: {judge_ok}/{n}, sorted into the correct bin: {sorted_ok}/{n}")
    u = inspector.usage
    print(f"tokens: in={u['input_tokens']} out={u['output_tokens']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
