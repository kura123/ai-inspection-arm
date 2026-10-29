"""VLMの試し判定：部品を生成して真上から撮影し、2種類の基準書でClaudeに判定させて正解と比べる。

使い方: uv run python scripts/try_vlm.py [--per-variant 2]
出力: outputs/try_vlm/ に部品画像（images/）、判定結果（results.jsonl）、判定ルール（rules_*.json）
画像が既にあれば撮影は飛ばし、判定だけを行う。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from inspection_arm.inspector import Inspector
from inspection_arm.parts import VARIANTS, Defect, export_part, sample_defect

OUT_DIR = Path("outputs/try_vlm")
SPACING = 0.3  # 部品どうしの間隔 [m]


def render_parts(cfg: dict, per_variant: int) -> list[dict]:
    import genesis as gs

    rng = np.random.default_rng(cfg["sim"]["seed"])
    size = cfg["part"]["size"]
    defects = [sample_defect(v, rng) for v in VARIANTS for _ in range(per_variant)]

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    for i, d in enumerate(defects):
        glb = export_part(OUT_DIR / "parts" / f"{i:02d}.glb", size, d, rng)
        scene.add_entity(gs.morphs.Mesh(file=str(glb), pos=(i * SPACING, 0, size / 2)))
    cam_cfg = cfg["camera"]["inspect"]
    cam = scene.add_camera(
        res=tuple(cam_cfg["res"]), pos=(0, 0, cam_cfg["height"]), lookat=(0, 0, 0), fov=cam_cfg["fov"]
    )
    scene.build()

    records = []
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(defects):
        x = i * SPACING
        # 真上から見下ろす。up方向をyに固定して画像の向きをそろえる
        cam.set_pose(pos=(x, 0, cam_cfg["height"]), lookat=(x, 0, 0), up=(0, 1, 0))
        path = OUT_DIR / "images" / f"{i:02d}.png"
        Image.fromarray(np.asarray(cam.render()[0])).save(path)
        records.append({"id": i, "image": str(path), "defect": d.to_dict()})
    (OUT_DIR / "parts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-variant", type=int, default=2)
    parser.add_argument("--render-only", action="store_true", help="撮影だけ行い、APIは呼ばない")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("configs/step1.yaml").read_text())
    parts_json = OUT_DIR / "parts.json"
    if parts_json.exists() and not args.render_only:
        records = json.loads(parts_json.read_text())
    else:
        records = render_parts(cfg, args.per_variant)
    if args.render_only:
        return

    inspector = Inspector(cfg["llm"]["model"], cfg["llm"]["effort"])
    results = []
    for name, std in cfg["standards"].items():
        rules = inspector.read_standard(Path(std["file"]).read_text())
        (OUT_DIR / f"rules_{name}.json").write_text(
            json.dumps(rules.model_dump(), ensure_ascii=False, indent=2)
        )
        print(f"[{name}] rules: {rules.model_dump()}")
        for r in records:
            d = Defect(**r["defect"])
            truth = "NG" if d.is_ng(std["thresholds"]) else "OK"
            j = inspector.judge(Image.open(r["image"]), rules)
            ok = j.verdict == truth
            results.append({"standard": name, "id": r["id"], "variant": d.variant, "truth": truth,
                            "correct": ok, **j.model_dump()})
            print(f"  {r['id']:02d} {d.variant:14s} truth={truth} pred={j.verdict} "
                  f"{'o' if ok else 'x'}  ({j.defect_type}, {j.estimated_ratio:.2f}) {j.reason}")

    with open(OUT_DIR / "results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in cfg["standards"]:
        rows = [r for r in results if r["standard"] == name]
        print(f"[{name}] accuracy: {sum(r['correct'] for r in rows)}/{len(rows)}")
    u = inspector.usage
    cost = u["input_tokens"] * 4 / 1e6 + u["output_tokens"] * 20 / 1e6  # Opus 5.5の単価
    print(f"tokens: in={u['input_tokens']} out={u['output_tokens']}  (Opus 5.5換算 ${cost:.3f})")


if __name__ == "__main__":
    main()
