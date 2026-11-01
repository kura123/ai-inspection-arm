"""レポートの図1（検品ラインの全体、注釈付き）と図2（部品5種類）を作る。

使い方: uv run python scripts/make_report_figs.py
出力: docs/figures/fig1.png、docs/figures/fig2.png（元のファイルは docs/report_figures.md を参照）
"""

import subprocess
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

D = "docs/figures"
VIDEO = "outputs/step3_vla/eval_claude_act_seed1000/eval.mp4"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
f = ImageFont.truetype(FONT, 26, index=0)

# 図1：注釈付きの全体
with tempfile.TemporaryDirectory() as tmp:  # 評価動画の1秒時点のコマ
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", "1", "-i", VIDEO,
                    "-frames:v", "1", f"{tmp}/frame.png"], check=True)
    im = Image.open(f"{tmp}/frame.png").convert("RGB")
d = ImageDraw.Draw(im)
def label(text, xy, target):
    x, y = xy
    d.line([xy, target], fill=(255, 220, 0), width=4)
    d.ellipse([target[0]-6, target[1]-6, target[0]+6, target[1]+6], fill=(255, 220, 0))
    b = d.textbbox((x, y), text, font=f, anchor="mm")
    d.rectangle([b[0]-10, b[1]-6, b[2]+10, b[3]+6], fill=(255, 255, 255), outline=(255, 220, 0), width=3)
    d.text((x, y), text, font=f, fill=(20, 20, 20), anchor="mm")
label("ロボットアーム", (930, 360), (760, 300))
label("NGの箱", (300, 330), (480, 395))
label("OKの箱", (960, 600), (790, 520))
label("部品", (240, 470), (490, 515))
label("ベルトコンベア", (330, 660), (450, 585))
d.text((600, 470), "流れる向き", font=f, fill=(255, 220, 0), anchor="mm")
# 部品の流れる向き
d.line([(560, 545), (640, 500)], fill=(255, 220, 0), width=5)
d.polygon([(640, 500), (618, 502), (628, 520)], fill=(255, 220, 0))
im.save(f"{D}/fig1.png")

# 図2：部品5種類
src = "outputs/step3_vla/eval_claude_act_seed1000/images"
items = [("scratch_only_04.png", "良品"), ("scratch_only_06.png", "傷・小"), ("scratch_only_01.png", "傷・大"),
         ("scratch_only_00.png", "色ムラ・小"), ("scratch_only_02.png", "色ムラ・大")]
w, pad, cap = 300, 12, 50
out = Image.new("RGB", (len(items) * (w + pad) + pad, w + cap + pad), "white")
d2 = ImageDraw.Draw(out)
for i, (fn, name) in enumerate(items):
    x = pad + i * (w + pad)
    out.paste(Image.open(f"{src}/{fn}").convert("RGB").resize((w, w)), (x, pad))
    d2.text((x + w // 2, pad + w + cap // 2), name, font=f, fill=(20, 20, 20), anchor="mm")
out.save(f"{D}/fig2.png")
print("ok")
