"""提出用の動画（YouTube）を作る：見出し画面と3つの場面をつなぐ（等倍、1920×1080）。

使い方: uv run python scripts/make_final_video.py
出力: outputs/final_video.mp4
  ① 基準書の読み分け（outputs/step1/video_v1.mp4 から、基準書で行き先が変わる部品だけを左右に並べる）
  ② 流れるベルト（outputs/step2_conveyor/compare.mp4）
  ③ VLAとの比較（outputs/step3_vla/compare_vla_A.mp4、compare_vla_B.mp4）
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
W, H, FPS = 1920, 1080, 25
BG = (16, 24, 38)
V1 = "outputs/step1/video_v1.mp4"
OUT = Path("outputs/final_video.mp4")
ENC = ["-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]


def font(size):
    return ImageFont.truetype(FONT, size, index=0)


def card(path, title, lines):
    """見出し画面"""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 430), title, font=font(64), fill="white", anchor="mm")
    for i, line in enumerate(lines):
        d.text((W // 2, 560 + i * 60), line, font=font(38), fill=(200, 210, 225), anchor="mm")
    im.save(path)


def band(path, text):
    """左右に並べた動画（高さ540）の上の余白に入れる説明（中央は透明）"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 270], fill=BG + (255,))
    d.rectangle([0, 810, W, H], fill=BG + (255,))
    size = 44
    while d.textlength(text, font=font(size)) > W - 120:  # 長い文は文字を小さくして収める
        size -= 2
    d.text((W // 2, 135), text, font=font(size), fill="white", anchor="mm")
    im.save(path)


def run(args):
    subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []

        def still(name, title, lines, sec=4):
            card(tmp / f"{name}.png", title, lines)
            run(["-loop", "1", "-t", str(sec), "-i", tmp / f"{name}.png", *ENC, tmp / f"{name}.mp4"])
            parts.append(tmp / f"{name}.mp4")

        def wide(name, src, text):
            """横長（1920×540）の比較動画に上下の余白と説明を付ける"""
            band(tmp / f"{name}.png", text)
            run(["-i", src, "-i", tmp / f"{name}.png", "-filter_complex",
                 f"[0]pad={W}:{H}:0:270[v];[v][1]overlay", *ENC, tmp / f"{name}.mp4"])
            parts.append(tmp / f"{name}.mp4")

        def pair(name, left, right, text):
            """video_v1 の2つの区間（開始秒, 終了秒）を左右に並べる。短い方は最後のコマを伸ばす"""
            dur = max(left[1] - left[0], right[1] - right[0])
            band(tmp / f"{name}.png", text)
            run(["-ss", str(left[0]), "-t", str(left[1] - left[0]), "-i", V1,
                 "-ss", str(right[0]), "-t", str(right[1] - right[0]), "-i", V1,
                 "-i", tmp / f"{name}.png", "-filter_complex",
                 f"[0]scale=960:540,tpad=stop_mode=clone:stop_duration=5[l];"
                 f"[1]scale=960:540,tpad=stop_mode=clone:stop_duration=5[r];"
                 f"[l][r]hstack,trim=duration={dur},setpts=PTS-STARTPTS,pad={W}:{H}:0:270[v];[v][2]overlay",
                 *ENC, tmp / f"{name}.mp4"])
            parts.append(tmp / f"{name}.mp4")

        still("title", "検査基準書を読んで仕分けるAI検品アーム",
              ["Genesisによるシミュレーション（等倍再生）"])
        still("card1", "① 基準書を差し替えると、同じ部品の仕分け先が変わる",
              ["判定：Claude（LLMが基準書を読み、VLMが検査画像を判定）",
               "左：標準の基準書　／　右：厳格の基準書（動作はスクリプト）"])
        pair("v1_stain", (20.0, 28.5), (67.0, 76.0), "色ムラ・小　　左：標準 → OK（青の箱）　／　右：厳格 → NG（橙の箱）")
        pair("v1_scratch", (37.5, 46.5), (85.0, 93.5), "傷・小　　左：標準 → OK（青の箱）　／　右：厳格 → NG（橙の箱）")
        still("card2", "② 速度を教えずに、流れ続けるベルトから部品をつかむ",
              ["左：速度4cm/sを仮定したスクリプト　／　右：ACT（カメラ画像から模倣学習）"])
        wide("conveyor", "outputs/step2_conveyor/compare.mp4",
             "左：スクリプト（4cm/sを仮定）　／　右：ACT　　前半 4cm/s、後半 6cm/s")
        still("card3", "③ 判定も動作も1つのモデル（VLA）に任せるとどうなるか",
              ["前半：VLAが学習した基準書を読み分ける",
               "後半：学習にない基準書「傷のみ」（色ムラは問題にしない）"])
        wide("vla_a", "outputs/step3_vla/compare_vla_A.mp4",
             "VLA・色ムラ・小　　左：標準 → OK　／　右：厳格 → NG")
        wide("vla_b", "outputs/step3_vla/compare_vla_B.mp4",
             "「傷のみ」・色ムラ・大（正解はOK）　左：VLA → NG（失敗）　／　右：Claude＋ACT → OK（成功）")

        (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        run(["-f", "concat", "-safe", "0", "-i", tmp / "list.txt", "-c", "copy", OUT])
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
