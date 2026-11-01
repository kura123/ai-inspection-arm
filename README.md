# AI検品アーム

検査基準書（日本語の文章）を読んで、ベルトコンベアで流れてくる部品をOKの箱とNGの箱に仕分けるロボットアーム（Genesisのシミュレーション）。

- 判定：LLMが基準書を読んで判定ルールに変換し、VLMが検査カメラの画像から判定する（Claude API）
- 動作：カメラ画像から模倣学習した方策（ACT）が、流れ続けるベルトから部品をつかんで箱に入れる
- 比較：判定も動作も1つのモデルで行うVLA（SmolVLA）を追加学習し、判定と動作を分けた構成と比べた

## レポートと動画

- レポート：[docs/report.md](docs/report.md)
- 動画：https://youtu.be/7UCoAdr0RrY

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA GPU（CUDA） |
| Python | 3.12（[uv](https://docs.astral.sh/uv/) で管理） |
| 主なライブラリ | Genesis 1.4.5、PyTorch 2.11.0、Anthropic SDK 2.3.0、LeRobot 0.6.1（学習用） |
| LLM・VLM | Claude Opus 5.5（`claude-opus-5-5`）。Claude APIのAPIキーが必要 |
| フォント | 動画の日本語字幕に Noto Sans CJK を使う（`sudo apt install fonts-noto-cjk`） |

## セットアップ

```bash
uv sync                    # 判定と仕分けに必要なもの
uv sync --extra train      # 学習（ACT、SmolVLA）も行う場合
cp .env.example .env       # .env の ANTHROPIC_API_KEY に自分のAPIキーを記入
uv run python scripts/check_env.py   # GPU、Genesis、APIキーの有無を確認
```

## 実行方法

学習データ（`data/`）、学習済みモデル・評価結果・動画（`outputs/`）はリポジトリに含めていないため、以下のコマンドで作り直す。

### 基準書の読み取り・判定・仕分け（Claude APIを使う）

```bash
# 5個を仕分けて動画に残す（基準書：standard＝標準、strict＝厳格）
uv run python scripts/run_step1.py --standard standard
uv run python scripts/run_step1.py --standard strict

# 50個で判定と仕分けの正解率を測る（動画なし）
uv run python scripts/run_step1.py --standard standard --per-variant 10 --seed 1 --no-video
uv run python scripts/run_step1.py --standard strict   --per-variant 10 --seed 1 --no-video
```

### 流れ続けるベルトでの模倣学習（ACT）

```bash
# お手本の生成（150エピソード）→ LeRobot形式への変換 → 学習
uv run python scripts/gen_demos.py --config configs/step2_conveyor.yaml --episodes 150
uv run python scripts/to_lerobot.py --config configs/step2_conveyor.yaml
bash scripts/train_act.sh conveyor

# 評価（act：ACT、script：4cm/sを仮定したスクリプト、oracle：正確な速度を使うスクリプト）
for m in act script oracle; do
  uv run python scripts/eval_act.py --config configs/step2_conveyor.yaml --mode $m \
    --policy outputs/train/act_conveyor/checkpoints/last/pretrained_model \
    --episodes 50 --seed 1000 --speeds 0.02 0.03 0.04 0.05 0.06
done
```

### VLA（SmolVLA）との比較

```bash
# お手本の生成（300エピソード）→ LeRobot形式への変換 → 追加学習（20,000ステップ）
uv run python scripts/gen_demos.py --config configs/step3_vla.yaml --episodes 300
uv run python scripts/to_lerobot.py --config configs/step3_vla.yaml
bash scripts/train_vla.sh 20000

# 評価：VLA（標準・厳格・学習にない基準書「傷のみ」、各50個）
uv run python scripts/eval_vla.py --mode vla --per-variant 10 --video 5

# 評価：学習にない基準書を、VLAとClaude＋ACTで同じ条件で比べる（Claude＋ACTはClaude APIを使う）
uv run python scripts/eval_vla.py --mode vla        --standards scratch_only --per-variant 10 --tag _scratch_only
uv run python scripts/eval_vla.py --mode claude_act --standards scratch_only --per-variant 10
```

### 動画とレポートの図

```bash
bash scripts/make_compare_video.sh          # ベルトでの比較（左：スクリプト、右：ACT）
bash scripts/make_vla_video.sh              # VLAとClaude＋ACTの比較（Claude APIを使う）
uv run python scripts/make_final_video.py   # 提出用の動画 → outputs/final_video.mp4
uv run python scripts/make_report_figs.py   # レポートの図1・図2 → docs/figures/
```

`make_final_video.py` は、判定と仕分けの動画（標準・厳格の2本をつないだもの）と2つの比較動画を使う。2本は次のコマンドでつなぐ。

```bash
FF=$(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
printf "file '%s'\n" "$PWD/outputs/step1/standard_seed0/run.mp4" "$PWD/outputs/step1/strict_seed0/run.mp4" > outputs/step1/list.txt
"$FF" -f concat -safe 0 -i outputs/step1/list.txt -c copy outputs/step1/video_v1.mp4
```
