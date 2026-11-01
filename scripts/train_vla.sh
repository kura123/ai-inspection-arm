#!/usr/bin/env bash
# Step3: 公開済みの SmolVLA（lerobot/smolvla_base）を、流れ続けるベルトのお手本で追加学習する
# 使い方: bash scripts/train_vla.sh [ステップ数（既定 10000）] [出力先（既定 outputs/train/smolvla）]
# 事前に uv sync --extra train と scripts/to_lerobot.py --config configs/step3_vla.yaml
#
# VLM（画像と言語の部分）は凍結したまま、動作を出す部分（action expert）だけを学習する（SmolVLAの既定）
# 指示文は基準書の英語の要約（48トークン以内）。入れる箱は教えない
# カメラ名は smolvla_base に合わせて camera1〜3 に読み替える（前方・手首・検査の順）
# chunk_size は事前学習と同じ50（20Hzで2.5秒）。n_action_steps はACTと同じ10（0.5秒ごとに予測し直す）
set -euo pipefail

steps=${1:-10000}
out=${2:-outputs/train/smolvla}

uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/inspection_sort_vla \
  --dataset.root=data/step3_vla/lerobot \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.inspect": "observation.images.camera3"}' \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=${steps} \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir=${out} \
  --job_name=smolvla_inspection_sort
