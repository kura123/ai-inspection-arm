#!/usr/bin/env bash
# Step2: お手本データでACTを学習する（L40Sで30,000ステップ約30分）
# 使い方:
#   bash scripts/train_act.sh                 # 止まった部品（data/step2/lerobot → outputs/train/act）
#   bash scripts/train_act.sh conveyor        # 流れ続けるベルト（data/step2_conveyor/lerobot → outputs/train/act_conveyor）
# 事前に uv sync --extra train と scripts/to_lerobot.py
#
# chunk_size: 一度に予測する行動の数（20 = 1秒分 @20Hz）
# n_action_steps: 予測のうち実際に実行する数（10 = 0.5秒ごとに予測し直す。評価時に変えられる）
set -euo pipefail

if [[ "${1:-}" == "conveyor" ]]; then
  name=inspection_sort_conveyor; root=data/step2_conveyor/lerobot; out=outputs/train/act_conveyor
else
  name=inspection_sort; root=data/step2/lerobot; out=outputs/train/act
fi

uv run lerobot-train \
  --dataset.repo_id=local/${name} \
  --dataset.root=${root} \
  --policy.type=act \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false \
  --output_dir=${out} \
  --job_name=act_${name}
