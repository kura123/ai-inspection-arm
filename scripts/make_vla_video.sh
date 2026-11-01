#!/usr/bin/env bash
# Step3の比較動画：流れ続けるベルトで、同じ部品・同じ速度の2つの実行を左右に並べる
# 使い方: bash scripts/make_vla_video.sh [場面Aのseed（既定 4000）] [場面Bのseed（既定 4001）]
#   （Claudeを2回呼ぶ。約0.02ドル）
# 出力: outputs/step3_vla/compare_vla.mp4
#   A: VLAが指示文を読み分ける（色ムラ・小。左：標準の基準書→OK、右：厳格の基準書→NG）
#   B: 学習にない基準書「傷のみ」（色ムラ・大、正解はOK。左：VLA、右：Claude＋ACT）
# 動画は例示。成功率は eval_vla.py の50個ずつの評価結果を使う
set -euo pipefail

seed_a=${1:-4000}
seed_b=${2:-4001}
FF=$(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
dir=outputs/step3_vla
common="--per-variant 1 --speeds 0.04 --video 1"

run() {  # mode standard variant tag seed
  uv run python scripts/eval_vla.py --mode $1 --standards $2 --variants $3 --tag $4 --seed $5 $common \
    | grep -E "^(standard|strict|scratch_only) "
}
run vla standard stain_small _videoA_standard $seed_a
run vla strict stain_small _videoA_strict $seed_a
run vla scratch_only stain_large _videoB_vla $seed_b
run claude_act scratch_only stain_large _videoB_claude $seed_b

# 短い方の動画は最後のコマを伸ばして長さをそろえ、左右に並べる
pair() {  # left right out
  dur=$(uv run python -c "
import imageio.v2 as i
print(max(i.get_reader(f).get_meta_data()['duration'] for f in ['$1', '$2']))")
  "$FF" -y -loglevel error -i "$1" -i "$2" -filter_complex \
    "[0]scale=960:540,tpad=stop_mode=clone:stop_duration=10[l];[1]scale=960:540,tpad=stop_mode=clone:stop_duration=10[r];[l][r]hstack,trim=duration=$dur" \
    -c:v libx264 -pix_fmt yuv420p "$3"
}
pair $dir/eval_vla_seed${seed_a}_videoA_standard/eval.mp4 $dir/eval_vla_seed${seed_a}_videoA_strict/eval.mp4 \
  $dir/compare_vla_A.mp4
pair $dir/eval_vla_seed${seed_b}_videoB_vla/eval.mp4 $dir/eval_claude_act_seed${seed_b}_videoB_claude/eval.mp4 \
  $dir/compare_vla_B.mp4

printf "file '%s'\n" "$PWD/$dir/compare_vla_A.mp4" "$PWD/$dir/compare_vla_B.mp4" > $dir/compare_vla_list.txt
"$FF" -y -loglevel error -f concat -safe 0 -i $dir/compare_vla_list.txt -c copy $dir/compare_vla.mp4
echo "saved: $dir/compare_vla.mp4"
