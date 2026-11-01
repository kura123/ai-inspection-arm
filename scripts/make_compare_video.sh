#!/usr/bin/env bash
# Step2の比較動画：流れ続けるベルトで、同じ条件のスクリプト制御（左）とACT（右）を並べる
# 使い方: bash scripts/make_compare_video.sh
# 出力: outputs/step2_conveyor/compare.mp4
#   速度ごとに1エピソードずつ（seedが同じなので、部品・位置・入れる箱は左右で同じ）
set -euo pipefail

FF=$(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
dir=outputs/step2_conveyor
policy=outputs/train/act_conveyor/checkpoints/last/pretrained_model
# seed:速度[m/s]。4cm/sは両方成功する例、6cm/sはスクリプトだけ失敗する例
# （評価全体の成功率は eval_act.py の50エピソードの結果を使う。動画は例示）
cases=("2001:0.04" "3000:0.06")

clips=()
for c in "${cases[@]}"; do
  seed=${c%%:*}; speed=${c##*:}
  for mode in script act; do
    uv run python scripts/eval_act.py --config configs/step2_conveyor.yaml --mode $mode \
      --policy $policy --episodes 1 --seed $seed --video 1 --speeds $speed | grep -E "^ep"
  done
  # 短い方の動画は最後のコマを伸ばして長さをそろえ、左右に並べる
  out=$dir/compare_seed$seed.mp4
  "$FF" -y -loglevel error -i $dir/eval_script_seed$seed/eval.mp4 -i $dir/eval_act_seed$seed/eval.mp4 \
    -filter_complex "[0]scale=960:540,tpad=stop_mode=clone:stop_duration=10[l];[1]scale=960:540,tpad=stop_mode=clone:stop_duration=10[r];[l][r]hstack,trim=duration=$(uv run python -c "
import imageio.v2 as i
print(max(i.get_reader(f).get_meta_data()['duration'] for f in ['$dir/eval_script_seed$seed/eval.mp4', '$dir/eval_act_seed$seed/eval.mp4']))")" \
    -c:v libx264 -pix_fmt yuv420p $out
  clips+=("$out")
done

printf "file '%s'\n" "${clips[@]/#/$PWD/}" > $dir/compare_list.txt
"$FF" -y -loglevel error -f concat -safe 0 -i $dir/compare_list.txt -c copy $dir/compare.mp4
echo "saved: $dir/compare.mp4"
