#!/usr/bin/env bash
# 実機（D405）で撮影 → セグメンテーション → 姿勢推定を繰り返し，RViz2 で TF・モデル・推定画像を表示する．
#   bash scripts/run_live_pose.sh                       # hier_improved_v2
#   bash scripts/run_live_pose.sh --save                # 推定ごとに results/live/ に保存
# 終了は Ctrl+C（RViz2 も閉じる）
set -e
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vlm_rag
# 同じネットワークの他の ROS 2 システム（ドメイン 0 の xArm など）と混ざらないよう，ドメイン 11 で PC 内だけで通信する
# 他の PC から見るときは ROS_LOCALHOST_ONLY_LIVE=0 を付けて起動する
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID_LIVE:-11}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY_LIVE:-1}"
rviz2 -d scripts/rviz/live_pose.rviz &
RVIZ_PID=$!
trap 'kill $RVIZ_PID 2>/dev/null' EXIT
python scripts/live_pose_node.py "$@"
