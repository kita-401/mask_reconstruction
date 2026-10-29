"""D405 でボールを撮影し，色付きモデル点群（カメラ側の面のお椀形）を作る．

カメラノード（scripts/camera_node.py）の配信を受け取るので，先にカメラノードを起動しておく：
    ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1 python scripts/camera_node.py
    ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1 python scripts/capture_ball_model.py --out pose_estimation/models/ball_d405.pcd

ボールは1個だけ，カメラの視野の中に置き，撮影中は動かさない．手順：
  1. 静止した状態で --frames 枚撮り，深度は画素ごとの中央値，色は平均にする（計測ノイズを減らす）
  2. セグメンテーション（推定ノードと同じ設定）で深度を LingBot-Depth で補完し，マスクを求める
  3. 外形の円形度が最も大きいマスクをボールとし，補完後の深度から色付き点群（--voxel 間隔）にする
  4. 統計的外れ値除去のあと球を当てはめ，球面から --shell 以上離れた点を除く
  5. 原点を球の中心に移す（軸は撮影時のカメラの向き．カメラは -z 側にある）
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rclpy
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_node import image_from_msg  # noqa: E402
from export_masks import METHODS  # noqa: E402
from live_pose_node import make_sample, temporal_median_depth  # noqa: E402
from mask_reconstruction import MaskReconstructor  # noqa: E402
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation.color_registration import mask_circularity, mask_to_colored_points  # noqa: E402
from pose_estimation.frames import Frame  # noqa: E402


def fit_sphere(points):
    A = np.hstack([2 * points, np.ones((len(points), 1))])
    x, *_ = np.linalg.lstsq(A, (points ** 2).sum(1), rcond=None)
    c = x[:3]
    r = float(np.sqrt(max(x[3] + c @ c, 0.0)))
    return c, r, np.linalg.norm(points - c, axis=1) - r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="保存先の .pcd")
    ap.add_argument("--frames", type=int, default=15, help="平均・中央値を取るフレーム数")
    ap.add_argument("--voxel", type=float, default=0.0005, help="点群の間隔 [m]")
    ap.add_argument("--shell", type=float, default=0.003, help="球面からこれ以上離れた点を除く [m]")
    ap.add_argument("--method", default="hier_improved_v2_fast", choices=list(METHODS))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} は既に存在します．別の --out を指定してください．")

    rclpy.init()
    node = rclpy.create_node("capture_ball_model")
    got, frames = {}, []

    def on_pair(kind, m):
        got[kind] = m

    node.create_subscription(Image, "/camera/color/image_raw", lambda m: on_pair("c", m), 2)
    node.create_subscription(Image, "/camera/aligned_depth/image_raw", lambda m: on_pair("d", m), 2)
    node.create_subscription(CameraInfo, "/camera/color/camera_info", lambda m: on_pair("k", m), 2)
    print(f"カメラノードから {args.frames} フレーム受け取ります（ボールを動かさないでください）…", flush=True)
    t_end = time.time() + 60
    while len(frames) < args.frames and time.time() < t_end:
        got.pop("c", None)
        got.pop("d", None)
        while ("c" not in got or "d" not in got or "k" not in got) and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.5)
        if "c" in got and "d" in got:
            frames.append((image_from_msg(got["c"]).astype(np.float32), image_from_msg(got["d"])))
    if len(frames) < args.frames:
        sys.exit("カメラの画像が届きません．カメラノードが同じ ROS_DOMAIN_ID で動いているか確認してください．")
    k = got["k"]
    intr = dict(w=k.width, h=k.height, fx=k.k[0], fy=k.k[4], cx=k.k[2], cy=k.k[5])
    rgb = np.round(np.mean([f[0] for f in frames], axis=0)).astype(np.uint8)   # 色は平均
    depth = temporal_median_depth([f[1] for f in frames])                       # 深度は画素ごとの中央値

    print("セグメンテーション・深度補完…", flush=True)
    cfg = replace(METHODS[args.method], use_plane_removal=True, use_cpp=True)
    recon = MaskReconstructor(cfg)
    s = make_sample(rgb, depth, intr)
    masks = recon.reconstruct(s)
    if not masks:
        sys.exit("マスクが1つも得られませんでした．")
    circ = [mask_circularity(m) for m in masks]
    i = int(np.argmax(circ))
    print(f"マスク {len(masks)} 個，円形度の最大 {circ[i]:.2f}（2番目 {sorted(circ)[-2] if len(circ) > 1 else 0:.2f}）")
    if circ[i] < 0.85:
        sys.exit("円形度 0.85 以上のマスクがありません．ボールが視野に入っているか確認してください．")
    if sum(c >= 0.85 for c in circ) > 1:
        sys.exit("ボールらしいマスクが複数あります．ボールは1個だけ置いてください．")

    K = np.array([[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]])
    cp = mask_to_colored_points(masks[i], Frame(rgb=s.rgb, depth=recon.last_depth, K=K), PoseConfig(), args.voxel)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cp[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(cp[:, 3:6])
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    pts = np.asarray(pcd.points)
    c, r, res = fit_sphere(pts)
    keep = np.abs(res) < args.shell
    pcd = pcd.select_by_index(np.nonzero(keep)[0])
    c, r, res = fit_sphere(np.asarray(pcd.points))
    pcd.translate(-c)   # 原点を球の中心に
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    p = np.asarray(pcd.points)
    col = np.asarray(pcd.colors)
    print(f"保存: {out}（{len(p)} 点，球の半径 {r*1000:.1f} mm，球面からの誤差 RMS {np.sqrt(np.mean(res**2))*1000:.2f} mm，"
          f"広がり {np.round((p.max(0) - p.min(0)) * 1000, 1)} mm，色の範囲 {col.min():.2f}〜{col.max():.2f}，"
          f"撮影距離 {c[2]*1000:.0f} mm）")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
