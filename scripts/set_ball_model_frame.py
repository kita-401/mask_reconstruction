"""球全体のボールのモデル（YCB）の軸の向きを，実物の置き方を基準に決め直す．

ボールを基準にしたい向きで置いて実行すると，そのときの推定姿勢が「軸がカメラの座標系と揃った向き」
（x：画像の右，y：画像の下，z：カメラの奥）になるようにモデルを回転して保存する．原点（球の中心）は動かさない
（球全体の位置合わせはモデルの原点が球の中心であることを前提にしているため）．

カメラノード（scripts/camera_node.py）を先に起動しておく：
    ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1 python scripts/set_ball_model_frame.py
    → pose_estimation/models/ball_ycb_ref.pcd

縫い目の対称性で同じ縫い目に見える姿勢が複数あるので，各フレームの推定を最初のフレームに最も近い対称な姿勢に揃えてから平均する．
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_node import image_from_msg  # noqa: E402
from export_masks import METHODS  # noqa: E402
from live_pose_node import make_sample  # noqa: E402
from mask_reconstruction import MaskReconstructor  # noqa: E402
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation.color_registration import CIRCULARITY_MIN, load_colored_model, mask_circularity, mask_to_colored_points  # noqa: E402
from pose_estimation.frames import Frame  # noqa: E402
from pose_estimation.sphere_seam_registration import SphereSeamRegistration  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "pose_estimation/models/ball_ycb.pcd"))
    ap.add_argument("--out", default=str(ROOT / "pose_estimation/models/ball_ycb_ref.pcd"))
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--lingbot-tokens", type=int, default=2400)
    ap.add_argument("--voxel", type=float, default=0.0003)
    args = ap.parse_args()

    model_pcd = o3d.io.read_point_cloud(args.src)
    model = load_colored_model(args.src, 0.0005)
    reg = SphereSeamRegistration(model)
    syms = [np.eye(3)] + [S[:3, :3] for S in reg.seam_symmetries()]
    cfg = replace(METHODS["hier_improved_v2_fast"], use_plane_removal=True, use_cpp=True, lingbot_tokens=args.lingbot_tokens)
    recon = MaskReconstructor(cfg)
    pc = PoseConfig()

    rclpy.init()
    node = rclpy.create_node("set_ball_model_frame")
    got = {}
    node.create_subscription(Image, "/camera/color/image_raw", lambda m: got.__setitem__("c", m), 2)
    node.create_subscription(Image, "/camera/aligned_depth/image_raw", lambda m: got.__setitem__("d", m), 2)
    node.create_subscription(CameraInfo, "/camera/color/camera_info", lambda m: got.__setitem__("k", m), 2)
    Rs, cs, scores = [], [], []
    print(f"{args.frames} フレーム推定します（ボールを動かさないでください）…", flush=True)
    for fi in range(args.frames):
        got.pop("c", None)
        got.pop("d", None)
        while len(got) < 3:
            rclpy.spin_once(node, timeout_sec=1)
        k = got["k"]
        intr = dict(w=k.width, h=k.height, fx=k.k[0], fy=k.k[4], cx=k.k[2], cy=k.k[5])
        s = make_sample(image_from_msg(got["c"]), image_from_msg(got["d"]), intr)
        masks = [m for m in recon.reconstruct(s) if mask_circularity(m) >= CIRCULARITY_MIN]
        if not masks:
            print(f"  frame {fi}: ボールが見つかりません")
            continue
        h, w = s.shape

        def dist_center(m):
            ys, xs = np.nonzero(m)
            return np.hypot(xs.mean() - w / 2, ys.mean() - h / 2)
        m = min(masks, key=dist_center)   # 画像の中央に最も近いボール
        K = np.array([[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]])
        T = reg(mask_to_colored_points(m, Frame(rgb=s.rgb, depth=recon.last_depth, K=K), pc, args.voxel))
        if T is None:
            print(f"  frame {fi}: 縫い目が見えません")
            continue
        R = T[:3, :3]
        if Rs:   # 最初のフレームに最も近い対称な姿勢に揃える（R と R @ S は同じ縫い目）
            R = min((R @ S for S in syms), key=lambda C: Rotation.from_matrix(C @ Rs[0].T).magnitude())
        Rs.append(R)
        cs.append(T[:3, 3])
        scores.append(reg.last_score)
        print(f"  frame {fi}: 一致度 {reg.last_score:.2f}，最初のフレームとの差 "
              f"{np.rad2deg(Rotation.from_matrix(R @ Rs[0].T).magnitude()):.1f}°", flush=True)
    if len(Rs) < 3:
        sys.exit("推定できたフレームが少なすぎます．")
    # 外れたフレーム（別の縫い目の合わせ方になったもの）に引っ張られないよう，他との差の合計が最小のフレーム（medoid）を基準に
    # 対称な姿勢を揃え直し，基準から 25° 以内のフレームだけを平均する
    def ang(A, B):
        return np.rad2deg(Rotation.from_matrix(A @ B.T).magnitude())
    med = Rs[int(np.argmin([sum(ang(A, B) for B in Rs) for A in Rs]))]
    Rs = [min((R @ S for S in syms), key=lambda C: ang(C, med)) for R in Rs]
    inl = [R for R in Rs if ang(R, med) < 25.0]
    print(f"基準（medoid）から 25° 以内のフレーム {len(inl)}/{len(Rs)} を平均します")
    R_mean = Rotation.from_matrix(np.array(inl)).mean().as_matrix()
    dev = [ang(R, R_mean) for R in inl]
    # 今の置き方で推定姿勢が単位行列（カメラの座標系と揃う）になるよう，モデルを M = R_mean で回す
    # （元のモデル座標 x に対して，新しいモデル座標は R_mean x．カメラ座標は R_mean x + c で変わらない）
    pts = np.asarray(model_pcd.points) @ R_mean.T
    model_pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(args.out, model_pcd)
    print(f"保存: {args.out}（{len(inl)} フレームの平均，平均からのずれ 中央値 {np.median(dev):.1f}° 最大 {np.max(dev):.1f}°，"
          f"一致度の中央値 {np.median(scores):.2f}）")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
