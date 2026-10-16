"""実機（RealSense D405）の画像でセグメンテーション → 姿勢推定を繰り返し，結果を ROS 2 で配信する．

    bash scripts/run_live_pose.sh                 # カメラノード・RViz2 と一緒に起動（通常はこちら）
    python scripts/live_pose_node.py --method hier_improved_v2_fast

カメラは別プロセスの camera_node.py が開き，このノードは画像を受け取るだけ
（画像・点群の配信が推定の処理時間を奪わないように）．
受け取るもの: /camera/color/image_raw, /camera/aligned_depth/image_raw（同じ時刻印の組）, /camera/color/camera_info
配信するもの（固定フレーム: camera_color_optical_frame）
  /tf                     : 各部品の姿勢 part_<k>（モデル座標 → カメラ）
  /pose/overlay           : 推定結果（マスク輪郭・モデル点・座標軸）を重ねた画像（推定ごと）
  /pose/models            : 各部品のモデル点群（MarkerArray，part_<k> 座標系）

セグメンテーションは export_masks.py の METHODS の設定，姿勢推定は既存の SVD+ICP（カメラ側半分のモデル）．
既存コードは読み込んで使うだけで変更しない．
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_node import CAM_FRAME, image_from_msg, image_msg  # noqa: E402
from export_masks import METHODS  # noqa: E402
from mask_reconstruction import MaskReconstructor  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.io import Sample, input_size  # noqa: E402
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.evaluation import free_space_violation  # noqa: E402
from pose_estimation.frames import Frame  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.registration_gpu import GPUBatchRegistration  # noqa: E402
from pose_estimation.visualization import _color, draw_poses  # noqa: E402

def make_sample(rgb, depth, intr, patch=14):
    """撮影画像を load_sample と同じ形（14 の倍数に縮小，内部パラメータも合わせる）にする．"""
    h_in, w_in = input_size(rgb.shape[:2], patch)
    sx, sy = w_in / intr["w"], h_in / intr["h"]
    return Sample(idx=0, rgb=cv2.resize(rgb, (w_in, h_in)),
                  depth=cv2.resize(depth, (w_in, h_in), interpolation=cv2.INTER_NEAREST), gt_masks=[],
                  fx=intr["fx"] * sx, fy=intr["fy"] * sy, cx=intr["cx"] * sx, cy=intr["cy"] * sy)


class LivePoseNode(Node):
    def __init__(self, args):
        super().__init__("live_pose")
        self.args = args
        self.tf = TransformBroadcaster(self)
        self.pub_overlay = self.create_publisher(Image, "/pose/overlay", 2)
        self.pub_models = self.create_publisher(MarkerArray, "/pose/models", 2)
        self.poses = []                # [(name, T)]
        self.pose_lock = threading.Lock()
        self.create_timer(0.1, self.publish_tf)   # 最新の姿勢を 10Hz で配信し続ける（RViz で古くならないように）

        # カメラノードからの画像（カラーと深度を時刻印で組にし，最新の組だけ持つ）
        self.frame_lock = threading.Lock()
        self.latest, self.intr = None, None
        self._pending = {}
        self.create_subscription(Image, "/camera/color/image_raw", lambda m: self.on_image("color", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/aligned_depth/image_raw", lambda m: self.on_image("depth", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, qos_profile_sensor_data)
        self.get_logger().info(f"セグメンテーション（{args.method}）と姿勢推定のモデルを読み込んでいます…")
        from dataclasses import replace
        cfg = METHODS[args.method]
        if not args.no_plane_removal:   # 黒いトレイ以外の背景（机など）を平面として除く
            cfg = replace(cfg, use_plane_removal=True)
        if not args.no_cpp:             # 重い基本処理を C++ で（結果は Python 版と同じ）
            cfg = replace(cfg, use_cpp=True)
        self.recon = MaskReconstructor(cfg)
        self.pc = PoseConfig()
        # 位置合わせ：既定は GPU で全部品まとめて（registration_gpu.py）．--cpu-registration で従来の Open3D 版
        self.reg = SVDICPRegistration(self.pc) if args.cpu_registration else GPUBatchRegistration(self.pc)
        self.model = load_model(self.pc)
        self.model_vis = self.model[::3]
        self.save_dir = None
        if args.save:
            self.save_dir = RESULTS_DIR / "live" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info("カメラノードからの画像を待っています…")

    def on_info(self, m):
        if self.intr is None:
            self.intr = dict(w=m.width, h=m.height, fx=m.k[0], fy=m.k[4], cx=m.k[2], cy=m.k[5])
            self.get_logger().info(f"カメラ: {self.intr}")
            if self.save_dir is not None:
                with open(self.save_dir / "camera_intrinsics.json", "w") as f:
                    json.dump(self.intr, f)

    def on_image(self, kind, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        with self.frame_lock:
            self._pending.setdefault(key, {})[kind] = m
            pair = self._pending[key]
            if "color" in pair and "depth" in pair:
                self.latest = (pair["color"], pair["depth"])
                self._pending = {k: v for k, v in self._pending.items() if k > key}
            elif len(self._pending) > 10:
                self._pending = dict(sorted(self._pending.items())[-5:])

    def publish_tf(self):
        stamp = self.get_clock().now().to_msg()
        with self.pose_lock:
            poses = list(self.poses)
        msgs = []
        for name, T in poses:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, CAM_FRAME, name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, T[:3, 3])
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
            msgs.append(t)
        if msgs:
            self.tf.sendTransform(msgs)

    def estimate_once(self, k):
        """1回推定する．まだカメラ画像がなければ False．"""
        with self.frame_lock:
            got, self.latest = self.latest, None   # 同じ画像で2回推定しない
        if got is None or self.intr is None:
            return False
        rgb, depth = image_from_msg(got[0]), image_from_msg(got[1])
        t0 = time.perf_counter()
        s = make_sample(rgb, depth, self.intr)
        masks = self.recon.reconstruct(s)
        t_seg = time.perf_counter() - t0
        K = np.array([[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]])
        frame = Frame(rgb=s.rgb, depth=s.depth, K=K)
        poses, rows = [], []
        t1 = time.perf_counter()
        clouds = {}
        for i, m in enumerate(masks):
            p = mask_to_points(m, frame, self.pc)
            if len(p) >= self.pc.min_points:
                clouds[i] = preprocess(p, self.pc)
        t_prep = time.perf_counter() - t1
        t2 = time.perf_counter()
        if isinstance(self.reg, GPUBatchRegistration):
            results = dict(zip(clouds, self.reg.batch(self.model, list(clouds.values()))))
        else:
            results = {i: self.reg(self.model, p) for i, p in clouds.items()}
        t_reg = time.perf_counter() - t2
        for i in range(len(masks)):
            r = results.get(i)
            if r is None:
                poses.append(None)
                continue
            T = r.T_model_to_cam
            poses.append(T)
            rows.append(dict(id=i, T=T.tolist(), icp_fitness=r.icp_fitness,
                             fsv=free_space_violation(self.model, T, frame, self.pc.depth_scale)))
        t_pose = time.perf_counter() - t1
        named = [(f"part_{i}", T) for i, T in enumerate(poses) if T is not None]
        with self.pose_lock:
            self.poses = named
        self.publish_tf()
        stamp = self.get_clock().now().to_msg()
        overlay = draw_poses(frame, self.model, poses, masks)
        cv2.putText(overlay, f"#{k}  parts {len(named)}  seg {t_seg*1000:.0f}ms  pose {t_pose*1000:.0f}ms",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.pub_overlay.publish(image_msg(overlay, stamp))
        self.publish_models(named, len(poses), stamp)
        self.get_logger().info(f"#{k}: マスク {len(masks)} 個，姿勢 {len(named)} 個 "
                               f"（セグメンテーション {t_seg*1000:.0f}ms，姿勢推定 {t_pose*1000:.0f}ms "
                               f"= 点群化 {t_prep*1000:.0f} + 位置合わせ {t_reg*1000:.0f} + 評価）")
        if self.save_dir is not None:
            d = self.save_dir / f"frame{k:04d}"
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / "rgb.png"), rgb)
            cv2.imwrite(str(d / "depth.png"), depth)
            cv2.imwrite(str(d / "overlay.png"), overlay)
            # 圧縮して保存（非圧縮だと 1 フレーム約 5MB になり，連続運転でディスクを使い切る）
            np.savez_compressed(d / "masks.npz",
                                masks=np.array(masks, dtype=bool) if masks else np.zeros((0, *s.shape), bool))
            with open(d / "poses.json", "w") as f:
                json.dump(dict(seg_ms=t_seg * 1000, pose_ms=t_pose * 1000, parts=rows), f, indent=1)
        return True

    def publish_models(self, named, n, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for name, _ in named:
            i = int(name.split("_")[1])
            b, g, r = _color(i, n)
            mk = Marker()
            mk.header = Header(stamp=stamp, frame_id=name)
            mk.ns, mk.id, mk.type, mk.action = "models", i, Marker.POINTS, Marker.ADD
            mk.scale.x = mk.scale.y = 0.002
            mk.color = ColorRGBA(r=r / 255, g=g / 255, b=b / 255, a=0.9)
            mk.pose.orientation.w = 1.0
            mk.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in self.model_vis]
            arr.markers.append(mk)
        self.pub_models.publish(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hier_improved_v2_fast", choices=list(METHODS))
    ap.add_argument("--cpu-registration", action="store_true", help="位置合わせを従来の CPU（Open3D）版で行う")
    ap.add_argument("--no-cpp", action="store_true", help="C++ の基本処理を使わず Python 版で計算する")
    ap.add_argument("--no-plane-removal", action="store_true", help="平面除去による背景断片の除去をしない")
    ap.add_argument("--save", action="store_true", help="推定ごとに RGB・深度・マスク・姿勢を results/live/ に保存")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 ならこの回数推定して終了（動作確認用）")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = LivePoseNode(args)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    k = 0
    try:
        while rclpy.ok() and (args.max_frames <= 0 or k < args.max_frames):
            if node.estimate_once(k):
                k += 1
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
