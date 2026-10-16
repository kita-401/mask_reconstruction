"""実機（RealSense D405）で撮影 → セグメンテーション → 姿勢推定を繰り返し，結果を ROS 2 で配信する．

    bash scripts/run_live_pose.sh                 # RViz2 と一緒に起動（通常はこちら）
    python scripts/live_pose_node.py --method hier_improved_v2 --save

配信するもの（固定フレーム: camera_color_optical_frame）
  /tf                     : 各部品の姿勢 part_<k>（モデル座標 → カメラ）
  /camera/color/image_raw : カメラ画像（カメラの速さで配信）
  /camera/depth/points    : 色付き点群（間引き）
  /pose/overlay           : 推定結果（マスク輪郭・モデル点・座標軸）を重ねた画像（推定ごと）
  /pose/models            : 各部品のモデル点群（MarkerArray，part_<k> 座標系）

セグメンテーションは export_masks.py の METHODS の設定，姿勢推定は既存の SVD+ICP（カメラ側半分のモデル）．
既存コードは読み込んで使うだけで変更しない．
"""
import argparse
import array
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_masks import METHODS  # noqa: E402
from mask_reconstruction import MaskReconstructor  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.io import Sample, input_size  # noqa: E402
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.evaluation import free_space_violation  # noqa: E402
from pose_estimation.frames import Frame  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.visualization import _color, draw_poses  # noqa: E402

CAM_FRAME = "camera_color_optical_frame"
W, H, FPS = 848, 480, 10       # USB 2 接続でも使える設定（USB 3 なら 30fps も可）


class Camera:
    """D405 の RGB と，RGB に位置合わせした深度．最新のフレームを別スレッドで取り続ける．"""

    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = dict(w=intr.width, h=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy)
        self.lock = threading.Lock()
        self.latest = None
        for _ in range(30):            # 自動露出が落ち着くまで
            self.pipe.wait_for_frames()

    def grab(self):
        f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return None
        rgb, depth = np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy()
        with self.lock:
            self.latest = (rgb, depth)
        return rgb, depth

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.pipe.stop()


def make_sample(rgb, depth, intr, patch=14):
    """撮影画像を load_sample と同じ形（14 の倍数に縮小，内部パラメータも合わせる）にする．"""
    h_in, w_in = input_size(rgb.shape[:2], patch)
    sx, sy = w_in / intr["w"], h_in / intr["h"]
    return Sample(idx=0, rgb=cv2.resize(rgb, (w_in, h_in)),
                  depth=cv2.resize(depth, (w_in, h_in), interpolation=cv2.INTER_NEAREST), gt_masks=[],
                  fx=intr["fx"] * sx, fy=intr["fy"] * sy, cx=intr["cx"] * sx, cy=intr["cy"] * sy)


def image_msg(img, stamp, frame_id=CAM_FRAME):
    m = Image()
    m.header = Header(stamp=stamp, frame_id=frame_id)
    m.height, m.width = img.shape[:2]
    m.encoding = "bgr8"
    m.step = img.shape[1] * 3
    m.data = array.array("B", np.ascontiguousarray(img).tobytes())   # bytes だと要素ごとの検査で遅い
    return m


def cloud_msg(rgb, depth, intr, stamp, stride=3):
    """色付き点群（stride 画素ごとに間引き）．"""
    d = depth[::stride, ::stride].astype(np.float32) / 1000.0
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    ok = (d > 0.05) & (d < 1.5)
    z = d[ok]
    x = (u[ok] - intr["cx"]) * z / intr["fx"]
    y = (v[ok] - intr["cy"]) * z / intr["fy"]
    c = rgb[::stride, ::stride][ok].astype(np.uint32)
    packed = (c[:, 2] << 16) | (c[:, 1] << 8) | c[:, 0]
    arr = np.zeros(len(z), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
    arr["x"], arr["y"], arr["z"], arr["rgb"] = x, y, z, packed
    m = PointCloud2()
    m.header = Header(stamp=stamp, frame_id=CAM_FRAME)
    m.height, m.width = 1, len(arr)
    m.fields = [PointField(name=n, offset=o, datatype=t, count=1)
                for n, o, t in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("rgb", 12, 6))]
    m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 16, 16 * len(arr), True
    m.data = array.array("B", arr.tobytes())
    return m


class LivePoseNode(Node):
    def __init__(self, args):
        super().__init__("live_pose")
        self.args = args
        self.tf = TransformBroadcaster(self)
        self.pub_img = self.create_publisher(Image, "/camera/color/image_raw", 2)
        self.pub_cloud = self.create_publisher(PointCloud2, "/camera/depth/points", 2)
        self.pub_overlay = self.create_publisher(Image, "/pose/overlay", 2)
        self.pub_models = self.create_publisher(MarkerArray, "/pose/models", 2)
        self.poses = []                # [(name, T)]
        self.pose_lock = threading.Lock()
        self.create_timer(0.1, self.publish_tf)   # 最新の姿勢を 10Hz で配信し続ける（RViz で古くならないように）

        self.get_logger().info("RealSense を起動しています…")
        self.cam = Camera()
        self.get_logger().info(f"カメラ: {self.cam.intr}")
        self.get_logger().info(f"セグメンテーション（{args.method}）と姿勢推定のモデルを読み込んでいます…")
        cfg = METHODS[args.method]
        if not args.no_plane_removal:   # 黒いトレイ以外の背景（机など）を平面として除く
            from dataclasses import replace
            cfg = replace(cfg, use_plane_removal=True)
        self.recon = MaskReconstructor(cfg)
        self.pc = PoseConfig()
        self.reg = SVDICPRegistration(self.pc)
        self.model = load_model(self.pc)
        self.model_vis = self.model[::3]
        self.save_dir = None
        if args.save:
            self.save_dir = RESULTS_DIR / "live" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            with open(self.save_dir / "camera_intrinsics.json", "w") as f:
                json.dump(self.cam.intr, f)
        self.running = True
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()

    def camera_loop(self):
        while self.running:
            got = self.cam.grab()
            if got is None:
                continue
            stamp = self.get_clock().now().to_msg()
            self.pub_img.publish(image_msg(got[0], stamp))
            self.pub_cloud.publish(cloud_msg(got[0], got[1], self.cam.intr, stamp))

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
        got = self.cam.get()
        if got is None:
            return False
        rgb, depth = got
        t0 = time.perf_counter()
        s = make_sample(rgb, depth, self.cam.intr)
        masks = self.recon.reconstruct(s)
        t_seg = time.perf_counter() - t0
        K = np.array([[s.fx, 0, s.cx], [0, s.fy, s.cy], [0, 0, 1]])
        frame = Frame(rgb=s.rgb, depth=s.depth, K=K)
        poses, rows = [], []
        t1 = time.perf_counter()
        for i, m in enumerate(masks):
            p = mask_to_points(m, frame, self.pc)
            if len(p) < self.pc.min_points:
                poses.append(None)
                continue
            r = self.reg(self.model, preprocess(p, self.pc))
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
                               f"（セグメンテーション {t_seg*1000:.0f}ms，姿勢推定 {t_pose*1000:.0f}ms）")
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

    def destroy_node(self):
        self.running = False
        self.cam_thread.join(timeout=2.0)   # wait_for_frames の途中で止めないように
        self.cam.stop()
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hier_improved_v2", choices=list(METHODS))
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
