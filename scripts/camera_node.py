"""RealSense D405 の画像を ROS 2 で配信するノード（推定とは別のプロセスで動かす）．

    python scripts/camera_node.py --cloud-hz 2 --cloud-stride 4

配信するもの（フレーム: camera_color_optical_frame，カラーと深度は同じ時刻印）
  /camera/color/image_raw          : カラー画像（bgr8）
  /camera/aligned_depth/image_raw  : カラーに位置合わせした深度（16UC1，mm）
  /camera/color/camera_info        : 内部パラメータ
  /camera/depth/points             : 色付き点群（表示用．間引いて低い頻度で配信）

推定ノード（live_pose_node.py）と別プロセスにすることで，画像・点群の配信が推定の処理時間を奪わないようにする．
"""
import argparse
import array

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

CAM_FRAME = "camera_color_optical_frame"
W, H, FPS = 848, 480, 10       # USB 2 接続でも使える設定（USB 3 なら 30fps も可）


def image_msg(img, stamp, frame_id=CAM_FRAME):
    """bgr8（3チャネル）または 16UC1（uint16）の画像メッセージ．"""
    m = Image()
    m.header = Header(stamp=stamp, frame_id=frame_id)
    m.height, m.width = img.shape[:2]
    if img.dtype == np.uint16:
        m.encoding, m.step = "16UC1", img.shape[1] * 2
    else:
        m.encoding, m.step = "bgr8", img.shape[1] * 3
    m.data = array.array("B", np.ascontiguousarray(img).tobytes())   # bytes だと要素ごとの検査で遅い
    return m


def image_from_msg(m):
    dtype, ch = (np.uint16, 1) if m.encoding == "16UC1" else (np.uint8, 3)
    a = np.frombuffer(m.data, dtype=dtype)
    return a.reshape(m.height, m.width) if ch == 1 else a.reshape(m.height, m.width, ch)


def cloud_msg(rgb, depth, intr, stamp, stride=4):
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


class CameraNode(Node):
    def __init__(self, args):
        super().__init__("d405_camera")
        self.args = args
        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 2)
        self.pub_depth = self.create_publisher(Image, "/camera/aligned_depth/image_raw", 2)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 2)
        self.pub_cloud = self.create_publisher(PointCloud2, "/camera/depth/points", 2)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = dict(w=i.width, h=i.height, fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy)
        self.info = CameraInfo(width=i.width, height=i.height, distortion_model="plumb_bob",
                               d=[0.0] * 5, k=[i.fx, 0.0, i.ppx, 0.0, i.fy, i.ppy, 0.0, 0.0, 1.0],
                               p=[i.fx, 0.0, i.ppx, 0.0, 0.0, i.fy, i.ppy, 0.0, 0.0, 0.0, 1.0, 0.0])
        for _ in range(30):            # 自動露出が落ち着くまで
            self.pipe.wait_for_frames()
        self.get_logger().info(f"カメラ: {self.intr}")
        self.last_cloud = 0.0

    def spin_once(self):
        f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return
        rgb, depth = np.asanyarray(c.get_data()), np.asanyarray(d.get_data())
        now = self.get_clock().now()
        stamp = now.to_msg()
        self.pub_color.publish(image_msg(rgb, stamp))
        self.pub_depth.publish(image_msg(depth, stamp))
        self.info.header = Header(stamp=stamp, frame_id=CAM_FRAME)
        self.pub_info.publish(self.info)
        t = now.nanoseconds * 1e-9
        if self.args.cloud_hz > 0 and t - self.last_cloud >= 1.0 / self.args.cloud_hz:
            self.last_cloud = t
            self.pub_cloud.publish(cloud_msg(rgb, depth, self.intr, stamp, self.args.cloud_stride))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-hz", type=float, default=2.0, help="表示用点群の配信頻度（0 で配信しない）")
    ap.add_argument("--cloud-stride", type=int, default=4, help="表示用点群の間引き [px]")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = CameraNode(args)
    try:
        while rclpy.ok():
            node.spin_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.pipe.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
