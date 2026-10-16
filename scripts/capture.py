"""RealSense D405 で RGB・深度（カラーに位置合わせ）・内部パラメータを1枚撮影する．

    python scripts/capture.py --index 11
    → data/real_rgb_11.png, data/real_depth_11.png, data/camera_intrinsics_11.json
    python scripts/capture.py --index 11 --frames 5
    → 上記に加えて，時間メディアン用の深度 data/real_depth_11_t0.png 〜 _t4.png（静止シーン）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction.io import sample_paths  # noqa: E402


def capture(index, warmup=30, n_frames=1):
    paths = sample_paths(index)
    if paths["rgb"].exists():
        sys.exit(f"{paths['rgb']} は既に存在します．別の --index を指定してください．")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    pipeline.start(config)
    print("Pipeline started. Waiting for auto-exposure to stabilize...")

    try:
        for _ in range(warmup):
            pipeline.wait_for_frames()
            time.sleep(0.05)

        frames = rs.align(rs.stream.color).process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            sys.exit("Failed to capture")

        cv2.imwrite(str(paths["rgb"]), np.asanyarray(color_frame.get_data()))
        cv2.imwrite(str(paths["depth"]), np.asanyarray(depth_frame.get_data()))

        if n_frames > 1:
            align = rs.align(rs.stream.color)
            for k in range(n_frames):
                f = align.process(pipeline.wait_for_frames()).get_depth_frame()
                cv2.imwrite(str(paths["depth"].with_name(f"{paths['depth'].stem}_t{k}.png")),
                            np.asanyarray(f.get_data()))

        intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        with open(paths["intrinsics"], "w") as f:
            json.dump({"w": intr.width, "h": intr.height, "fx": intr.fx, "fy": intr.fy,
                       "cx": intr.ppx, "cy": intr.ppy}, f)
        print(f"Saved: {paths['rgb'].name}, {paths['depth'].name}, {paths['intrinsics'].name}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--frames", type=int, default=1, help="時間メディアン用に保存する深度フレーム数")
    args = ap.parse_args()
    capture(args.index, n_frames=args.frames)
