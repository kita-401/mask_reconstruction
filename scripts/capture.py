"""RealSense D405 で RGB・深度（カラーに位置合わせ）・内部パラメータを1枚撮影する．

    python scripts/capture.py --index 11
    → data/real_rgb_11.png, data/real_depth_11.png, data/camera_intrinsics_11.json
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


def capture(index, warmup=30):
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
    capture(ap.parse_args().index)
