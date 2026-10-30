"""GPU 版の深度の前処理・点群化（gpu_ops.py）が CPU 版と同じ値を返すことの確認（実データ 10 配置）．

    python tests/test_gpu_ops.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mask_reconstruction import Config, load_sample  # noqa: E402
from mask_reconstruction import gpu_ops  # noqa: E402
from mask_reconstruction.depth import depth_to_points, remove_depth_outliers  # noqa: E402


def main():
    from live_pose_node import temporal_median_depth
    cfg = Config(intrinsics_ref_size=(848, 480), depth_refine=False)
    rng = np.random.default_rng(0)
    t = {"cpu": [0, 0, 0], "gpu": [0, 0, 0]}
    for i in range(1, 11):
        s = load_sample(i, cfg)
        # 時間方向の中央値：実データの深度に雑音と欠損を加えた 5 フレーム
        frames = [np.where(rng.random(s.depth.shape) < 0.1, 0,
                           np.clip(s.depth.astype(int) + rng.integers(-2, 3, s.depth.shape), 0, 65535)).astype(np.uint16)
                  for _ in range(5)]
        t0 = time.perf_counter(); a = temporal_median_depth(frames); t["cpu"][0] += time.perf_counter() - t0
        t0 = time.perf_counter(); b = gpu_ops.temporal_median_depth(frames); t["gpu"][0] += time.perf_counter() - t0
        assert np.array_equal(a, b), f"config {i}: 時間方向の中央値が違う"
        t0 = time.perf_counter(); a = remove_depth_outliers(s.depth, cfg); t["cpu"][1] += time.perf_counter() - t0
        t0 = time.perf_counter(); b = gpu_ops.remove_depth_outliers(s.depth, cfg); t["gpu"][1] += time.perf_counter() - t0
        assert np.array_equal(a, b), f"config {i}: 飛び値の除去が違う（{np.count_nonzero(a != b)} 画素）"
        t0 = time.perf_counter(); A = depth_to_points(a, s, cfg); t["cpu"][2] += time.perf_counter() - t0
        t0 = time.perf_counter(); B = gpu_ops.depth_to_points(a, s, cfg); t["gpu"][2] += time.perf_counter() - t0
        for name, x, y in zip(("pts", "valid_idx", "v_u", "v_v"), A, B):
            assert x.shape == y.shape and np.array_equal(x, y), f"config {i}: 点群化の {name} が違う"
        print(f"config {i}: 一致", flush=True)
    names = ("時間方向の中央値", "飛び値の除去", "点群化")
    for k, n in enumerate(names):
        print(f"{n}: CPU {t['cpu'][k] / 10 * 1000:.1f} ms / GPU {t['gpu'][k] / 10 * 1000:.1f} ms")


if __name__ == "__main__":
    main()
