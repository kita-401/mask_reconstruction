"""姿勢推定用の点群化の GPU 版（pointcloud_gpu.py）が CPU 版（pointcloud.py）と同じ点の集合を返すことの確認．

    python tests/test_pose_gpu.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation import pointcloud_gpu as pg  # noqa: E402
from pose_estimation.config import MASKS_DIR  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import mask_to_points, preprocess  # noqa: E402


def main():
    pc = PoseConfig()
    # 1回目の GPU の準備時間を除くため，先に1回動かす
    fr0 = load_frame(1, load_masks(MASKS_DIR / "hier_improved_v2", 1)[0].shape)
    m0 = load_masks(MASKS_DIR / "hier_improved_v2", 1)[0]
    pg.preprocess(pg.mask_points(pg.to_gpu_mask(m0), pg.to_gpu_depth(fr0.depth), fr0.K, pc), pc)
    worst_raw, worst, n_parts, n_same, t_cpu, t_gpu = 0.0, 0.0, 0, 0, 0.0, 0.0
    t_batch, worst_b = 0.0, [0.0]
    for i in range(1, 11):
        masks = load_masks(MASKS_DIR / "hier_improved_v2", i)
        fr = load_frame(i, masks[0].shape)
        dt = pg.to_gpu_depth(fr.depth)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raws, outs = pg.batch_mask_points_preprocess(masks, fr.depth, fr.K, pc)
        torch.cuda.synchronize()
        t_batch += time.perf_counter() - t0
        for m, br, bo in zip(masks, raws, outs):
            a_raw0 = mask_to_points(m, fr, pc)
            assert np.array_equal(a_raw0, br)
            if bo is not None:
                a0 = preprocess(a_raw0, pc)
                from scipy.spatial import cKDTree
                assert a0.shape == bo.shape, ("まとめて処理", a0.shape, bo.shape)
                worst_b[0] = max(worst_b[0], float(cKDTree(bo).query(a0)[0].max()))
        for m in masks:
            t0 = time.perf_counter()
            a_raw = mask_to_points(m, fr, pc)
            a = preprocess(a_raw, pc) if len(a_raw) >= pc.min_points else None
            t_cpu += time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b_raw = pg.mask_points(pg.to_gpu_mask(m), dt, fr.K, pc)
            b = pg.preprocess(b_raw, pc) if len(b_raw) >= pc.min_points else None
            torch.cuda.synchronize()
            t_gpu += time.perf_counter() - t0
            b_raw = b_raw.cpu().numpy()
            assert a_raw.shape == b_raw.shape, (i, a_raw.shape, b_raw.shape)
            worst_raw = max(worst_raw, float(np.abs(a_raw - b_raw).max()) if len(a_raw) else 0.0)
            if a is None:
                continue
            b = b.cpu().numpy()
            n_parts += 1
            if a.shape == b.shape:
                n_same += 1
                from scipy.spatial import cKDTree
                worst = max(worst, float(cKDTree(b).query(a)[0].max()), float(cKDTree(a).query(b)[0].max()))
            else:
                print(f"  config {i}: 点数が違う CPU {len(a)} / GPU {len(b)}")
    print(f"マスク内の点：最大の差 {worst_raw:.1e} m")
    print(f"間引き＋外れ値除去：点数が一致 {n_same}/{n_parts} 部品，もう一方の最も近い点までの距離の最大 {worst:.1e} m")
    print(f"時間（全部品の合計／10配置）：CPU {t_cpu / 10 * 1000:.1f} ms，GPU 部品ごと {t_gpu / 10 * 1000:.1f} ms，"
          f"GPU まとめて {t_batch / 10 * 1000:.1f} ms（まとめて処理の差の最大 {worst_b[0]:.1e} m）")


if __name__ == "__main__":
    main()
