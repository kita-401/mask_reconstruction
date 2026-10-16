"""C++ の基本処理（cpp_kernels.py）が Python 版と同じ値を返すことの確認（実データ 10 配置）．

    python tests/test_cpp_kernels.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, load_sample  # noqa: E402
from mask_reconstruction import cpp_kernels as ck  # noqa: E402
from mask_reconstruction.convexity import organized_normals, point_image  # noqa: E402
from mask_reconstruction.depth import depth_to_points, remove_depth_outliers  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.graph_merging import _Region, boundary_normal_angle  # noqa: E402
from mask_reconstruction.merging import boundary_depth_gap, depth_image  # noqa: E402


def main():
    cfg = Config(intrinsics_ref_size=(848, 480), depth_refine=False)
    recon = MaskReconstructor(cfg)
    worst = dict(P=0, N=0, gap=0, ang=0)
    for i in range(1, 11):
        s = load_sample(i, cfg)
        pts, vi, vu, vv = depth_to_points(remove_depth_outliers(s.depth, cfg), s, cfg)
        masks = filter_fragments(recon.fragments(s), s.rgb, vu, vv, vi, cfg)
        shape = masks[0].shape
        P0, V0 = point_image(pts, vi, vu, vv, shape)
        P1, V1 = ck.point_image(pts, vi, vu, vv, shape)
        assert np.array_equal(V0, V1)
        worst["P"] = max(worst["P"], np.abs(P0 - P1).max())
        N0, ok0 = organized_normals(P0, V0, 9)
        N1, ok1 = ck.organized_normals(P1, V1, 9)
        assert np.array_equal(ok0, ok1)
        worst["N"] = max(worst["N"], np.abs(N0 - N1).max())
        stack = ck.stack_masks(masks)
        idx = ck.gather_indices(stack, ck.index_image(vi, vu, vv, shape))
        for m, ix in zip(masks, idx):
            assert np.array_equal(np.flatnonzero(m[vv, vu]), ix)
        kern = np.ones((cfg.adjacency_kernel,) * 2, np.uint8)
        regs = [_Region(m, kern) for m in masks]
        Z = depth_image(pts, vi, vu, vv, shape)
        adj, gap, ang, ov, area = ck.pair_features(stack, cfg.adjacency_kernel, Z, N0)
        for k in range(len(masks)):
            assert area[k] == masks[k].sum()
            for l in range(k + 1, len(masks)):
                a = bool(np.any(regs[k].dilated & masks[l]))
                assert a == adj[k, l], (i, k, l)
                assert ov[k, l] == np.sum(masks[k] & masks[l])
                if not a:
                    continue
                g = boundary_depth_gap(masks[k], masks[l], regs[k].dilated, regs[l].dilated, pts, vi, vu, vv)
                assert (g is None) == np.isnan(gap[k, l])
                if g is not None:
                    worst["gap"] = max(worst["gap"], abs(g - gap[k, l]))
                an = boundary_normal_angle(regs[k], regs[l], N0)
                assert (an is None) == np.isnan(ang[k, l]), (i, k, l, an, ang[k, l])
                if an is not None:
                    worst["ang"] = max(worst["ang"], abs(an - ang[k, l]))
        print(f"config {i}: {len(masks)} fragments ok", flush=True)
    print("最大の差:", {k: float(v) for k, v in worst.items()})


if __name__ == "__main__":
    main()
