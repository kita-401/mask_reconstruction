"""use_cpp=True のセグメンテーションが Python 版と同じマスクになることと，時間の比較（実データ 10 配置）．

    python tests/test_cpp_pipeline.py
    python tests/test_cpp_pipeline.py --methods hier_improved_v2_fast --plane

FastSAM・深度は手法ごとに1回だけ計算し，断片フィルタ＋統合の結果と時間を比べる．
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_masks import METHODS  # noqa: E402
from mask_reconstruction import MaskReconstructor, load_sample  # noqa: E402
from mask_reconstruction.depth import depth_to_points  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.merging import merge_fragments  # noqa: E402
from mask_reconstruction.pipeline import compute_depth  # noqa: E402


def run(cfg, s, frags, depth):
    pts, vi, vu, vv = depth_to_points(depth, s, cfg)
    t0 = time.perf_counter()
    kept = filter_fragments(frags, s.rgb, vu, vv, vi, cfg, pts=pts)
    masks = merge_fragments(kept, pts, vi, vu, vv, cfg, cam=(s.fx, s.fy, s.cx, s.cy))
    return masks, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["hier_improved_v2_fast", "hier_improved_v2", "proposed",
                                                     "baseline"])
    ap.add_argument("--plane", action="store_true", help="平面除去（use_plane_removal）も有効にする")
    args = ap.parse_args()
    recon = MaskReconstructor(METHODS["baseline"])
    for name in args.methods:
        cfg = replace(METHODS[name], use_plane_removal=args.plane)
        same, t_py, t_cpp = 0, [], []
        for i in range(1, 11):
            s = load_sample(i, cfg)
            frags = recon.fragments(s)
            depth = compute_depth(recon.lingbot, s, cfg, recon.device)
            run(cfg, s, frags, depth)   # 初回だけの処理（CAD の形状推定など）を時間に含めない
            a, ta = run(cfg, s, frags, depth)
            b, tb = run(replace(cfg, use_cpp=True), s, frags, depth)
            ok = len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
            same += ok
            if not ok:
                print(f"  {name} config {i}: 不一致（Python {len(a)} 個 / C++ {len(b)} 個）")
            t_py.append(ta)
            t_cpp.append(tb)
        print(f"{name}{'（平面除去あり）' if args.plane else ''}: マスク一致 {same}/10，"
              f"断片フィルタ＋統合 Python {np.mean(t_py) * 1000:.0f} ms → C++ {np.mean(t_cpp) * 1000:.0f} ms "
              f"（{np.mean(t_py) / np.mean(t_cpp):.1f} 倍）", flush=True)


if __name__ == "__main__":
    main()
