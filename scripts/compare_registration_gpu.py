"""姿勢推定の位置合わせ：CPU 版（Open3D ICP，1部品ずつ）と GPU 版（torch，全部品まとめて）の比較．

    python scripts/compare_registration_gpu.py --masks hier_improved_v2 gt

比べるもの：両者の姿勢の差（回転 [deg]・並進 [mm]・ADD-S [mm]），自由空間違反率（FSV）による成功率，時間．
時間は前処理（マスク → 点群）を除いた位置合わせだけを，1フレームあたりで測る．
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.config import MASKS_DIR  # noqa: E402
from pose_estimation.evaluation import add_s, free_space_violation  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.registration_gpu import GPUBatchRegistration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["hier_improved_v2", "gt"])
    ap.add_argument("--repeat", type=int, default=3, help="時間は各フレームこの回数の中央値")
    args = ap.parse_args()
    pc = PoseConfig()
    cpu, gpu = SVDICPRegistration(pc), GPUBatchRegistration(pc)
    model = load_model(pc)
    gpu.batch(model, [model[:500]])   # 初回の CUDA 初期化を時間に含めない
    for name in args.masks:
        rows, t_cpu, t_gpu = [], [], []
        for i in range(1, 11):
            masks = load_masks(MASKS_DIR / name, i)
            fr = load_frame(i, masks[0].shape)
            pts = [p for p in (mask_to_points(m, fr, pc) for m in masks) if len(p) >= pc.min_points]
            pts = [preprocess(p, pc) for p in pts]
            tc, tg = [], []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                rc = [cpu(model, p) for p in pts]
                tc.append(time.perf_counter() - t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rg = gpu.batch(model, pts)
                torch.cuda.synchronize()
                tg.append(time.perf_counter() - t0)
            t_cpu.append(np.median(tc))
            t_gpu.append(np.median(tg))
            for a, b in zip(rc, rg):
                Ta, Tb = a.T_model_to_cam, b.T_model_to_cam
                ang = np.degrees(np.arccos(np.clip((np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2, -1, 1)))
                rows.append(dict(rot=ang, trans=np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000,
                                 adds=add_s(model, Ta, Tb) * 1000, same_flip=a.flip == b.flip,
                                 fsv_c=free_space_violation(model, Ta, fr, pc.depth_scale),
                                 fsv_g=free_space_violation(model, Tb, fr, pc.depth_scale),
                                 fit_c=a.icp_fitness, fit_g=b.icp_fitness))
        r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
        print(f"\n== {name}（{len(rows)} 部品）")
        print(f"姿勢の差（CPU と GPU）: 回転 中央値 {np.median(r['rot']):.3f}° / 95% {np.percentile(r['rot'], 95):.2f}° / "
              f"最大 {r['rot'].max():.1f}°，並進 中央値 {np.median(r['trans']):.3f} mm，"
              f"ADD-S<1mm {np.mean(r['adds'] < 1) * 100:.0f}%，同じ初期候補 {np.mean(r['same_flip']) * 100:.0f}%")
        print(f"成功（FSV<5%）: CPU {np.mean(r['fsv_c'] < 0.05) * 100:.1f}% / GPU {np.mean(r['fsv_g'] < 0.05) * 100:.1f}%，"
              f"失敗（FSV>20%）: CPU {np.mean(r['fsv_c'] > 0.2) * 100:.1f}% / GPU {np.mean(r['fsv_g'] > 0.2) * 100:.1f}%，"
              f"ICP fitness: CPU {r['fit_c'].mean():.3f} / GPU {r['fit_g'].mean():.3f}")
        print(f"時間/フレーム（位置合わせのみ）: CPU {np.mean(t_cpu) * 1000:.0f} ms / GPU {np.mean(t_gpu) * 1000:.0f} ms "
              f"（{np.mean(t_cpu) / np.mean(t_gpu):.1f} 倍）")


if __name__ == "__main__":
    main()
