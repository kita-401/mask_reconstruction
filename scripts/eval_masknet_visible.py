"""MaskNet で削ったモデルで初期姿勢を求め，ICP は全周モデルの「見える面だけ」で行う組み合わせの評価（10配置）．

    python scripts/eval_masknet_visible.py --masks gt hier_improved_v2

比べるもの（計測点群・モデルとも 2mm ボクセル）
  半分（従来）          : カメラ側半分のモデルで SVD+ICP
  全周＋見える面        : 全周モデルで SVD の初期候補 → 見える面だけの ICP（icp_visible_only）
  MaskNet→全周＋見える面: MaskNet で削ったモデルで SVD（符号探索）+ ICP → その姿勢から全周モデルの見える面だけの ICP
  良い方を選ぶ          : 上の2つのうち，最後の ICP の fitness が高い方を部品ごとに選ぶ
評価は eval_full_model.py と同じ（全周モデルのうちカメラを向く面の点で FSV，<5% 成功，>20% 失敗）．
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from masksvd_exp.evaluation import fitness, free_space_violation  # noqa: E402
from masksvd_exp.masksvd import MaskSVD, MaskSVDConfig  # noqa: E402
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation.config import MASKS_DIR, PKG_DIR  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.registration_gpu import GPUBatchRegistration  # noqa: E402

FULL = str(PKG_DIR / "models" / "t_pipe_full.pcd")
HALF = str(PKG_DIR / "models" / "t_pipe_half.pcd")
BASE = PoseConfig()
VOXEL = BASE.voxel_size
# masksvd_exp で最も良かった設定（中心化＋半径1に拡大＋符号探索＋ICP 1cm），モデルは計測と同じボクセル
MASKNET = replace(MaskSVDConfig(), normalize_scale=True, center_inputs=True, flip_mode="search",
                  icp_threshold=0.01, model_voxel=VOXEL)
NAMES = ("半分（従来）", "全周＋見える面", "MaskNet→全周＋見える面", "良い方を選ぶ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["gt", "hier_improved_v2"])
    args = ap.parse_args()
    half_cfg = replace(BASE, model_path=HALF)
    vis_cfg = replace(BASE, model_path=FULL, icp_visible_only=True)
    reg_half, half = GPUBatchRegistration(half_cfg), load_model(half_cfg)
    reg_vis, full = GPUBatchRegistration(vis_cfg), load_model(vis_cfg)
    masknet = MaskSVD(MASKNET)
    for name in args.masks:
        res = {k: dict(fsv=[], fit=[], t=[]) for k in NAMES}
        for i in range(1, 11):
            masks = load_masks(MASKS_DIR / name, i)
            fr = load_frame(i, masks[0].shape)
            depth_m = fr.depth.astype(np.float64) / 1000.0
            pts = [preprocess(p, BASE) for p in (mask_to_points(m, fr, BASE) for m in masks)
                   if len(p) >= BASE.min_points]
            out, times = {}, {}

            def timed(key, fn):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out[key] = fn()
                torch.cuda.synchronize()
                times[key] = time.perf_counter() - t0

            timed(NAMES[0], lambda: reg_half.batch(half, pts))
            timed(NAMES[1], lambda: reg_vis.batch(full, pts))
            timed(NAMES[2], lambda: reg_vis.batch(full, pts, init=[masknet.register(p) for p in pts]))
            out[NAMES[3]] = [a if a.icp_fitness >= b.icp_fitness else b for a, b in zip(out[NAMES[1]], out[NAMES[2]])]
            times[NAMES[3]] = times[NAMES[1]] + times[NAMES[2]]
            for k in NAMES:
                res[k]["t"].append(times[k])
                for p, r in zip(pts, out[k]):
                    res[k]["fsv"].append(free_space_violation(FULL, r.T_model_to_cam, depth_m, fr.K))
                    res[k]["fit"].append(fitness(FULL, r.T_model_to_cam, p))
            print(f"  {name} config {i} done", flush=True)
        print(f"\n== {name}（{len(res[NAMES[0]]['fsv'])} 部品）")
        print("| 方法 | 成功（FSV<5%） | 失敗（FSV>20%） | fitness | 時間 [ms/フレーム] |")
        print("|---|---|---|---|---|")
        for k in NAMES:
            fsv = np.array(res[k]["fsv"])
            print(f"| {k} | {np.nanmean(fsv < 0.05) * 100:.1f}% | {np.nanmean(fsv > 0.2) * 100:.1f}% | "
                  f"{np.mean(res[k]['fit']):.3f} | {np.mean(res[k]['t']) * 1000:.0f} |")


if __name__ == "__main__":
    main()
