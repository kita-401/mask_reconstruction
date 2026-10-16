"""位置合わせのモデルの比較：カメラ側半分 / 全周 / 全周＋見える面だけ（icp_visible_only）．10配置．

    python scripts/eval_full_model.py --masks gt hier_improved_v2

評価はどの手法も同じ基準（masksvd_exp と同じ）：全周モデルのうち推定姿勢でカメラを向く面の点で
自由空間違反率（FSV）を求め，<5% を成功，>20% を失敗とする．fitness は計測点のうちその面から 3mm 以内の割合．
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from masksvd_exp.evaluation import fitness, free_space_violation  # noqa: E402
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation.config import MASKS_DIR, PKG_DIR  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.registration_gpu import GPUBatchRegistration  # noqa: E402

FULL = str(PKG_DIR / "models" / "t_pipe_full.pcd")
HALF = str(PKG_DIR / "models" / "t_pipe_half.pcd")
BASE = PoseConfig()
METHODS = {
    "カメラ側半分（従来）": replace(BASE, model_path=HALF),
    "全周": replace(BASE, model_path=FULL),
    "全周＋見える面だけ": replace(BASE, model_path=FULL, icp_visible_only=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["gt", "hier_improved_v2"])
    args = ap.parse_args()
    half, full = load_model(METHODS["カメラ側半分（従来）"]), load_model(METHODS["全周"])
    d = cKDTree(full).query(half)[0]
    print(f"半分モデルと全周モデルの座標系の確認：半分の各点から全周への距離 中央値 {np.median(d) * 1000:.2f} mm")
    regs = {k: (GPUBatchRegistration(c), load_model(c)) for k, c in METHODS.items()}
    for name in args.masks:
        res = {k: dict(fsv=[], fit=[], t=[]) for k in METHODS}
        for i in range(1, 11):
            masks = load_masks(MASKS_DIR / name, i)
            fr = load_frame(i, masks[0].shape)
            depth_m = fr.depth.astype(np.float64) / 1000.0
            pts = [preprocess(p, BASE) for p in (mask_to_points(m, fr, BASE) for m in masks)
                   if len(p) >= BASE.min_points]
            for k, (reg, model) in regs.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = reg.batch(model, pts)
                torch.cuda.synchronize()
                res[k]["t"].append(time.perf_counter() - t0)
                for p, r in zip(pts, out):
                    res[k]["fsv"].append(free_space_violation(FULL, r.T_model_to_cam, depth_m, fr.K))
                    res[k]["fit"].append(fitness(FULL, r.T_model_to_cam, p))
        print(f"\n== {name}（{len(res['全周']['fsv'])} 部品）")
        print("| モデル | 成功（FSV<5%） | 失敗（FSV>20%） | fitness | 位置合わせ [ms/フレーム] |")
        print("|---|---|---|---|---|")
        for k, r in res.items():
            fsv = np.array(r["fsv"])
            print(f"| {k} | {np.nanmean(fsv < 0.05) * 100:.1f}% | {np.nanmean(fsv > 0.2) * 100:.1f}% | "
                  f"{np.mean(r['fit']):.3f} | {np.mean(r['t']) * 1000:.0f} |")


if __name__ == "__main__":
    main()
