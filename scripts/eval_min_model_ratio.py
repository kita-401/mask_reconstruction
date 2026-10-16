"""「間引き後の計測点数が全周モデルの点数の r 未満なら位置合わせしない」の閾値 r ごとの評価（10配置）．

    python scripts/eval_min_model_ratio.py --masks hier_improved_v2 gt

推定する部品の数と，その中の成功率（FSV<5%）・失敗率（FSV>20%）を比べる．
推定しなかった部品の成功率も出す（隠れていて一部しか見えない部品が多いほど低いはず）．
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation import PoseConfig  # noqa: E402
from pose_estimation.config import MASKS_DIR  # noqa: E402
from pose_estimation.evaluation import free_space_violation  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess, ratio_model_size  # noqa: E402
from pose_estimation.registration_gpu import GPUBatchRegistration  # noqa: E402

RATIOS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["hier_improved_v2", "gt"])
    args = ap.parse_args()
    pc = PoseConfig()
    reg, model = GPUBatchRegistration(pc), load_model(pc)
    n_full = ratio_model_size(pc.ratio_model_path, pc.voxel_size)
    print(f"全周モデルの点数（{pc.voxel_size * 1000:.0f}mm ボクセル後）: {n_full}（位置合わせは半分モデル {len(model)} 点）")
    for name in args.masks:
        ratio, fsv = [], []
        for i in range(1, 11):
            masks = load_masks(MASKS_DIR / name, i)
            fr = load_frame(i, masks[0].shape)
            pts = [preprocess(p, pc) for p in (mask_to_points(m, fr, pc) for m in masks) if len(p) >= pc.min_points]
            for p, r in zip(pts, reg.batch(model, pts)):
                ratio.append(len(p) / n_full)
                fsv.append(free_space_violation(model, r.T_model_to_cam, fr, pc.depth_scale))
        ratio, fsv = np.array(ratio), np.array(fsv)
        print(f"\n== {name}（{len(ratio)} 部品，点数比 中央値 {np.median(ratio):.2f}）")
        print("| 閾値 r | 推定する部品 | 成功（FSV<5%） | 失敗（FSV>20%） | 推定しない部品 | その成功率 |")
        print("|---|---|---|---|---|---|")
        for r in RATIOS:
            keep = ratio >= r
            skip_ok = f"{np.mean(fsv[~keep] < 0.05) * 100:.0f}%" if (~keep).any() else "–"
            print(f"| {r:.2f} | {keep.sum()}（{keep.mean() * 100:.0f}%） | {np.mean(fsv[keep] < 0.05) * 100:.1f}% | "
                  f"{np.mean(fsv[keep] > 0.2) * 100:.1f}% | {(~keep).sum()} | {skip_ok} |")


if __name__ == "__main__":
    main()
