"""正解マスクの点群で，MaskSVD と既存の SVD+ICP の姿勢推定を比べる．

    python scripts/eval_masksvd_pose.py

成功：自由空間違反率 < 5%（カメラ側の面が，写っている物より手前に浮いていない）
既存コードは読み込みにだけ使う．
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from masksvd_exp.evaluation import fitness, free_space_violation  # noqa: E402
from masksvd_exp.masksvd import MASKSVD_DIR, MaskSVD, MaskSVDConfig  # noqa: E402
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.config import MASKS_DIR  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402

BASE = MaskSVDConfig()
CK = MASKSVD_DIR / "checkpoint"
CK = MASKSVD_DIR / "checkpoint"
N = dict(normalize_scale=True, center_inputs=True)
VARIANTS = {
    "MaskSVD 中心化（拡大なし）": replace(BASE, center_inputs=True),
    "MaskSVD 拡大 plane": replace(BASE, **N),
    "MaskSVD 拡大 plane 符号探索 ICP1cm": replace(BASE, **N, flip_mode="search", icp_threshold=0.01),
    "MaskSVD 拡大 45deg 符号探索 ICP1cm": replace(BASE, **N, flip_mode="search", icp_threshold=0.01,
                                            checkpoint=str(CK / "model_epoch500_45deg.pth")),
}
R = dict(normalize_scale=True, icp_threshold=0.01, icp_iter=30)
REFINE = {
    "2段階 拡大 plane": replace(BASE, **R),
    "2段階 拡大 plane 残す割合0.3": replace(BASE, **R, keep_relative=0.3),
    "2段階 拡大 45deg": replace(BASE, **R, checkpoint=str(CK / "model_epoch500_45deg.pth")),
}
EVAL_MODEL = BASE.model_path


def main():
    pc = PoseConfig()
    svd_icp = SVDICPRegistration(pc)
    half = load_model(pc)
    regs = {name: MaskSVD(cfg) for name, cfg in VARIANTS.items()}
    refs = {name: MaskSVD(cfg) for name, cfg in REFINE.items()}
    full_noncrop = replace(BASE, flip_mode="search")
    res = {"既存 SVD+ICP（半分モデル）": [], "SVD のみ（MaskNet なし，全周モデル）": [],
           **{k: [] for k in regs}, **{k: [] for k in refs}}
    times = {k: [] for k in res}
    for i in range(1, 11):
        masks = load_masks(MASKS_DIR / "gt", i)
        fr = load_frame(i, masks[0].shape)
        depth_m = fr.depth.astype(np.float64) / 1000.0
        for m in masks:
            p = mask_to_points(m, fr, pc)
            if len(p) < pc.min_points:
                continue
            p = preprocess(p, pc)

            def record(name, fn):
                t0 = time.time()
                T = fn()
                times[name].append(time.time() - t0)
                res[name].append((free_space_violation(EVAL_MODEL, T, depth_m, fr.K), fitness(EVAL_MODEL, T, p)))

            T0 = svd_icp(half, p).T_model_to_cam
            record("既存 SVD+ICP（半分モデル）", lambda: T0)
            for name, r in refs.items():
                record(name, lambda r=r: r.refine(p, T0))
            base = next(iter(regs.values()))
            full = np.asarray(base.model.voxel_down_sample(0.002).points)
            record("SVD のみ（MaskNet なし，全周モデル）",
                   lambda: MaskSVD.register(base, p, template=full))
            for name, r in regs.items():
                record(name, lambda r=r: r.register(p))
        print(f"config {i} done", flush=True)

    print(f"\n{'手法':>34} {'成功(FSV<5%)':>12} {'失敗(FSV>20%)':>13} {'fitness':>8} {'時間[ms]':>9}")
    for name, rows in res.items():
        r = np.array(rows)
        print(f"{name:>34} {np.mean(r[:, 0] < 0.05) * 100:11.0f}% {np.mean(r[:, 0] > 0.2) * 100:12.0f}% "
              f"{np.mean(r[:, 1]):8.3f} {np.mean(times[name]) * 1000:9.1f}")


if __name__ == "__main__":
    main()
