"""種となる点群から，T字管の姿勢の仮説を複数作る（SVD による初期値 + ICP）．"""
import itertools

import numpy as np
import open3d as o3d

from .config import HypConfig

_PERMS = list(itertools.permutations(range(3)))
_FLIPS = [np.array(f) for f in itertools.product([1, -1], repeat=3)]


def _basis(p):
    _, _, Vt = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
    return Vt.T


def pose_hypotheses(seed_pts, icp_model, cfg: HypConfig):
    """種の点群に ICP で合わせたモデルの姿勢（モデル→カメラ）を fitness 上位 top_k_per_seed 個返す．

    SVD の主軸の対応（並べ替え）と符号の組み合わせを初期値として全て試す．
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed_pts)).voxel_down_sample(cfg.voxel)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(icp_model))
    s = np.asarray(src.points)
    if len(s) < 30:
        return []
    ms, mt = s.mean(axis=0), icp_model.mean(axis=0)
    Bs, Bt = _basis(s), _basis(icp_model)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_iter)
    perms = _PERMS if cfg.use_axis_perms else [(0, 1, 2)]
    out = []
    for perm in perms:
        for f in _FLIPS:
            R = Bt @ np.diag(f) @ Bs[:, list(perm)].T   # 計測 → モデル
            if np.linalg.det(R) < 0:
                continue
            T0 = np.eye(4)
            T0[:3, :3] = R
            T0[:3, 3] = mt - R @ ms
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, cfg.icp_threshold, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
            out.append((reg.fitness, -reg.inlier_rmse, np.linalg.inv(reg.transformation)))
    out.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # ほぼ同じ姿勢の重複を除く
    kept = []
    for fit, _, T in out:
        if all(np.linalg.norm(T[:3, 3] - K[:3, 3]) > 0.005 or
               np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ K[:3, :3]) - 1) / 2, -1, 1))) > 15
               for K in kept):
            kept.append(T)
        if len(kept) >= cfg.top_k_per_seed:
            break
    return kept
