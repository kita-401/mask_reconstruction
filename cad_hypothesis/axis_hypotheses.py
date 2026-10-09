"""円筒軸に基づく姿勢仮説の生成．

T字管モデル（t_pipe_full.pcd）の座標系：
    主管：X 軸方向（x ∈ [-43, 43] mm），枝管：+Y 方向（y ∈ [0, 43] mm），半径 14.5mm，原点は主管の軸上
断片の円筒軸（方向 a，通過点 c0，範囲 [t0, t1]）が分かれば，その断片は
    主管の一部 → モデルの X 軸 = ±a，原点は軸上
    枝管の一部 → モデルの Y 軸 = ±a，原点は軸上で枝管の根元
のどちらかなので，残る自由度は「軸まわりの回転角 φ」と「軸方向の位置」だけになる．
これらを並べた粗い候補を，周辺の観測点の説明数で絞り込み，上位を ICP で詰める．
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from mask_reconstruction.config import Config
from mask_reconstruction.merging import cylinder_axis

MAIN_HALF = 0.043     # [m] 主管の半長
BRANCH_LEN = 0.043    # [m] 原点から枝管の先端まで


@dataclass
class AxisHypConfig:
    n_angles: int = 12            # 軸まわりの回転角の分割数
    pos_step: float = 0.006       # [m] 軸方向の位置の刻み
    pos_slack: float = 0.012      # [m] 位置の範囲を両側にこれだけ広げる（断片の範囲の測定誤差対策）
    local_radius: float = 0.08    # [m] 粗い評価に使う周辺観測点の範囲（種の重心から）
    local_max_points: int = 3000
    coarse_tol: float = 0.003     # [m] 粗い評価で「説明できた」とみなす距離
    coarse_keep: int = 6          # ICP で詰める候補数
    refine_radius: float = 0.006  # [m] ICP に使う観測点（姿勢に置いたモデルからこの距離以内）
    refine_threshold: float = 0.005
    refine_iter: int = 20
    out_keep: int = 3             # 種ごとに出力する仮説数


def _perp_basis(a):
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, h)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _positions(lo_t, hi_t, step, slack):
    """[lo_t, hi_t] の範囲の位置候補．範囲が空（断片が長めに測られた場合）なら中央の前後を使う．"""
    if hi_t < lo_t:
        mid = 0.5 * (lo_t + hi_t)
        lo_t, hi_t = mid, mid
    return np.arange(lo_t - slack, hi_t + slack + 1e-9, step)


def coarse_poses(axis, cfg: AxisHypConfig):
    """円筒軸から粗い姿勢候補（モデル→カメラの 4x4）を列挙する．"""
    d, c0, t0, t1 = axis
    out = []
    for sign in (1.0, -1.0):
        a = sign * d
        lo, hi = (t0, t1) if sign > 0 else (-t1, -t0)   # a 方向での断片の範囲
        u, v = _perp_basis(a)
        for k in range(cfg.n_angles):
            phi = 2 * np.pi * k / cfg.n_angles
            b = np.cos(phi) * u + np.sin(phi) * v
            # 主管：X = a, Y = b
            R_main = np.stack([a, b, np.cross(a, b)], axis=1)
            for t in _positions(hi - MAIN_HALF, lo + MAIN_HALF, cfg.pos_step, cfg.pos_slack):
                out.append((R_main, c0 + t * a))
            # 枝管：Y = a, X = b（Z = X × Y）
            R_branch = np.stack([b, a, np.cross(b, a)], axis=1)
            for t in _positions(hi - BRANCH_LEN, lo, cfg.pos_step, cfg.pos_slack):
                out.append((R_branch, c0 + t * a))
    return out


class AxisHypothesisGenerator:
    def __init__(self, full_model, cfg: AxisHypConfig = AxisHypConfig(), axis_cfg: Config = Config()):
        self.cfg = cfg
        self.axis_cfg = axis_cfg
        self.model = full_model
        self.tree = cKDTree(full_model)
        self.rng = np.random.default_rng(0)

    def __call__(self, seed_pts, scene_pts):
        cfg = self.cfg
        if len(seed_pts) < 100:
            return []
        axis = cylinder_axis(seed_pts, self.axis_cfg, self.rng)
        center = seed_pts.mean(axis=0)
        local = scene_pts[np.linalg.norm(scene_pts - center, axis=1) < cfg.local_radius]
        if len(local) > cfg.local_max_points:
            local = local[self.rng.choice(len(local), cfg.local_max_points, replace=False)]
        seed_sub = seed_pts if len(seed_pts) <= 1500 else \
            seed_pts[self.rng.choice(len(seed_pts), 1500, replace=False)]

        # 粗い評価：周辺観測点のうちモデル表面から coarse_tol 以内の数（種の点は必ず説明すること）
        scored = []
        for R, o in coarse_poses(axis, cfg):
            seed_ok = np.mean(self.tree.query((seed_sub - o) @ R)[0] < cfg.coarse_tol)
            if seed_ok < 0.5:
                continue
            n_local = np.sum(self.tree.query((local - o) @ R)[0] < cfg.coarse_tol)
            scored.append((n_local + len(local) * seed_ok, R, o))
        scored.sort(key=lambda x: -x[0])

        # 上位を ICP（モデル → 周辺観測点）で詰める
        refined = []
        local_tree = cKDTree(local)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.refine_iter)
        for _, R, o in scored[:cfg.coarse_keep]:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, o
            posed = self.model @ R.T + o
            near = local_tree.query(posed)[0] < cfg.refine_radius
            idx = np.unique(local_tree.query(posed[near])[1]) if near.any() else []
            if len(idx) < 50:
                continue
            # 観測点の近くにあるモデル点（カメラから見えている側）だけを ICP に使う
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.model[near]))
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local[idx]))
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, cfg.refine_threshold, T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
            refined.append((reg.fitness, reg.transformation.copy()))
        refined.sort(key=lambda x: -x[0])
        return [T for _, T in refined[:cfg.out_keep]]
