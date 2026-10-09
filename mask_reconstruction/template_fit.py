"""CADテンプレートの当てはまり評価（統合の検証用）．

SVD による初期姿勢（軸の符号反転4通り）+ ICP で，点群にテンプレートを当てはめたときの
fitness（点群のうちテンプレートから threshold 以内にある割合）を返す．
姿勢推定パッケージ（pose_estimation）とは独立に保つため，ここに最小限の実装を置く．
"""
from functools import lru_cache

import numpy as np
import open3d as o3d

_FLIPS = [np.diag(f) for f in ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])]


@lru_cache(maxsize=4)
def _load_template(path, voxel):
    pcd = o3d.io.read_point_cloud(path).voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    return pts - pts.mean(axis=0)


def _basis(p):
    _, _, Vt = np.linalg.svd(p, full_matrices=False)
    B = Vt.T
    if np.linalg.det(B) < 0:
        B[:, 2] *= -1
    return B


def template_fitness(points, cfg, max_points=1500, seed=0):
    tmpl = _load_template(cfg.template_model, cfg.template_voxel)
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    src = points - points.mean(axis=0)
    src_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    tmpl_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tmpl))
    Bt, Bs = _basis(tmpl), _basis(src)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.template_icp_iter)
    best = 0.0
    for F in _FLIPS:
        T = np.eye(4)
        T[:3, :3] = Bt @ F @ Bs.T
        reg = o3d.pipelines.registration.registration_icp(
            src_o3d, tmpl_o3d, cfg.template_icp_threshold, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
        fit = o3d.pipelines.registration.evaluate_registration(
            src_o3d, tmpl_o3d, cfg.template_fit_threshold, reg.transformation).fitness
        best = max(best, fit)
    return best
