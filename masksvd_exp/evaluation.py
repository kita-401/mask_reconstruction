"""姿勢の評価（どの位置合わせ手法にも同じ基準）．

自由空間違反率：全周モデルのうち，推定姿勢でカメラ側を向く面の点だけを使い，
その点が計測深度より tol 以上手前に来る割合．隠れた部分は奥にあるので違反しない．
"""
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@lru_cache(maxsize=4)
def model_with_normals(path, voxel=0.002):
    pcd = o3d.io.read_point_cloud(path).voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    pcd.orient_normals_consistent_tangent_plane(20)
    p, n = np.asarray(pcd.points), np.asarray(pcd.normals)
    if np.mean(np.sum((p - p.mean(axis=0)) * n, axis=1) > 0) < 0.5:
        n = -n
    return p, n


def front_points(model_path, T):
    p, n = model_with_normals(model_path)
    pc = p @ T[:3, :3].T + T[:3, 3]
    nc = n @ T[:3, :3].T
    return pc[np.sum(nc * -pc, axis=1) > 0]     # カメラ（原点）を向く面


def free_space_violation(model_path, T, depth_m, K, tol=0.010):
    q = front_points(model_path, T)
    z = q[:, 2]
    u = np.round(K[0, 0] * q[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * q[:, 1] / z + K[1, 2]).astype(int)
    h, w = depth_m.shape
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = depth_m[v[ok], u[ok]]
    has = d > 0
    return float(np.mean(z[ok][has] < d[has] - tol)) if has.any() else float("nan")


def fitness(model_path, T, measured, tol=0.003):
    q = front_points(model_path, T)
    return float(np.mean(cKDTree(q).query(measured)[0] < tol))
