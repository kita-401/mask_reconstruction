"""モデル点群からの形状特徴の自動推定（部品の種類に依存しない）．

円筒面上の点 p（法線 n）を軸側へ半径 r だけ移した点 c = p − r·n は，正しい r のときだけ
円筒の中心軸（線）の上に集まる．そこで r を変えながら c の集まり具合（近傍点までの距離）が
最小になる r を半径とし，そのとき線状に集まった点の割合を「円筒でできている部分の割合」とする．
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CylinderInfo:
    radius: float        # [m] 推定した円筒の半径
    fraction: float      # モデル点のうち円筒面とみなせる割合
    score_curve: tuple   # (r, 集まり具合) の組（確認用）


def _normals(points, k=20):
    """表面に沿って向きを伝えて一貫させた外向き法線．"""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
    pcd.orient_normals_consistent_tangent_plane(k)
    n = np.asarray(pcd.normals)
    # 全体として重心から外向きが多数派になるようにそろえる
    if np.mean(np.sum((points - points.mean(axis=0)) * n, axis=1) > 0) < 0.5:
        n = -n
    return n


def _spacing(points):
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def estimate_cylinder(points, n_radii=60, k=10, linear_thresh=0.9):
    """モデル点群から円筒の半径と，円筒面の割合を推定する．"""
    pts = np.asarray(points)
    if len(pts) > 6000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 6000, replace=False)]
    n = _normals(pts)
    spacing = _spacing(pts)
    extent = np.ptp(pts, axis=0)
    radii = np.linspace(2 * spacing, 0.5 * np.sort(extent)[1], n_radii)

    curve = []
    for r in radii:
        c = pts - r * n
        d, _ = cKDTree(c).query(c, k=k + 1)
        curve.append(float(np.median(d[:, -1])))
    r_best = float(radii[int(np.argmin(curve))])

    # r_best で軸側へ移した点のうち，周囲（半径 line_radius 以内）が線状に並ぶ点の割合
    c = pts - r_best * n
    tree = cKDTree(c)
    line_radius = 5 * spacing
    linear = np.zeros(len(c), dtype=bool)
    for i, nbr in enumerate(tree.query_ball_point(c, line_radius)):
        if len(nbr) < 5:
            continue
        q = c[nbr] - c[nbr].mean(axis=0)
        ev = np.linalg.eigvalsh(q.T @ q / len(q))                       # 昇順
        linear[i] = ev[2] / max(ev.sum(), 1e-18) > linear_thresh
    frac = float(np.mean(linear))
    return CylinderInfo(radius=r_best, fraction=frac, score_curve=tuple(zip(radii.tolist(), curve)))


def outer_shell(points, n_views=42):
    """モデルの周囲の多数の方向から見て，少なくとも1方向から見える点だけを返す（内側の面などを除く）．

    カメラに写るのは外から見える面だけなので，形状特徴はこの面から求める．
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    center = points.mean(axis=0)
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    # 球面上にほぼ一様な視点（フィボナッチ格子）
    k = np.arange(n_views) + 0.5
    phi = np.arccos(1 - 2 * k / n_views)
    theta = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    visible = set()
    for d in dirs:
        _, idx = pcd.hidden_point_removal(center + d * diameter * 2, diameter * 100)
        visible.update(idx)
    return points[sorted(visible)]


@lru_cache(maxsize=8)
def cylinder_info_from_file(path):
    return estimate_cylinder(outer_shell(np.asarray(o3d.io.read_point_cloud(path).points)))
