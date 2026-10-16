"""CADモデル（点群）から部品固有の大きさ・凹角をオフラインで求めてキャッシュする（改善2・3）．

部品の種類に依存する値（円筒の半径など）は使わず，モデル点群だけから計算する．
  L1 ≥ L2 ≥ L3 : PCA 軸に合わせた OBB の軸長
  D_cad        : 最大点間距離（凸包の頂点で計算）
  A_cad        : ランダムな回転で正射影したシルエット面積の最大値
  θ_cad_max    : 外から見える面での凹の二面角（LCCP の判定で凹となる点対の法線角）の上位パーセンタイル
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .config import Config


@dataclass(frozen=True)
class CadPrior:
    L: tuple             # [m] (L1, L2, L3)
    D: float             # [m]
    A: float             # [m^2]
    theta_max: float     # [deg]
    concave_hist: tuple  # 凹角の分布（確認用）：(ビンの左端 [deg], 割合) の組


def obb_lengths(p):
    c = p - p.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ Vt.T
    return np.sort(np.ptp(proj, axis=0))[::-1]


def max_distance(p):
    """凸包の頂点どうしの最大距離．"""
    try:
        v = p[ConvexHull(p).vertices]
    except Exception:  # 退化（平面上など）
        v = p if len(p) <= 3000 else p[np.random.default_rng(0).choice(len(p), 3000, replace=False)]
    d2 = np.sum((v[:, None] - v[None]) ** 2, axis=2)
    return float(np.sqrt(d2.max()))


def _spacing(p):
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def max_projected_area(p, n_rot, seed):
    """ランダムな回転で xy 平面に正射影し，ラスタライズしたシルエットの面積の最大値 [m^2]．

    ラスタの画素は点間隔の 2 倍とし，3x3 の closing で点の間の隙間だけを埋める（貫通穴は埋めない）．
    """
    res = 2.0 * _spacing(p)
    c = p - p.mean(axis=0)
    best = 0.0
    k = np.ones((3, 3), np.uint8)
    for R in Rotation.random(n_rot, random_state=seed).as_matrix():
        q = (c @ R.T)[:, :2]
        ij = np.floor((q - q.min(axis=0)) / res).astype(int) + 1
        img = np.zeros(ij.max(axis=0) + 2, np.uint8)
        img[ij[:, 0], ij[:, 1]] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
        best = max(best, float(img.sum()) * res * res)
    return best


def _view_dirs(n_views):
    k = np.arange(n_views) + 0.5
    phi = np.arccos(1 - 2 * k / n_views)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def visible_points(q, d, res, tol):
    """方向 d（モデル→カメラ）から正射影で見たときに見える点（z バッファの 3x3 近傍最小 + tol 以内）．"""
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    ij = np.floor(np.stack([q @ e1, q @ e2], axis=1) / res).astype(int)
    ij -= ij.min(axis=0) - 1
    depth = -(q @ d)                                    # 小さいほどカメラに近い
    zb = np.full(ij.max(axis=0) + 2, np.inf)
    np.minimum.at(zb, (ij[:, 0], ij[:, 1]), depth)
    zb3 = np.min([np.roll(np.roll(zb, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)
    return depth <= zb3[ij[:, 0], ij[:, 1]] + tol


def concave_angles(p, cfg: Config, n_views=42, graze_deg=75.0, max_points=10000):
    """カメラと同じ条件で見た CAD 表面の点対のうち，凹と判定される対の法線角 [deg]．

    多数の方向から正射影で見て（z バッファで見える点だけ），法線をカメラ側に向け，
    視線とのなす角が graze_deg 以上の点（センサで測れない斜面）を除く．
    距離 cad_concave_pair_dist の範囲の点対について，観測側（convexity.py）と同じ判定をする：
    角 < flat_angle は平坦，(n_i − n_j)·d > 0 なら凸，それ以外は凹．
    """
    if len(p) > max_points:
        p = p[np.random.default_rng(0).choice(len(p), max_points, replace=False)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    normals = np.asarray(pc.normals)
    q = p - p.mean(axis=0)
    sp = _spacing(p)
    r0, r1 = cfg.cad_concave_pair_dist
    angles, n_pairs = [], 0
    for d in _view_dirs(n_views):
        vis = visible_points(q, d, 2 * sp, 2 * sp)
        x, n = q[vis], normals[vis].copy()
        dot = n @ d
        n[dot < 0] *= -1
        keep = np.abs(dot) > np.cos(np.radians(graze_deg))
        x, n = x[keep], n[keep]
        pairs = cKDTree(x).query_pairs(r1, output_type="ndarray")
        diff = x[pairs[:, 0]] - x[pairs[:, 1]]
        dist = np.linalg.norm(diff, axis=1)
        ok = dist >= r0
        pairs, diff, dist = pairs[ok], diff[ok], dist[ok]
        ni, nj = n[pairs[:, 0]], n[pairs[:, 1]]
        ang = np.degrees(np.arccos(np.clip(np.sum(ni * nj, axis=1), -1, 1)))
        convex = np.sum((ni - nj) * diff, axis=1) / dist > 0
        angles.append(ang[(ang >= cfg.flat_angle) & ~convex])
        n_pairs += len(pairs)
    return np.concatenate(angles), n_pairs


def _cache_path(cfg: Config):
    key = hashlib.sha1(Path(cfg.cad_model).read_bytes())
    key.update(repr((cfg.cad_proj_rotations, cfg.cad_proj_seed, cfg.cad_concave_pair_dist,
                     cfg.cad_concave_percentile, cfg.flat_angle)).encode())
    return Path(cfg.cad_cache_dir) / f"{Path(cfg.cad_model).stem}_{key.hexdigest()[:12]}.json"


_MEMO = {}


def load_cad_prior(cfg: Config) -> CadPrior:
    path = _cache_path(cfg)
    if path in _MEMO:
        return _MEMO[path]
    if path.exists():
        d = json.loads(path.read_text())
    else:
        p = np.asarray(o3d.io.read_point_cloud(cfg.cad_model).points)
        ang, n_pairs = concave_angles(p, cfg)
        hist, edges = np.histogram(ang, bins=18, range=(0, 180))
        d = dict(L=obb_lengths(p).tolist(), D=max_distance(p),
                 A=max_projected_area(p, cfg.cad_proj_rotations, cfg.cad_proj_seed),
                 theta_max=float(np.percentile(ang, cfg.cad_concave_percentile)) if len(ang) else 0.0,
                 concave_hist=[[float(e), float(h / max(n_pairs, 1))] for e, h in zip(edges[:-1], hist)])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))
    prior = CadPrior(L=tuple(d["L"]), D=d["D"], A=d["A"], theta_max=d["theta_max"],
                     concave_hist=tuple(map(tuple, d["concave_hist"])))
    _MEMO[path] = prior
    return prior
