"""マスクと深度からインスタンスごとの計測点群を作る．"""
from functools import lru_cache

import cv2
import numpy as np
import open3d as o3d

from .config import PoseConfig


def mask_to_points(mask, frame, cfg: PoseConfig):
    """マスク内の有効深度画素を3次元点群 (N, 3) [m] に変換する．"""
    if cfg.mask_erode_px > 0:
        k = np.ones((2 * cfg.mask_erode_px + 1,) * 2, np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k) > 0
    v, u = np.nonzero(mask)
    z = frame.depth[v, u].astype(np.float64) / cfg.depth_scale
    ok = (z > cfg.depth_min) & (z < cfg.depth_max)
    u, v, z = u[ok], v[ok], z[ok]
    K = frame.K
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def preprocess(points, cfg: PoseConfig):
    """ダウンサンプリングと統計的外れ値除去．"""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(cfg.voxel_size)
    if cfg.outlier_nb > 0 and len(pcd.points) > cfg.outlier_nb:
        pcd, _ = pcd.remove_statistical_outlier(cfg.outlier_nb, cfg.outlier_std)
    return np.asarray(pcd.points)


@lru_cache(maxsize=8)
def ratio_model_size(path, voxel_size):
    """点数の比に使う全周モデルを voxel_size で間引いた点数．"""
    return len(o3d.io.read_point_cloud(path).voxel_down_sample(voxel_size).points)


def enough_points(measured, cfg: PoseConfig):
    """位置合わせするだけの点があるか：間引き後の計測点数 ≥ 全周モデルの点数 × min_model_ratio．

    measured は preprocess の出力（cfg.voxel_size で間引いたもの）．全周モデルも同じ voxel_size で間引く．
    """
    if cfg.min_model_ratio <= 0:
        return True
    return len(measured) >= cfg.min_model_ratio * ratio_model_size(cfg.ratio_model_path, cfg.voxel_size)


def load_model(cfg: PoseConfig):
    pcd = o3d.io.read_point_cloud(cfg.model_path)
    return np.asarray(pcd.voxel_down_sample(cfg.voxel_size).points)
