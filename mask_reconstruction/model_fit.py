"""「モデル1個で説明できるか」の判定（部品の種類に依存しない形状チェック）．

必要なのはモデル点群だけで，寸法や軸などの形状パラメータは使わない．
2つの断片の点群をまとめたものにモデルを当てはめ（FPFH + RANSAC → ICP），
各断片の点のうちモデル表面から tol 以内にある割合（説明率）を返す．
同じ部品なら両方が説明され，別の部品ならモデル1個では片方しか説明できない．
"""
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

reg = o3d.pipelines.registration


def _fpfh(pcd, voxel):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


@lru_cache(maxsize=4)
def _model(path, voxel):
    pcd = o3d.io.read_point_cloud(path).voxel_down_sample(voxel)
    return pcd, _fpfh(pcd, voxel), np.asarray(pcd.points)


def fit_model(obs_pts, cfg, seed=0):
    """モデル → 観測 の姿勢（4x4）を求める．"""
    o3d.utility.random.seed(seed)
    voxel = cfg.model_check_voxel
    model_pcd, model_f, _ = _model(cfg.model_check_model, voxel)
    obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs_pts)).voxel_down_sample(voxel)
    obs_f = _fpfh(obs, voxel)
    ransac = reg.registration_ransac_based_on_feature_matching(
        model_pcd, obs, model_f, obs_f, True, voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        reg.RANSACConvergenceCriteria(cfg.model_check_ransac_iter, 0.999))
    icp = reg.registration_icp(model_pcd, obs, voxel * 2, ransac.transformation,
                               reg.TransformationEstimationPointToPoint())
    return icp.transformation


def explained_ratios(parts, cfg):
    """parts（点群のリスト）をまとめてモデル1個を当てはめ，各点群の説明率を返す．"""
    _, _, model_pts = _model(cfg.model_check_model, cfg.model_check_voxel)
    T = fit_model(np.vstack(parts), cfg)
    tree = cKDTree(model_pts @ T[:3, :3].T + T[:3, 3])
    return [float(np.mean(tree.query(p)[0] < cfg.model_check_tol)) for p in parts]
