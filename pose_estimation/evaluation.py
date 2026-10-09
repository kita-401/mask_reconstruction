"""姿勢推定の評価．真値姿勢がないため，以下で評価する．

- fitness / RMSE : 計測点のうち，推定姿勢のモデルから eval_threshold 以内にある割合とその RMSE
- ADD-S          : 2つの姿勢でモデルを配置したときの対称性を考慮した平均最近傍距離 [m]．
                   正解マスクから推定した姿勢を基準とし，セグメンテーション結果による姿勢のずれを測る．
"""
import numpy as np
from scipy.spatial import cKDTree


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def fit_quality(model_pts, measured_pts, T_model_to_cam, threshold):
    d, _ = cKDTree(transform(model_pts, T_model_to_cam)).query(measured_pts)
    inl = d < threshold
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan")
    return float(inl.mean()), rmse


def free_space_violation(model_pts, T_model_to_cam, frame, depth_scale=1000.0, tol=0.010):
    """推定姿勢のモデル点のうち，計測深度より tol 以上手前に来る点の割合．

    隠れている部分は計測深度より奥にあるので違反にならない．手前に来る点は
    「カメラに写っているはずなのに写っていない」ので，姿勢が誤っていることを示す．
    """
    p = transform(model_pts, T_model_to_cam)
    z = p[:, 2]
    K = frame.K
    u = np.round(K[0, 0] * p[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[:, 1] / z + K[1, 2]).astype(int)
    h, w = frame.shape
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = frame.depth[v[ok], u[ok]].astype(np.float64) / depth_scale
    has = d > 0
    if not has.any():
        return float("nan")
    return float(np.mean(z[ok][has] < d[has] - tol))


def add_s(model_pts, T_a, T_b):
    pa, pb = transform(model_pts, T_a), transform(model_pts, T_b)
    d, _ = cKDTree(pb).query(pa)
    return float(d.mean())


def match_masks(masks, ref_masks, iou_thresh=0.5):
    """masks の各マスクに対し IoU 最大の ref_masks のインデックス（閾値未満なら -1）．"""
    out = []
    for m in masks:
        best, bj = 0.0, -1
        for j, r in enumerate(ref_masks):
            inter = np.sum(m & r)
            if inter == 0:
                continue
            iou = inter / np.sum(m | r)
            if iou > best:
                best, bj = iou, j
        out.append(bj if best >= iou_thresh else -1)
    return out
