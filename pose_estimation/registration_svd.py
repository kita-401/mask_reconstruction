"""SVD による初期姿勢推定 + ICP による精密位置合わせ．

pose_estimation_demo/scripts/Registration_SVD.py（SVD_ICP）を ROS 非依存に移植したもの．
アルゴリズムは同一：
  1. モデル・計測点群をそれぞれ重心で中心化し，SVD で主軸基底を求める
  2. 軸の符号反転 8 パターンのうち右手系（det > 0）のものを試し，fitness 最大を初期回転とする
  3. Open3D の Point-to-Point ICP で精密化
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

from .config import PoseConfig

FLIP_PATTERNS = [
    [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
    [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
]


@dataclass
class RegistrationResult:
    T_model_to_cam: np.ndarray   # (4, 4) モデル座標 → カメラ座標（＝カメラ座標系での物体姿勢）
    svd_fitness: float           # SVD 初期姿勢での fitness（flip_check_threshold 以内の割合）
    icp_fitness: float           # Open3D ICP の fitness
    icp_rmse: float              # Open3D ICP の inlier RMSE [m]
    flip: list                   # 選ばれた軸反転パターン


def _fitness_gpu(src, dst, threshold):
    """src の各点について dst への最近傍距離が threshold 未満の割合．"""
    min_dist, _ = torch.min(torch.cdist(src, dst), dim=1)
    return (torch.sum(min_dist < threshold).float() / src.shape[0]).item()


def _svd_basis(p):
    _, _, Vt = torch.linalg.svd(p, full_matrices=False)
    return Vt.T


class SVDICPRegistration:
    def __init__(self, cfg: PoseConfig = PoseConfig(), device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg.icp_max_iteration)

    def _candidates(self, basis_t, basis_s, src_c, tmpl_c):
        """初期回転の候補 [(fitness, R, flip, perm)]．

        original : 軸の対応は固定（第1↔第1, ...），符号反転のみ（元実装と同じ）
        all_perms: 隠れで主軸の順序が入れ替わる場合に備え，軸の対応の並べ替え 6 通りも試す
        """
        perms = [(0, 1, 2)] if self.cfg.hypotheses == "original" else \
            [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
        out = []
        for perm in perms:
            bs = basis_s[:, list(perm)]
            for p in FLIP_PATTERNS:
                R = basis_t @ (bs * torch.tensor(p, device=self.device, dtype=torch.float32)).T
                if torch.linalg.det(R) < 0:
                    continue
                fit = _fitness_gpu(src_c @ R.T, tmpl_c, self.cfg.flip_check_threshold)
                out.append((fit, R, p, perm))
        return out

    def _icp(self, src_o3d, tmpl_o3d, R):
        T_init = np.eye(4)
        T_init[:3, :3] = R.cpu().numpy()
        return o3d.pipelines.registration.registration_icp(
            src_o3d, tmpl_o3d, self.cfg.icp_threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), self.criteria)

    def __call__(self, model_pts, measured_pts) -> RegistrationResult:
        """model_pts, measured_pts: (N, 3) ndarray [m]（ダウンサンプリング済み）"""
        tmpl = torch.from_numpy(model_pts[:, :3]).to(self.device).float()
        src = torch.from_numpy(measured_pts[:, :3]).to(self.device).float()
        tmpl_mean = tmpl.mean(dim=0, keepdim=True)
        src_mean = src.mean(dim=0, keepdim=True)
        tmpl_c = tmpl - tmpl_mean
        src_c = src - src_mean
        src_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_c.cpu().numpy().astype(np.float64)))
        tmpl_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tmpl_c.cpu().numpy().astype(np.float64)))

        # --- 1. SVD による初期回転（計測 → モデル） ---
        cands = self._candidates(_svd_basis(tmpl_c), _svd_basis(src_c), src_c, tmpl_c)
        if not cands:
            cands = [(0.0, torch.eye(3, device=self.device), [1, 1, 1], (0, 1, 2))]

        # --- 2. ICP ---
        if self.cfg.icp_all_candidates:
            # 全候補で ICP を行い，ICP 後の fitness が最大のものを採用
            best = max(((c, self._icp(src_o3d, tmpl_o3d, c[1])) for c in cands),
                       key=lambda x: (x[1].fitness, -x[1].inlier_rmse))
            (best_fit, _, best_flip, _), reg = best
        else:
            # 元実装：ICP 前の fitness 最大の候補だけ ICP
            best_fit, best_R, best_flip, _ = max(cands, key=lambda c: c[0])
            reg = self._icp(src_o3d, tmpl_o3d, best_R)

        # 中心化を戻して 計測 → モデル の変換を作り，逆行列で物体姿勢にする
        R_f = reg.transformation[:3, :3]
        t_f = (reg.transformation[:3, 3] + tmpl_mean.cpu().numpy().ravel()
               - R_f @ src_mean.cpu().numpy().ravel())
        T_cam_to_model = np.eye(4)
        T_cam_to_model[:3, :3] = R_f
        T_cam_to_model[:3, 3] = t_f
        return RegistrationResult(
            T_model_to_cam=np.linalg.inv(T_cam_to_model),
            svd_fitness=float(best_fit),
            icp_fitness=float(reg.fitness),
            icp_rmse=float(reg.inlier_rmse),
            flip=list(best_flip),
        )
