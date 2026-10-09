"""MaskSVD の ROS 非依存版．

vision_ws/src/MaskSVD/estimate_demo.py と Registration_SVD.py の処理を移植したもの：
  1. 計測点群の密度からボクセルサイズを決めてモデルを間引く（weight × (1/密度)^(1/3)）
  2. MaskNet でモデル点群を計測点群に対応する部分だけに削る
  3. 削ったモデルと計測点群の主成分で回転を決める（det<0 なら第1軸を反転，T字管用）
  4. ICP（対応距離 0.1m，100 回）
MaskNet のネットワーク定義と重みは vision_ws から読み込む（vision_ws は編集しない）．
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

MASKSVD_DIR = Path(__file__).resolve().parents[2] / "vision_ws" / "src" / "MaskSVD"


@dataclass
class MaskSVDConfig:
    checkpoint: str = str(MASKSVD_DIR / "checkpoint" / "model_weight_epoch300_batchsize32_plane.pth")
    model_path: str = str(MASKSVD_DIR / "TNUTEJN016_10000.pcd")
    density_weight: float = 0.8
    point_selection: str = "as_trained"   # "as_trained"（読み込んだまま＝topk）/ "threshold"（確率 0.5 以上）
    center_inputs: bool = False           # True: MaskNet に入れる前に両方を各自の重心で中心化
    icp_threshold: float = 0.1            # [m]（元実装の値）
    icp_iter: int = 100
    flip_mode: str = "original"           # "original": det<0 で第1軸反転 / "search": 符号4通りを fitness で選ぶ
    normalize_scale: bool = False         # True: モデルが半径1程度になるよう両方を拡大して MaskNet に入れる
    keep_relative: float = 0.5            # refine：確率が最大値のこの割合以上のモデル点を残す
    input_sizes: str = "density"          # "density": 元実装の密度ベース / "trained": 学習時と同じ モデル1024点・計測256点


def load_masknet(cfg: MaskSVDConfig, device):
    if str(MASKSVD_DIR) not in sys.path:
        sys.path.insert(0, str(MASKSVD_DIR))
    import network  # noqa: F401  （torch.load がクラス定義を参照する）
    obj = torch.load(cfg.checkpoint, weights_only=False, map_location=device)
    if isinstance(obj, dict):                 # state_dict 形式の重み
        net = network.MaskNet()
        net.load_state_dict(obj)
    else:
        net = obj
    if cfg.point_selection == "threshold":
        net.is_training = False
    net.eval()
    return net.to(device)


def _pca_axes(p):
    c = p - p.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return Vt.T


class MaskSVD:
    def __init__(self, cfg: MaskSVDConfig = MaskSVDConfig(), device=None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_masknet(cfg, self.device)
        self.model = o3d.io.read_point_cloud(cfg.model_path)
        mp = np.asarray(self.model.points)
        self.model_radius = float(np.max(np.linalg.norm(mp - mp.mean(axis=0), axis=1)))

    def crop_model(self, measured):
        """MaskNet で計測点群に対応するモデル点だけを残す（モデル座標のまま返す）．"""
        cfg = self.cfg
        ext = np.maximum(np.ptp(measured, axis=0), 1e-4)
        density = len(measured) / np.prod(ext)
        voxel = cfg.density_weight * (1.0 / density) ** (1 / 3)
        tmpl = np.asarray(self.model.voxel_down_sample(voxel).points)
        rng = np.random.default_rng(0)
        src = measured
        if cfg.input_sizes == "trained":
            full = np.asarray(self.model.points)
            tmpl = full[rng.choice(len(full), min(1024, len(full)), replace=False)]
            src = measured[rng.choice(len(measured), min(256, len(measured)), replace=False)]
        elif len(src) >= len(tmpl):
            # topk は計測点の数だけモデル点を選ぶので，計測点をモデル点より少なくする
            src = measured[rng.choice(len(measured), len(tmpl) // 2, replace=False)]
        t_in, s_in = tmpl, src
        if cfg.center_inputs:
            t_in, s_in = tmpl - tmpl.mean(axis=0), src - src.mean(axis=0)
        if cfg.normalize_scale:
            k = 1.0 / self.model_radius
            t_in, s_in = t_in * k, s_in * k
        tt = torch.tensor(t_in, dtype=torch.float32, device=self.device).unsqueeze(0)
        st = torch.tensor(s_in, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.net(tt, st)
        idx = self.net.mask_idx[0].cpu().numpy()
        return tmpl[idx] if len(idx) >= 10 else tmpl

    def refine(self, measured, T_init, model_pts=None):
        """2段階：既存の推定姿勢 T_init（モデル→カメラ）でモデルを計測点群に重ねてから MaskNet で削り，
        削ったモデルで ICP をかけ直す（3Dto2D/test_masknet_icp.py と同じ使い方）．"""
        cfg = self.cfg
        full = np.asarray(self.model.voxel_down_sample(0.002).points) if model_pts is None else model_pts
        posed = full @ T_init[:3, :3].T + T_init[:3, 3]
        c = measured.mean(axis=0)
        rng = np.random.default_rng(0)
        t_idx = rng.choice(len(posed), min(1024, len(posed)), replace=False)
        s_idx = rng.choice(len(measured), min(256, len(measured)), replace=False)
        k = 1.0 / self.model_radius if cfg.normalize_scale else 1.0
        tt = torch.tensor((posed[t_idx] - c) * k, dtype=torch.float32, device=self.device).unsqueeze(0)
        st = torch.tensor((measured[s_idx] - c) * k, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, mask = self.net(tt, st)
        mk = mask[0].cpu().numpy()
        keep = mk >= cfg.keep_relative * mk.max() if cfg.normalize_scale else mk > 0.5
        crop = full[t_idx][keep] if keep.sum() >= 30 else full[t_idx]
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(measured))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(crop @ T_init[:3, :3].T + T_init[:3, 3]))
        res = o3d.pipelines.registration.registration_icp(
            tgt, src, cfg.icp_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_iter))
        return res.transformation @ T_init

    def register(self, measured, template=None):
        """モデル → カメラ の姿勢（4x4）を返す．template を渡すと MaskNet を省略（比較用）．"""
        cfg = self.cfg
        if template is None:
            template = self.crop_model(measured)
        tm, sm = template.mean(axis=0), measured.mean(axis=0)
        Tv, Sv = _pca_axes(template), _pca_axes(measured)
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(measured - sm))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template - tm))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_iter)

        if cfg.flip_mode == "original":
            R = Sv @ Tv.T
            if np.linalg.det(R) < 0:
                R = Sv @ np.diag([-1, 1, 1]) @ Tv.T
            cands = [R.T]
        else:
            cands = []
            for f in ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]):
                R = (Sv @ np.diag(f) @ Tv.T)
                if np.linalg.det(R) < 0:
                    R = Sv @ np.diag(f) @ np.diag([-1, 1, 1]) @ Tv.T
                cands.append(R.T)
        best = None
        for R in cands:                         # R: 計測 → モデル
            T0 = np.eye(4)
            T0[:3, :3] = R
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, cfg.icp_threshold, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
            if best is None or res.fitness > best.fitness:
                best = res
        Rf = best.transformation[:3, :3]
        tf = best.transformation[:3, 3] + tm - Rf @ sm
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rf, tf            # 計測 → モデル
        return np.linalg.inv(T)
