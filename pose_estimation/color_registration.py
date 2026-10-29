"""回転対称な物体（野球ボールなど）の，色情報を用いた位置合わせ．

color_Registration/scripts/color_Registration_edge.py（SVD クラス）を ROS 非依存に移植したもの．
アルゴリズム・パラメータは同一：
  1. モデル・計測点群をそれぞれ重心で中心化し，色を半径 3 mm の近傍で平滑化する
  2. 色の勾配（近傍との色差の平均）が大きい上位 5% の点を「色エッジ点群」とする
  3. 形状の主軸どうしを SVD で合わせて初期回転とする（鏡像なら第3軸を反転）
  4. 初期回転と，それを第2主軸まわりに 180° 反転したものの両方から，Z 軸まわりの回転を
     色エッジ点群の fitness が最大になるよう探索する（30° 刻み → 最良の ±30° を 2° 刻み）
  5. 平滑化した全点で Point-to-Point ICP（閾値 1 cm，30 回）
元のコードとの違い：
  - torch_cluster / torch_scatter を使わず，近傍探索を素の PyTorch（torch.cdist）で行う
    （vlm_rag 環境では torch_cluster が読み込めないため）．半径内の近傍は最大 32 点で，
    元の radius() はどの 32 点を取るかが決まっていないのに対し，ここでは近い順に 32 点を取る
  - Z 軸まわりの探索で，最良の候補を初期回転の行列に上書きしながら次の候補を作る動き
    （元のコードの best_transform_stage2_t が T_initial_guess_t と同じ行列を指すため）もそのまま再現する
"""
import cv2
import numpy as np
import open3d as o3d
import torch

V_SIZE = 0.001             # 元のコードの点群の間隔 [m]（計測点群・モデルともこのボクセルで間引く前提）
EDGE_RAD = V_SIZE * 1.5    # 色の勾配を求める近傍の半径
SMOOTH_RAD = V_SIZE * 3    # 色を平滑化する近傍の半径
EDGE_PERCENT = 0.05        # 色エッジ点群の割合
MAX_NEIGHBORS = 32


def mask_circularity(mask):
    """マスクの外形の円形度 4πA / P^2（円で 1，細長い・枝分かれした形ほど小さい）．"""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    return float(4 * np.pi * cv2.contourArea(c) / per ** 2) if per > 0 else 0.0


# 回転対称な物体（ボール）とみなすマスクの円形度の下限．T字パイプ（10配置の正解マスク 146 個）は最大 0.81，
# 実機のボールは 0.88〜0.91（実機の T字パイプは 0.61 以下）なので，その間に置く
CIRCULARITY_MIN = 0.85


def mask_to_colored_points(mask, frame, cfg, voxel):
    """マスク内の有効深度画素を色付き点群 (N, 6)（xyz [m]，RGB [0, 1]）にし，voxel で間引く．

    点の取り方は pointcloud.mask_to_points と同じ（マスクの収縮・深度の範囲）．frame.rgb は BGR．
    """
    if cfg.mask_erode_px > 0:
        k = np.ones((2 * cfg.mask_erode_px + 1,) * 2, np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k) > 0
    v, u = np.nonzero(mask)
    z = frame.depth[v, u].astype(np.float64) / cfg.depth_scale
    ok = (z > cfg.depth_min) & (z < cfg.depth_max)
    u, v, z = u[ok], v[ok], z[ok]
    K = frame.K
    xyz = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    rgb = frame.rgb[v, u][:, ::-1].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.voxel_down_sample(voxel)
    return np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)])


def load_colored_model(path, voxel=V_SIZE):
    """色付きモデル点群 (N, 6) を voxel で間引いて読み込む（元のノードと同じ前処理）．"""
    pcd = o3d.io.read_point_cloud(path).voxel_down_sample(voxel)
    return np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)])


def _radius_neighbors(points, r):
    """各点の半径 r 以内の近傍（自分を含む，近い順に最大 MAX_NEIGHBORS 点）．(index [N, k], 有効 [N, k])．"""
    d = torch.cdist(points, points)
    k = min(MAX_NEIGHBORS, points.shape[0])
    dist, idx = torch.topk(d, k, dim=1, largest=False)
    return idx, dist <= r


def smooth_colors_by_radius(points, colors, search_radius):
    idx, valid = _radius_neighbors(points, search_radius)
    w = valid.float().unsqueeze(-1)
    smoothed = (colors[idx] * w).sum(1) / w.sum(1).clamp(min=1)
    nan_mask = torch.isnan(smoothed).any(dim=1)
    smoothed[nan_mask] = colors[nan_mask]
    return smoothed


def compute_color_gradient(points, colors, search_radius):
    idx, valid = _radius_neighbors(points, search_radius)
    diff = torch.linalg.norm(colors[idx] - colors.unsqueeze(1), dim=2)
    w = valid.float()
    return (diff * w).sum(1) / w.sum(1).clamp(min=1)


def get_color_edge_points(points, colors, search_radius, top_k_percent):
    if points.shape[0] == 0:
        return points, colors
    scores = compute_color_gradient(points, colors, search_radius)
    k = min(points.shape[0], max(10, int(points.shape[0] * top_k_percent)))
    _, edge = torch.topk(scores, k=k, largest=True)
    return points[edge], colors[edge]


def principal_vectors(points):
    """主軸基底 (3, 3)（列が第1〜第3主軸）と特異値．第1主軸の符号は元のコードと同じ決め方．"""
    if points.shape[0] < 3:
        return torch.eye(3, device=points.device), torch.zeros(3, device=points.device)
    centered = points - points.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt.T.clone()
    if torch.sum(centered @ basis[:, 0]) < 0:
        basis[:, 0] = -basis[:, 0]
    return basis, s


def _fitness(source, target, threshold):
    d = torch.cdist(source, target).min(dim=1).values
    return (d < threshold).float().mean()


def _rot_z(angle_deg, device):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device).float()


class ColorEdgeRegistration:
    """色エッジによる回転対称物体の位置合わせ．__call__(model (M, 6), measured (N, 6)) → (4, 4) モデル → カメラ．"""

    def __init__(self, icp_threshold=0.01, icp_max_iteration=30, device=None):
        self.icp_threshold = icp_threshold
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iteration)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_fitness = float("nan")   # Z 軸まわりの探索で最良だった色エッジの fitness

    def _prepare(self, cloud):
        t = torch.as_tensor(cloud, device=self.device).float()
        p, c = t[:, :3], t[:, 3:6]
        mean = p.mean(0, keepdim=True)
        p = p - mean
        c = smooth_colors_by_radius(p, c, SMOOTH_RAD)
        ep, _ = get_color_edge_points(p, c, EDGE_RAD, EDGE_PERCENT)
        return p, c, ep, mean

    def _axis_search(self, T_init, source_edge, model_edge):
        threshold = V_SIZE * 5
        step_c, range_c = 30.0, 180.0
        best_fit, best_angle = -1.0, 0.0
        best = T_init   # 元のコードと同じく，初期回転の行列そのものに最良の候補を上書きしていく
        for angle in np.arange(-range_c, range_c, step_c):
            R = _rot_z(angle, self.device) @ T_init[:3, :3]
            fit = _fitness(source_edge @ R.T, model_edge, threshold)
            if fit > best_fit:
                best_fit, best_angle = fit, angle
                best[:3, :3] = R
        for angle in np.arange(best_angle - step_c, best_angle + step_c, 2.0):
            if angle == best_angle:
                continue
            R = _rot_z(angle, self.device) @ T_init[:3, :3]
            fit = _fitness(source_edge @ R.T, model_edge, threshold)
            if fit > best_fit:
                best_fit = fit
                best[:3, :3] = R
        return best, float(best_fit)

    def register(self, model, measured):
        """計測点群 → モデルの変換（元のコードの est_T と同じ向き）．"""
        mp, mc, me, m_mean = self._prepare(model)
        sp, sc, se, s_mean = self._prepare(measured)
        gm, _ = principal_vectors(mp)
        gs, _ = principal_vectors(sp)
        R0 = gm @ gs.T
        if torch.linalg.det(R0) < 0:
            R0 = gm @ torch.diag(torch.tensor([1.0, 1.0, -1.0], device=self.device)) @ gs.T
        T0 = torch.eye(4, device=self.device)
        T0[:3, :3] = R0
        v = gm[:, 1].unsqueeze(1)
        T0_flip = torch.eye(4, device=self.device)
        T0_flip[:3, :3] = (2.0 * v @ v.T - torch.eye(3, device=self.device)) @ R0
        T_a, fit_a = self._axis_search(T0, se, me)
        T_b, fit_b = self._axis_search(T0_flip, se, me)
        T_init, self.last_fitness = (T_b, fit_b) if fit_b > fit_a else (T_a, fit_a)

        def o3d_cloud(p, c):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.detach().cpu().numpy().astype(np.float64)))
            pc.colors = o3d.utility.Vector3dVector(c.detach().cpu().numpy().astype(np.float64))
            return pc

        reg = o3d.pipelines.registration.registration_icp(
            o3d_cloud(sp, sc), o3d_cloud(mp, mc), self.icp_threshold, T_init.detach().cpu().numpy().astype(np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), self.criteria)
        T = np.asarray(reg.transformation).copy()
        # 重心で中心化した分を戻す
        m0, s0 = m_mean.squeeze().cpu().numpy(), s_mean.squeeze().cpu().numpy()
        T[:3, 3] = T[:3, 3] + m0 - T[:3, :3] @ s0
        self.last_icp_fitness = float(reg.fitness)
        return T

    def __call__(self, model, measured):
        """モデル座標 → カメラ座標の姿勢 (4, 4)．"""
        return np.linalg.inv(self.register(model, measured))
