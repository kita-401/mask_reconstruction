"""球全体の色付きモデル（YCB の野球ボールなど）を使う，縫い目による 3 自由度の回転探索．

カメラに写るのはボールの手前の半球だけだが，球全体のモデルならどの面が写っていても対応する縫い目がモデルにある．
  1. 計測点群に半径既知（モデルの半径）の球を当てはめて中心を求める（位置はこれで決まる）
  2. 計測・モデルの模様の点を，中心からの方向（単位ベクトル）にする．模様の点の取り方（feature）は
     "red"：赤い点（野球ボールの縫い目専用．ロゴなどの黒い印刷は使わない）/
     "edge"：色の境界（近傍との色の差が大きい上位 edge_percent の点．模様の色によらない）．
     色の境界では，計測側のカメラから見て球面が max_view_deg 以上傾いた縁の点（背景との境目や縁の影）は使わない
  3. 回転 R（モデル → カメラ）の候補ごとに，計測の縫い目の方向を R^T でモデル座標に戻し，
     モデルの縫い目までの角度距離（あらかじめ緯度経度の格子に計算した表を引く）から一致度を求める
  4. 一様に散らした回転の候補（粗い探索）→ 上位の候補のまわりを段階的に細かく探す
球の縫い目の曲線には対称性があり，ロゴが写っていなければ同じ縫い目になる回転が複数ある（どれも形は同じ）．
"""
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def red_mask(rgb01):
    """色 (N, 3)（RGB，0〜1）のうち，縫い目の赤い糸の点．"""
    hsv = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return ((h < 12) | (h > 165)) & (s > 70) & (v > 40)


def fit_center_known_radius(points, radius, iters=20):
    """半径既知の球を当てはめた中心（ガウス・ニュートン法．初期値は半径も未知の線形最小二乗）．"""
    A = np.hstack([2 * points, np.ones((len(points), 1))])
    x, *_ = np.linalg.lstsq(A, (points ** 2).sum(1), rcond=None)
    c = x[:3]
    for _ in range(iters):
        d = points - c
        n = np.linalg.norm(d, axis=1)
        J = -d / n[:, None]
        r = n - radius
        c = c - np.linalg.lstsq(J, r, rcond=None)[0]
    return c


def otsu_threshold(x, bins=256):
    """値の分布を 2 つに分ける大津のしきい値．"""
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    var = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return centers[int(np.argmax(var))]


def color_edge_points(points, colors, smooth_rad=0.003, edge_rad=0.0015, percent=0.05, k=32):
    """色の境界の点の番号：色を半径 smooth_rad で平滑化し，半径 edge_rad の近傍との色の差の平均が大きい上位 percent．
    percent=None なら割合を決めず，色の差が大津のしきい値を超える点を全部返す．
    （color_registration.py の色エッジと同じ考え方．点が多くても使えるよう KD 木で近傍を探す）"""
    tree = cKDTree(points)
    k = min(k, len(points))

    def neigh(r):
        d, idx = tree.query(points, k=k, distance_upper_bound=r)
        valid = np.isfinite(d)
        idx = np.where(valid, idx, 0)
        return idx, valid
    idx, valid = neigh(smooth_rad)
    w = valid[..., None].astype(np.float64)
    smooth = (colors[idx] * w).sum(1) / np.maximum(w.sum(1), 1)
    idx, valid = neigh(edge_rad)
    diff = np.linalg.norm(smooth[idx] - smooth[:, None, :], axis=2)
    score = (diff * valid).sum(1) / np.maximum(valid.sum(1), 1)
    if percent is None:
        return np.nonzero(score > otsu_threshold(score))[0]
    n = max(10, int(len(points) * percent))
    return np.argsort(-score)[:n]


def color_edge_points_gpu(points, colors, device, smooth_rad=0.003, edge_rad=0.0015, percent=0.05, k=32, chunk=2048,
                          dtype=torch.float32, direct=False):
    """color_edge_points（CPU の KD 木）と同じ結果になるよう GPU で計算する（近傍は近い順に最大 k 点・半径内）．
    CPU 版と結果をそろえるため，重心で中心化した座標で距離を求める（中心化せずカメラ座標のまま行列積で距離を求めると，
    桁落ちで 1 mm 程度の近傍の大小が入れ替わり，選ばれる点がずれる）．中心化すれば座標は ±4 cm 程度なので，
    単精度・行列積でも距離の誤差は 0.1 µm 程度（direct=True で行列積を使わない計算．同じ結果だが遅い）．
    平滑化と勾配で同じ近傍を使うので，近傍探索は 1 回だけ行う．"""
    p = torch.as_tensor(points - points.mean(0), device=device, dtype=dtype)
    c = torch.as_tensor(colors, device=device, dtype=dtype)
    k = min(k, len(p))
    ds, ids = [], []
    for i in range(0, len(p), chunk):
        d = torch.cdist(p[i:i + chunk], p, compute_mode="donot_use_mm_for_euclid_dist" if direct else "use_mm_for_euclid_dist")
        d, idx = torch.topk(d, k, dim=1, largest=False, sorted=True)
        ds.append(d)
        ids.append(idx)
    d, idx = torch.cat(ds), torch.cat(ids)
    w = (d <= smooth_rad).to(dtype).unsqueeze(-1)
    smooth = (c[idx] * w).sum(1) / w.sum(1).clamp(min=1)
    ve = (d <= edge_rad).to(dtype)
    diff = torch.linalg.norm(smooth[idx] - smooth.unsqueeze(1), dim=2)
    score = (diff * ve).sum(1) / ve.sum(1).clamp(min=1)
    n = min(len(p), max(10, int(len(p) * percent)))
    return torch.topk(score, n).indices.cpu().numpy()


def random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


class SphereSeamRegistration:
    def __init__(self, model, grid_deg=0.5, sigma_deg=2.0, n_coarse=40000, n_refine=600, device=None,
                 feature="red", edge_percent=0.05, max_view_deg=70.0, edge_gpu=False, model_edge_all=False):
        """model: (N, 6) 球全体の色付き点群（原点は球の中心）．"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.radius = float(np.linalg.norm(model[:, :3], axis=1).mean())
        self.feature, self.edge_percent, self.max_view = feature, edge_percent, np.deg2rad(max_view_deg)
        # 計測側の色の境界を GPU で求めるか（速いが，実機で精度が下がったため既定は CPU の KD 木）
        self.edge_gpu = edge_gpu
        if feature == "edge":
            # model_edge_all：モデルは割合を決めず，色の境界と判定できる点を全部使う（大津のしきい値）
            seam = model[color_edge_points(model[:, :3], model[:, 3:6],
                                           percent=None if model_edge_all else edge_percent), :3]
        else:
            seam = model[red_mask(model[:, 3:6]), :3]
        self.model_seam = seam
        red = model[red_mask(model[:, 3:6]), :3]
        # 対称性を求める点：赤い縫い目があればそれ（色の境界にはロゴの輪郭が混ざり対称性が崩れるため），なければ模様の点
        self._sym_points = red if len(red) >= 200 else seam
        dirs = seam / np.linalg.norm(seam, axis=1, keepdims=True)
        # 緯度経度の格子の各点から，最も近いモデルの縫い目までの角度 [rad]
        self.nlat, self.nlon = int(round(180 / grid_deg)), int(round(360 / grid_deg))
        lat = (np.arange(self.nlat) + 0.5) / self.nlat * np.pi - np.pi / 2
        lon = (np.arange(self.nlon) + 0.5) / self.nlon * 2 * np.pi - np.pi
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        g = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
        chord = cKDTree(dirs).query(g)[0]
        ang = 2 * np.arcsin(np.clip(chord / 2, 0, 1))
        self.dist_map = torch.tensor(ang.reshape(self.nlat, self.nlon), device=self.device).float()
        self.sigma = np.deg2rad(sigma_deg)
        self.n_coarse, self.n_refine = n_coarse, n_refine
        self.rng = np.random.default_rng(0)
        self.coarse = torch.tensor(random_rotations(n_coarse, self.rng), device=self.device).float()
        self.last_score, self.last_n_seam = float("nan"), 0
        self.last_seam = np.zeros((0, 3))
        # 同じ縫い目に見える姿勢（R と R @ S）のうち，基準の向き canonical_ref（カメラ座標の 3x3）に最も近いものを出す．
        # canonical_syms に縫い目の対称性（seam_symmetries の 3x3 部分）を入れると有効になる
        self.canonical_syms = None
        self.canonical_ref = np.eye(3)
        # canonical_mode："ref"＝基準の向きに最も近いもの / "z_away"＝モデルの z 軸がカメラの奥（+z）を最も向くもの
        # （z の向きがほぼ同じ候補が複数あれば，その中で基準の向きに最も近いもの）
        self.canonical_mode = "ref"
        self.canonical_tol = 0.03

    def _score(self, R, u):
        """R: (K, 3, 3) モデル → カメラ，u: (M, 3) カメラ座標の縫い目の方向．一致度 (K,)．"""
        m = torch.einsum("kji,mj->kmi", R, u)     # R^T u：モデル座標での方向
        lat = torch.asin(m[..., 2].clamp(-1, 1))
        lon = torch.atan2(m[..., 1], m[..., 0])
        i = ((lat + np.pi / 2) / np.pi * self.nlat).long().clamp(0, self.nlat - 1)
        j = ((lon + np.pi) / (2 * np.pi) * self.nlon).long().clamp(0, self.nlon - 1)
        d = self.dist_map[i, j]
        return torch.exp(-(d / self.sigma) ** 2).mean(1)

    def __call__(self, measured):
        """measured: (N, 6) 色付き計測点群（カメラ座標）．モデル → カメラの姿勢 (4, 4)．縫い目が少なければ None．"""
        c = fit_center_known_radius(measured[:, :3], self.radius)
        if self.feature == "edge":
            # 色の境界では，カメラから見て球面が大きく傾いた縁の点は使わない（背景との境目・縁の影が模様に見えるため）．
            # 赤い点では縁の点も使う（赤以外は拾わないので，除くと縫い目の点が減るだけ）
            out = measured[:, :3] - c
            cosv = -(out * measured[:, :3]).sum(1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(measured[:, :3], axis=1) + 1e-12)
            front = measured[cosv > np.cos(self.max_view)]
            if self.edge_gpu and str(self.device).startswith("cuda"):
                seam = front[color_edge_points_gpu(front[:, :3], front[:, 3:6], self.device, percent=self.edge_percent), :3]
            else:
                seam = front[color_edge_points(front[:, :3], front[:, 3:6], percent=self.edge_percent), :3]
        else:
            seam = measured[red_mask(measured[:, 3:6]), :3]
        self.last_n_seam = len(seam)
        self.last_seam = seam   # 計測側の模様の点（カメラ座標．表示用）
        if len(seam) < 20:
            return None
        u = seam - c
        u = torch.tensor(u / np.linalg.norm(u, axis=1, keepdims=True), device=self.device).float()
        if len(u) > 800:   # 縫い目の点が多いときは間引く（一致度の計算を軽くする）
            u = u[torch.randperm(len(u), device=self.device)[:800]]
        scores = torch.cat([self._score(self.coarse[k:k + 4000], u) for k in range(0, self.n_coarse, 4000)])
        top = self.coarse[torch.topk(scores, 8).indices]
        best_R, best_s = top[0], float(scores.max())
        for step in (6.0, 3.0, 1.5, 0.7, 0.3):   # 上位の候補のまわりを段階的に細かく探す [deg]
            pert = Rotation.from_rotvec(self.rng.normal(0, np.deg2rad(step), (self.n_refine, 3))).as_matrix()
            pert = torch.tensor(pert, device=self.device).float()
            cand = torch.cat([top, torch.einsum("kij,pjl->kpil", top, pert).reshape(-1, 3, 3)])
            s = self._score(cand, u)
            idx = torch.topk(s, 8).indices
            top = cand[idx]
            if float(s[idx[0]]) > best_s:
                best_s, best_R = float(s[idx[0]]), cand[idx[0]]
        self.last_score = best_s
        T = np.eye(4)
        R = best_R.cpu().numpy().astype(np.float64)
        if self.canonical_syms:
            cands = [R] + [R @ S for S in self.canonical_syms]
            # 同じ縫い目に見えると確かめられた候補（一致度が最良から canonical_tol 以内）だけから選ぶ
            sc = self._score(torch.tensor(np.array(cands), device=self.device).float(), u).cpu().numpy()
            cands = [C for C, v in zip(cands, sc) if v >= sc.max() - self.canonical_tol]
            if self.canonical_mode == "z_away":
                zmax = max(C[2, 2] for C in cands)   # モデルの z 軸（C の第 3 列）とカメラの奥（+z）の内積
                cands = [C for C in cands if C[2, 2] >= zmax - 0.05]
            R = min(cands, key=lambda C: Rotation.from_matrix(C @ self.canonical_ref.T).magnitude())
        T[:3, :3] = R
        T[:3, 3] = c
        return T

    def seam_symmetries(self, max_err_deg=1.5, min_axis_sep_deg=45.0, n_axes=3000):
        """縫い目が自分に重なる 180° 回転（モデル座標の 4x4 のリスト）．同じ縫い目に見える姿勢の間の飛びを
        カルマンフィルタで外れ値として捨てないため，また同じ縫い目に見える姿勢から 1 つを選ぶために使う．
        重なりの誤差（回した縫い目から元の縫い目までの角度の中央値）の小さい順に，すでに選んだ軸から min_axis_sep_deg 以上
        離れた軸だけを選ぶ（本当の対称軸の近くの軸も誤差が小さく出るので，近い軸を別の対称性として数えないため．
        YCB のボールでは本当の 3 軸が約 1.0°，それ以外は 2.0° 以上）．"""
        dirs = self._sym_points / np.linalg.norm(self._sym_points, axis=1, keepdims=True)
        tree = cKDTree(dirs)
        g = np.arange(n_axes) + 0.5
        phi, th = np.arccos(1 - 2 * g / n_axes), np.pi * (1 + 5 ** 0.5) * g
        axes = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        axes = axes[axes[:, 2] >= 0]
        errs = np.array([np.median(tree.query(dirs @ Rotation.from_rotvec(np.pi * a).as_matrix().T)[0]) for a in axes])
        chosen = []
        for i in np.argsort(errs):
            if errs[i] > np.deg2rad(max_err_deg):
                break
            if all(np.rad2deg(np.arccos(min(1.0, abs(axes[i] @ b)))) >= min_axis_sep_deg for b in chosen):
                chosen.append(axes[i])
        out = []
        for a in chosen:
            S = np.eye(4)
            S[:3, :3] = Rotation.from_rotvec(np.pi * a).as_matrix()
            out.append(S)
        return out
