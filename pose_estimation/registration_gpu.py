"""SVD 初期姿勢 + Point-to-Point ICP を，複数の部品について GPU（torch）でまとめて計算する．

registration_svd.SVDICPRegistration（ICP は Open3D・CPU で1部品ずつ）と同じ手順：
  1. モデル・計測点群をそれぞれ重心で中心化し，SVD で主軸基底を求める
  2. 軸の符号反転 8 パターンのうち右手系のものを試し，fitness（flip_check_threshold 以内の割合）最大を初期回転とする
  3. Point-to-Point ICP（対応点は最近傍のうち icp_threshold 以内．Open3D と同じく，
     fitness と inlier RMSE の変化がともに 1e-6 未満になるか icp_max_iteration 回で終了）
全部品の点を1列に並べて部品番号で区別し，1回のテンソル計算でまとめて処理する．
数値は float32 なので，Open3D（float64）とはわずかに異なる．
"""
import numpy as np
import torch

from .config import PoseConfig
from .registration_svd import FLIP_PATTERNS, RegistrationResult

_EPS_REL = 1e-6   # Open3D の ICPConvergenceCriteria の既定値（relative_fitness, relative_rmse）


_HIDDEN = 1.0e4   # 見えないモデル点への罰則 [m²]（対応点の閾値よりずっと大きい）


def _nearest(x, tmpl, tmpl_sq, hidden=None):
    """x (P, 3) の各点から tmpl (M, 3) への最近傍距離とインデックス．

    |x − t|² = |x|² + |t|² − 2 x·t の |x|² は行ごとに一定なので，|t|² − 2 x·t（行列積1回）の最小を取ってから足す．
    """
    if hidden is None:
        d2 = torch.addmm(tmpl_sq[None], x, tmpl.T, beta=1.0, alpha=-2.0)
    else:
        # hidden = (pen (B, M)：見えないモデル点に _HIDDEN，見える点に 0，bid (P,))．
        # 計測点に部品番号の one-hot を足し，モデル側に罰則の行を足して，距離と罰則を1回の行列積で求める
        pen, bid = hidden
        onehot = torch.nn.functional.one_hot(bid, pen.shape[0]).to(x.dtype)
        d2 = torch.addmm(tmpl_sq[None], torch.cat([x, onehot], dim=1), torch.cat([-2.0 * tmpl.T, pen], dim=0))
    d2, idx = d2.min(dim=1)
    return (d2 + (x * x).sum(dim=1)).clamp(min=0).sqrt(), idx


def _per_part_sum(v, bid, B):
    """点ごとの値 v (P, ...) を部品ごとに合計する → (B, ...)．

    部品の割り当て表（B×P の 0/1）との行列積で求める．index_add（並列の足し込み）は足す順番が毎回変わり，
    結果が実行ごとにわずかに揺れて，初期候補の選び方まで入れ替わることがあったため．
    """
    oh = (bid[None, :] == torch.arange(B, device=bid.device)[:, None]).to(v.dtype)
    return (oh @ v.reshape(v.shape[0], -1)).reshape(B, *v.shape[1:])


def _kabsch(x, y, w, bid, B):
    """重み w (P,) 付きで x → y の剛体変換 (R, t) を部品ごとに求める．"""
    wsum = _per_part_sum(w, bid, B).clamp(min=1e-9)
    cx = _per_part_sum(x * w[:, None], bid, B) / wsum[:, None]
    cy = _per_part_sum(y * w[:, None], bid, B) / wsum[:, None]
    xc, yc = x - cx[bid], y - cy[bid]
    H = _per_part_sum((xc * w[:, None])[:, :, None] * yc[:, None, :], bid, B)        # (B, 3, 3)
    U, _, Vt = torch.linalg.svd(H)
    D = torch.ones(B, 3, device=x.device)
    D[:, 2] = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    R = Vt.transpose(1, 2) @ torch.diag_embed(D) @ U.transpose(1, 2)
    t = cy - (R @ cx[..., None]).squeeze(-1)
    return R, t


class GPUBatchRegistration:
    def __init__(self, cfg: PoseConfig = PoseConfig(), device=None):
        if cfg.hypotheses != "original" or cfg.icp_all_candidates:
            raise NotImplementedError("GPU 版は既定の候補の作り方（hypotheses='original'，ICP は最良候補のみ）だけ対応")
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flips = torch.tensor(FLIP_PATTERNS, dtype=torch.float32, device=self.device)
        self.check_every = 5   # 収束の確認（同期）を何回に1回行うか
        self.prune = False   # True: 収束した部品の点を計算から外す（同期が増え，この規模ではかえって遅い）

    def __call__(self, model_pts, measured_pts) -> RegistrationResult:
        return self.batch(model_pts, [measured_pts])[0]

    def _model_normals(self, model_pts):
        """モデル点の外向き法線（向きを表面に沿って伝えてそろえ，重心から外向きが多数派になるようにする）．"""
        key = (id(model_pts), len(model_pts))
        if getattr(self, "_normals_key", None) != key:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts[:, :3]))
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
            pcd.orient_normals_consistent_tangent_plane(20)
            n = np.asarray(pcd.normals)
            if np.mean(np.sum((model_pts[:, :3] - model_pts[:, :3].mean(axis=0)) * n, axis=1) > 0) < 0.5:
                n = -n
            self._normals_key = key
            self._normals = torch.as_tensor(n, dtype=torch.float32, device=self.device)
        return self._normals

    @torch.no_grad()
    def batch(self, model_pts, measured_list, init=None):
        """model_pts: (M, 3)，measured_list: [(N_i, 3)] → [RegistrationResult]

        init: 部品ごとの初期姿勢（モデル → カメラの 4x4）のリスト．与えると SVD の初期候補を省き，そこから ICP する．
        """
        if not measured_list:
            return []
        dev, cfg = self.device, self.cfg
        tmpl = torch.as_tensor(model_pts[:, :3], dtype=torch.float32, device=dev)
        tmpl_mean = tmpl.mean(dim=0)
        tmpl_c = tmpl - tmpl_mean
        tmpl_sq = (tmpl_c * tmpl_c).sum(dim=1)
        # 全部品の点を1列に並べ，部品番号 bid で区別する（点数をそろえて詰めると無駄な計算が増えるため）
        B = len(measured_list)
        X = torch.as_tensor(np.concatenate([p[:, :3] for p in measured_list]), dtype=torch.float32, device=dev)
        counts = torch.tensor([len(p) for p in measured_list], device=dev)
        bid = torch.repeat_interleave(torch.arange(B, device=dev), counts)
        n_pts = counts.float()
        src_mean = _per_part_sum(X, bid, B) / n_pts[:, None]
        Xc = X - src_mean[bid]

        visible = None
        if cfg.icp_visible_only:
            Nm = self._model_normals(model_pts)
            qn = (Nm * tmpl_c).sum(dim=1)

            def visible(R, t):
                """姿勢 (R, t)（中心化した計測 → 中心化したモデル）でカメラを向いているモデル点 (B, M)．"""
                cam = (R @ (-src_mean)[..., None]).squeeze(-1) + t      # カメラ位置（モデル座標）
                return cam @ Nm.T - qn[None] > 0

            def visible_centroid(R, t):
                v = visible(R, t).float()
                return (v @ tmpl_c) / v.sum(dim=1, keepdim=True).clamp(min=1)

        if init is not None:
            # モデル → カメラ の姿勢 (Rm, tm) を，中心化した計測 → 中心化したモデル の (R, t) に直す
            Ti = torch.as_tensor(np.stack(init), dtype=torch.float32, device=dev)
            Rm, tm = Ti[:, :3, :3], Ti[:, :3, 3]
            R0 = Rm.transpose(1, 2)
            t0 = (R0 @ (src_mean - tm)[..., None]).squeeze(-1) - tmpl_mean[None]
            return self._icp_and_results(B, bid, Xc, n_pts, src_mean, tmpl_c, tmpl_sq, tmpl_mean, visible,
                                         R0, t0, torch.zeros(B, device=dev), torch.zeros(B, dtype=torch.long, device=dev))

        # --- 1. SVD による初期回転（計測 → モデル） ---
        basis_t = torch.linalg.svd(tmpl_c, full_matrices=False)[2].T                 # (3, 3)
        cov = _per_part_sum(Xc[:, :, None] * Xc[:, None, :], bid, B)                  # (B, 3, 3)
        # 中心化した点群の右特異ベクトル ＝ 共分散の固有ベクトル（特異値の大きい順にそろえる）
        _, _, Vt = torch.linalg.svd(cov)
        basis_s = Vt.transpose(1, 2)
        best_fit = torch.full((B,), -1.0, device=dev)
        best_R = torch.eye(3, device=dev).repeat(B, 1, 1)
        best_flip = torch.zeros(B, dtype=torch.long, device=dev)
        best_t = torch.zeros(B, 3, device=dev)
        for k, f in enumerate(self.flips):
            R = basis_t[None] @ (basis_s * f).transpose(1, 2)                        # (B, 3, 3)
            ok = torch.linalg.det(R) > 0
            t = torch.zeros(B, 3, device=dev)
            allowed = None
            if visible is not None:   # 計測点の重心（見えている面の上）を，見える面の重心に合わせる
                t = visible_centroid(R, t)
                t = visible_centroid(R, t)
                allowed = ((~visible(R, t)).float() * _HIDDEN, bid)
            d, _ = _nearest(torch.einsum("pij,pj->pi", R[bid], Xc) + t[bid], tmpl_c, tmpl_sq, allowed)
            fit = _per_part_sum((d < cfg.flip_check_threshold).float(), bid, B) / n_pts
            better = ok & (fit > best_fit)                                            # 同点は先の候補（CPU 版の max と同じ）
            best_fit = torch.where(better, fit, best_fit)
            best_R = torch.where(better[:, None, None], R, best_R)
            best_t = torch.where(better[:, None], t, best_t)
            best_flip = torch.where(better, torch.full_like(best_flip, k), best_flip)

        return self._icp_and_results(B, bid, Xc, n_pts, src_mean, tmpl_c, tmpl_sq, tmpl_mean, visible,
                                     best_R, best_t, best_fit, best_flip)

    def _icp_and_results(self, B, bid, Xc, n_pts, src_mean, tmpl_c, tmpl_sq, tmpl_mean, visible,
                         best_R, best_t, best_fit, best_flip):
        """初期姿勢 (best_R, best_t) から全部品の ICP を行い，結果を返す．"""
        dev, cfg = self.device, self.cfg
        # --- 2. ICP（全部品を同時に．収束した部品の点は計算から外す） ---
        R_tot, t_tot = best_R.clone(), best_t.clone()
        fit = torch.zeros(B, device=dev)
        rmse = torch.zeros(B, device=dev)

        def evaluate(sel):
            """sel の点について，現在の姿勢での対応点と部品ごとの fitness・RMSE．"""
            # sel が None なら全点（真偽値での取り出しは GPU との同期を起こすので，点を外さないときは使わない）
            b, Xs = (bid, Xc) if sel is None else (bid[sel], Xc[sel])
            x = torch.einsum("pij,pj->pi", R_tot[b], Xs) + t_tot[b]
            allowed = ((~visible(R_tot, t_tot)).float() * _HIDDEN, b) if visible is not None else None
            d, idx = _nearest(x, tmpl_c, tmpl_sq, allowed)
            inl = (d < cfg.icp_threshold).float()
            n_in = _per_part_sum(inl, b, B)
            f = n_in / n_pts
            r = torch.sqrt(_per_part_sum(d * d * inl, b, B) / n_in.clamp(min=1)) * (n_in > 0)
            return x, idx, inl, f, r

        active = torch.ones(B, dtype=torch.bool, device=dev)
        sel = None
        x, idx, inl, fit, rmse = evaluate(sel)
        for it in range(cfg.icp_max_iteration):
            b = bid if sel is None else bid[sel]
            dR, dt = _kabsch(x, tmpl_c[idx], inl, b, B)
            R_tot = torch.where(active[:, None, None], dR @ R_tot, R_tot)
            t_tot = torch.where(active[:, None], (dR @ t_tot[..., None]).squeeze(-1) + dt, t_tot)
            x, idx, inl, fit2, rmse2 = evaluate(sel)
            conv = ((fit2 - fit).abs() < _EPS_REL) & ((rmse2 - rmse).abs() < _EPS_REL)
            fit = torch.where(active, fit2, fit)
            rmse = torch.where(active, rmse2, rmse)
            new_active = active & ~conv
            # 全部品が収束したかの確認（CPU と GPU の同期）は check_every 回に1回だけ行う．
            # 収束した部品はそれ以上更新しないので，確認が遅れても結果は変わらない
            if (it + 1) % self.check_every == 0 and not bool(new_active.any()):
                break
            if self.prune and not torch.equal(new_active, active):   # 収束した部品の点を外す
                keep = new_active[b]
                x, idx, inl = x[keep], idx[keep], inl[keep]
                sel = new_active[bid]
            active = new_active

        # 中心化を戻して 計測 → モデル の変換を作り，逆行列で物体姿勢にする
        t_f = t_tot + tmpl_mean[None] - (R_tot @ src_mean[..., None]).squeeze(-1)
        T = torch.eye(4, device=dev).repeat(B, 1, 1)
        T[:, :3, :3], T[:, :3, 3] = R_tot, t_f
        T_model_to_cam = torch.linalg.inv(T.double()).cpu().numpy()
        best_fit, fit, rmse, best_flip = (v.cpu().numpy() for v in (best_fit, fit, rmse, best_flip))
        return [RegistrationResult(T_model_to_cam=T_model_to_cam[b], svd_fitness=float(best_fit[b]),
                                   icp_fitness=float(fit[b]), icp_rmse=float(rmse[b]),
                                   flip=list(FLIP_PATTERNS[int(best_flip[b])]))
                for b in range(B)]
