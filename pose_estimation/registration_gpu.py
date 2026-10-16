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


def _nearest(x, tmpl, tmpl_sq):
    """x (P, 3) の各点から tmpl (M, 3) への最近傍距離とインデックス．

    |x − t|² = |x|² + |t|² − 2 x·t の |x|² は行ごとに一定なので，|t|² − 2 x·t（行列積1回）の最小を取ってから足す．
    """
    d2, idx = torch.addmm(tmpl_sq[None], x, tmpl.T, beta=1.0, alpha=-2.0).min(dim=1)
    return (d2 + (x * x).sum(dim=1)).clamp(min=0).sqrt(), idx


def _per_part_sum(v, bid, B):
    """点ごとの値 v (P, ...) を部品ごとに合計する → (B, ...)．"""
    out = torch.zeros((B, *v.shape[1:]), dtype=v.dtype, device=v.device)
    return out.index_add_(0, bid, v)


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
        self.prune = False   # True: 収束した部品の点を計算から外す（同期が増え，この規模ではかえって遅い）

    def __call__(self, model_pts, measured_pts) -> RegistrationResult:
        return self.batch(model_pts, [measured_pts])[0]

    @torch.no_grad()
    def batch(self, model_pts, measured_list):
        """model_pts: (M, 3)，measured_list: [(N_i, 3)] → [RegistrationResult]"""
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

        # --- 1. SVD による初期回転（計測 → モデル） ---
        basis_t = torch.linalg.svd(tmpl_c, full_matrices=False)[2].T                 # (3, 3)
        cov = _per_part_sum(Xc[:, :, None] * Xc[:, None, :], bid, B)                  # (B, 3, 3)
        # 中心化した点群の右特異ベクトル ＝ 共分散の固有ベクトル（特異値の大きい順にそろえる）
        _, _, Vt = torch.linalg.svd(cov)
        basis_s = Vt.transpose(1, 2)
        best_fit = torch.full((B,), -1.0, device=dev)
        best_R = torch.eye(3, device=dev).repeat(B, 1, 1)
        best_flip = torch.zeros(B, dtype=torch.long, device=dev)
        for k, f in enumerate(self.flips):
            R = basis_t[None] @ (basis_s * f).transpose(1, 2)                        # (B, 3, 3)
            ok = torch.linalg.det(R) > 0
            d, _ = _nearest(torch.einsum("pij,pj->pi", R[bid], Xc), tmpl_c, tmpl_sq)
            fit = _per_part_sum((d < cfg.flip_check_threshold).float(), bid, B) / n_pts
            better = ok & (fit > best_fit)                                            # 同点は先の候補（CPU 版の max と同じ）
            best_fit = torch.where(better, fit, best_fit)
            best_R = torch.where(better[:, None, None], R, best_R)
            best_flip = torch.where(better, torch.full_like(best_flip, k), best_flip)

        # --- 2. ICP（全部品を同時に．収束した部品の点は計算から外す） ---
        R_tot, t_tot = best_R.clone(), torch.zeros(B, 3, device=dev)
        fit = torch.zeros(B, device=dev)
        rmse = torch.zeros(B, device=dev)

        def evaluate(sel):
            """sel の点について，現在の姿勢での対応点と部品ごとの fitness・RMSE．"""
            b = bid[sel]
            x = torch.einsum("pij,pj->pi", R_tot[b], Xc[sel]) + t_tot[b]
            d, idx = _nearest(x, tmpl_c, tmpl_sq)
            inl = (d < cfg.icp_threshold).float()
            n_in = _per_part_sum(inl, b, B)
            f = n_in / n_pts
            r = torch.sqrt(_per_part_sum(d * d * inl, b, B) / n_in.clamp(min=1)) * (n_in > 0)
            return x, idx, inl, f, r

        active = torch.ones(B, dtype=torch.bool, device=dev)
        sel = torch.ones(len(X), dtype=torch.bool, device=dev)
        x, idx, inl, fit, rmse = evaluate(sel)
        for _ in range(cfg.icp_max_iteration):
            b = bid[sel]
            dR, dt = _kabsch(x, tmpl_c[idx], inl, b, B)
            R_tot = torch.where(active[:, None, None], dR @ R_tot, R_tot)
            t_tot = torch.where(active[:, None], (dR @ t_tot[..., None]).squeeze(-1) + dt, t_tot)
            x, idx, inl, fit2, rmse2 = evaluate(sel)
            conv = ((fit2 - fit).abs() < _EPS_REL) & ((rmse2 - rmse).abs() < _EPS_REL)
            fit = torch.where(active, fit2, fit)
            rmse = torch.where(active, rmse2, rmse)
            new_active = active & ~conv
            if not bool(new_active.any()):
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
