"""改善3：深度画像（organized point cloud）上の法線と，断片境界の凸／凹（LCCP 型）の判定．

法線：画素 (y, x) で，左右・上下に normal_window//2 離れた点の差分ベクトルの外積．カメラ側に向ける．
      中心または差分に使う点が欠損なら無効．
境界：隣接する断片 u, v の接触境界（u ∩ dilate(v)）から最大 K 点をサンプリングし，
      各サンプル b と v 側の最近傍画素 q を結ぶ方向に，u 側へ inset px（b から），v 側へ inset px（q から）
      入った点 x_u, x_v と法線 n_u, n_v を取る．
      d = (x_u − x_v)/|x_u − x_v|．法線のなす角 < flat_angle → 平坦（凸扱い），
      (n_u − n_v)·d > 0 → 凸，それ以外 → 凹．
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import Config


def point_image(pts, valid_idx, v_u, v_v, shape):
    P = np.zeros((*shape, 3))
    P[v_v, v_u] = pts[valid_idx]
    return P, P[:, :, 2] > 0


def organized_normals(P, V, window):
    """深度画像上の差分の外積による法線 (H, W, 3) と有効マスク．"""
    h = max(window // 2, 1)
    N = np.zeros_like(P)
    ok = np.zeros(V.shape, bool)
    dx = P[h:-h, 2 * h:] - P[h:-h, :-2 * h]
    dy = P[2 * h:, h:-h] - P[:-2 * h, h:-h]
    n = np.cross(dx, dy)
    valid = (V[h:-h, h:-h] & V[h:-h, 2 * h:] & V[h:-h, :-2 * h] & V[2 * h:, h:-h] & V[:-2 * h, h:-h])
    norm = np.linalg.norm(n, axis=2)
    valid &= norm > 0
    n = n / np.where(norm > 0, norm, 1)[..., None]
    # カメラ（原点）側へ向ける：n·p < 0
    flip = np.sum(n * P[h:-h, h:-h], axis=2) > 0
    n[flip] *= -1
    N[h:-h, h:-h] = np.where(valid[..., None], n, 0)
    ok[h:-h, h:-h] = valid
    return N, ok


class BoundaryConvexity:
    def __init__(self, masks, pts, valid_idx, v_u, v_v, cfg: Config):
        self.masks = masks
        self.cfg = cfg
        self.P, V = point_image(pts, valid_idx, v_u, v_v, masks[0].shape)
        self.N, self.Nok = organized_normals(self.P, V, cfg.normal_window)
        self._nearest = {}

    def nearest_in(self, k):
        """各画素から断片 k 内の最近傍画素の座標 (2, H, W)．"""
        if k not in self._nearest:
            _, idx = distance_transform_edt(~self.masks[k], return_indices=True)
            self._nearest[k] = idx
        return self._nearest[k]

    def samples(self, k, l, dilated):
        """u=k 側から見た境界サンプルの (x_u, x_v) 画素座標．"""
        band = self.masks[k] & dilated[l] & ~self.masks[l]
        ys, xs = np.nonzero(band)
        if len(ys) == 0:
            return np.zeros((0, 2), int), np.zeros((0, 2), int)
        sel = np.unique(np.linspace(0, len(ys) - 1, min(self.cfg.convex_samples, len(ys))).round().astype(int))
        b = np.stack([ys[sel], xs[sel]], axis=1)
        near = self.nearest_in(l)
        q = np.stack([near[0][b[:, 0], b[:, 1]], near[1][b[:, 0], b[:, 1]]], axis=1)
        t = (b - q).astype(float)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        xu = np.round(b + self.cfg.convex_inset * t).astype(int)
        xv = np.round(q - self.cfg.convex_inset * t).astype(int)
        H, W = self.masks[k].shape
        ok = np.all((xu >= 0) & (xu < (H, W)) & (xv >= 0) & (xv < (H, W)), axis=1)
        return xu[ok], xv[ok]

    def pair(self, k, l, dilated):
        """(concave_ratio, concave_angle [deg], 有効サンプル数)．"""
        xu, xv = self.samples(k, l, dilated)
        if len(xu) == 0:
            return np.nan, np.nan, 0
        mu, mv = self.masks[k], self.masks[l]
        ok = mu[xu[:, 0], xu[:, 1]] & mv[xv[:, 0], xv[:, 1]] \
            & self.Nok[xu[:, 0], xu[:, 1]] & self.Nok[xv[:, 0], xv[:, 1]]
        xu, xv = xu[ok], xv[ok]
        if len(xu) == 0:
            return np.nan, np.nan, 0
        pu, pv = self.P[xu[:, 0], xu[:, 1]], self.P[xv[:, 0], xv[:, 1]]
        nu, nv = self.N[xu[:, 0], xu[:, 1]], self.N[xv[:, 0], xv[:, 1]]
        d = pu - pv
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip(np.sum(nu * nv, axis=1), -1, 1)))
        convex = (ang < self.cfg.flat_angle) | (np.sum((nu - nv) * d, axis=1) > 0)
        concave = ~convex
        ratio = float(concave.mean())
        angle = float(np.median(ang[concave])) if concave.any() else 0.0
        return ratio, angle, int(len(xu))
