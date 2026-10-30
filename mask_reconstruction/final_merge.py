"""統合後の仕上げ（Config.final_merge）：1つの部品が2つ以上のマスクに分かれたものを，もう一度まとめる．

統合が終わったマスクどうしで，次をすべて満たす組を，深度差と法線の角度差の小さい順に1つずつまとめる
（まとめられる組がなくなるまで繰り返す）．
  - 2D で隣接している（統合と同じ膨張カーネル）
  - 境界の深度差 < bound_depth_thresh，境界両側の平均法線の角度差 < normal_angle_thresh（統合と同じ基準）
  - まとめても部品1個の大きさに収まる：主軸方向の長さ（3方向，[p, 100-p] パーセンタイル幅）と実面積
    Σz²/(fx·fy) が，CAD モデルから求めた値（cad_prior.py）の (1+final_merge_margin) 倍以内
部品の形に固有の値は使わない．
"""
import numpy as np

from . import cpp_kernels as ck
from .cad_prior import load_cad_prior
from .config import Config
from .merging import depth_image


def _extent(p, q):
    c = p - p.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ Vt.T
    ext = np.percentile(proj, 100 - q, axis=0) - np.percentile(proj, q, axis=0)
    return np.sort(np.concatenate([ext, np.zeros(3 - len(ext))]))[::-1]


def final_merge_masks(masks, pts, valid_idx, v_u, v_v, cfg: Config, cam):
    if len(masks) < 2:
        return masks
    from .graph_merging import _normal_image
    shape = masks[0].shape
    Z = depth_image(pts, valid_idx, v_u, v_v, shape)
    N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg)
    I = ck.index_image(valid_idx, v_u, v_v, shape)
    prior = load_cad_prior(cfg)
    lim = 1.0 + cfg.final_merge_margin
    L_lim = lim * np.array(prior.L)
    A_lim = lim * prior.A
    area_img = None
    if cam is not None:
        area_img = Z ** 2 / (cam[0] * cam[1])
    masks = [m.copy() for m in masks]

    def fits(m):
        idx = ck.gather_indices(ck.stack_masks([m]), I)[0]
        if len(idx) < 4:
            return True
        p = pts[valid_idx[idx]]
        if len(p) > cfg.max_points_per_fragment:
            p = p[np.random.default_rng(0).choice(len(p), cfg.max_points_per_fragment, replace=False)]
        if np.any(_extent(p, cfg.extent_percentile) > L_lim):
            return False
        return area_img is None or float(area_img[m].sum()) <= A_lim

    rejected = set()   # 大きさで断られた組（マスクが変わらない限り，もう一度確かめない）
    while True:
        adjacent, gap, ang, _, _ = ck.pair_features(ck.stack_masks(masks), cfg.adjacency_kernel, Z, N)
        cands = []
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                if not adjacent[a, b] or np.isnan(gap[a, b]) or np.isnan(ang[a, b]):
                    continue
                if gap[a, b] >= cfg.bound_depth_thresh or ang[a, b] >= cfg.normal_angle_thresh:
                    continue
                cands.append((gap[a, b] / cfg.bound_depth_thresh + ang[a, b] / cfg.normal_angle_thresh, a, b))
        merged = False
        for _, a, b in sorted(cands):
            key = (masks[a].tobytes().__hash__(), masks[b].tobytes().__hash__())
            if key in rejected:
                continue
            u = masks[a] | masks[b]
            if fits(u):
                masks[a] = u
                del masks[b]
                merged = True
                break
            rejected.add(key)
        if not merged:
            return masks
