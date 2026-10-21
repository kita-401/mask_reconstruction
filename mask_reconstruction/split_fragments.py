"""統合前の断片の切り分け（Config.split_fragments）：2つの部品にまたがった FastSAM の断片を，部品ごとに分ける．

断片の中で，深度か法線が途切れる所を境界とみなし，境界で区切られた領域ごとに別の断片にする．
  - 途切れ：上下左右に隣り合う画素で，深度差 ≥ split_depth_thresh，または法線の角度差 ≥ split_normal_thresh
    （どちらも両方の画素で値があるときだけ判定する．深度の欠けた画素は途切れとみなさない）
  - 領域：断片から途切れの画素を除いて 4 近傍でつながる領域．split_min_area [px] 未満の切れ端は捨てる
  - 途切れの画素は，隣接する一番大きな領域に戻す（境界の細い線が抜けたままにならないように）
部品の形に固有の値は使わない．
"""
import cv2
import numpy as np

from .config import Config
from .merging import depth_image


def edge_map(Z, N, cfg: Config):
    """深度か法線が途切れる画素（H×W bool）．"""
    valid = Z > 0
    nok = np.any(N != 0, axis=2)
    cos_t = np.cos(np.radians(cfg.split_normal_thresh))
    edge = np.zeros(Z.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, Z.shape[0] - dy), slice(0, Z.shape[1] - dx))
        b = (slice(dy, Z.shape[0]), slice(dx, Z.shape[1]))
        both = valid[a] & valid[b]
        jump = both & (np.abs(Z[a] - Z[b]) >= cfg.split_depth_thresh)
        nb = nok[a] & nok[b]
        crease = nb & (np.sum(N[a] * N[b], axis=2) <= cos_t)
        e = jump | crease
        edge[a] |= e
        edge[b] |= e
    return edge


def split_fragments(masks, pts, valid_idx, v_u, v_v, cfg: Config):
    if not masks:
        return masks
    from .graph_merging import _normal_image
    shape = masks[0].shape
    Z = depth_image(pts, valid_idx, v_u, v_v, shape)
    N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg)
    edge = edge_map(Z, N, cfg)
    out = []
    kernel = np.ones((3, 3), np.uint8)
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = m[y0:y1, x0:x1]
        inner = (sub & ~edge[y0:y1, x0:x1]).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=4)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= cfg.split_min_area]
        if len(keep) <= 1:
            out.append(m)      # 切れなかった（または大きな領域が1つだけ）：元の断片のまま
            continue
        # 途切れの画素と捨てた切れ端の画素を，隣接する一番大きな領域に戻す
        big = max(keep, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        rest = sub & ~np.isin(lab, keep)
        for k in keep:
            part = lab == k
            if k == big:
                part = part | (rest & (cv2.dilate(part.astype(np.uint8), kernel) > 0))
            full = np.zeros(shape, bool)
            full[y0:y1, x0:x1] = part
            out.append(full)
    return out
