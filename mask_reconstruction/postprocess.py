"""改善5：統合マスクの軽量な後処理（ネットワークなし）．

各マスクに closing と穴埋めを行い，新しく加わる画素のうち，
  - 深度が有効で，マスク内の近傍画素の平均深度との差が post_depth_thresh 未満の画素
  - 深度が欠損でも，マスクに囲まれた穴の画素
だけを採用する（深度が不連続な隣の物体・背景へははみ出さない）．他のマスクに属する画素は加えない．
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .config import Config


def close_masks(masks, pts, valid_idx, v_u, v_v, cfg: Config):
    if not masks:
        return masks
    shape = masks[0].shape
    Z = np.zeros(shape, np.float32)
    Z[v_v, v_u] = pts[valid_idx, 2]
    valid = Z > 0
    k = cfg.post_close_ksize
    kernel = np.ones((k, k), np.uint8)
    owner = np.sum(masks, axis=0)
    out = []
    for m in masks:
        mu = m.astype(np.uint8)
        closed = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, kernel) > 0
        holes = binary_fill_holes(closed) & ~closed
        cand = (closed | holes) & ~m & (owner == 0)
        # マスク内の有効深度の局所平均（窓 k）
        w = (m & valid).astype(np.float32)
        s = cv2.boxFilter(Z * w, -1, (2 * k + 1, 2 * k + 1), normalize=False)
        c = cv2.boxFilter(w, -1, (2 * k + 1, 2 * k + 1), normalize=False)
        local = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
        cont = valid & (c > 0) & (np.abs(Z - local) < cfg.post_depth_thresh)
        add = cand & (cont | (holes & ~valid))
        out.append(m | add)
    return out
