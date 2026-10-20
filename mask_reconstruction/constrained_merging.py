"""改善2・3：CADサイズの cannot-link 制約付き平均連結クラスタリングと，凸性を使った距離行列．

- pair_table      : 断片対ごとの特徴（基本コスト・隣接・段差・凸性）
- distance_from_table : 特徴から距離行列を作る（フラグがすべてオフなら merging.distance_matrix と同一）
- SizeChecker     : 統合後クラスタが CAD の大きさ（L1〜L3, D_cad, A_cad）×(1+m) を超えるか
- constrained_average_linkage : 平均連結の凝集型クラスタリング（制約なしなら scipy の linkage+fcluster と同じ分割）

統合の経過（制約で止めた回数・条件）は LAST_INFO と logger に残す．
"""
import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .cad_prior import load_cad_prior, max_distance
from .config import Config
from .merging import boundary_depth_gap, distance_matrix, fragment_pca

log = logging.getLogger(__name__)
LAST_INFO = {}

SIZE_KEYS = ("L1", "L2", "L3", "D", "A")


@dataclass
class PairTable:
    base: np.ndarray       # (n, n) 重心・主軸による距離（7/31 の式，ペナルティなし）
    adjacent: np.ndarray   # (n, n) bool
    gap: np.ndarray        # (n, n) 境界の深度差 [m]（計算できなければ nan）
    cratio: np.ndarray     # (n, n) concave_ratio（未計算は nan）
    cangle: np.ndarray     # (n, n) concave_angle [deg]（凹サンプルがなければ 0，未計算は nan）
    n_samples: np.ndarray  # (n, n) 凸性の有効サンプル数


def pair_table(masks, pts, valid_idx, v_u, v_v, cfg: Config, convexity=False, cam=None):
    n = len(masks)
    centers, axes = fragment_pca(masks, pts, valid_idx, v_u, v_v)
    kernel = np.ones((cfg.adjacency_kernel, cfg.adjacency_kernel), np.uint8)
    dilated = [cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in masks]
    T = PairTable(base=np.zeros((n, n)), adjacent=np.zeros((n, n), bool), gap=np.full((n, n), np.nan),
                  cratio=np.full((n, n), np.nan), cangle=np.full((n, n), np.nan),
                  n_samples=np.zeros((n, n), int))
    conv = None
    if convexity:
        from .convexity import BoundaryConvexity
        conv = BoundaryConvexity(masks, pts, valid_idx, v_u, v_v, cfg)
    for k in range(n):
        for l in range(k + 1, n):
            d_vec = centers[l] - centers[k]
            d_norm = np.linalg.norm(d_vec)
            d_hat = d_vec / d_norm if d_norm > 1e-6 else axes[k]
            angle_diff = 1.0 - np.abs(np.dot(axes[k], axes[l]))
            axis_align = 1.0 - max(abs(np.dot(d_hat, axes[k])), abs(np.dot(d_hat, axes[l])))
            T.base[k, l] = (d_norm * cfg.w_center
                            + (angle_diff * cfg.w_angle + axis_align * cfg.w_axis_align) * cfg.L_cad)
            if np.any(dilated[k] & masks[l]):
                T.adjacent[k, l] = True
                g = boundary_depth_gap(masks[k], masks[l], dilated[k], dilated[l], pts, valid_idx, v_u, v_v)
                if g is not None:
                    T.gap[k, l] = g
                if conv is not None:
                    r, a, ns = conv.pair(k, l, dilated)
                    T.cratio[k, l], T.cangle[k, l], T.n_samples[k, l] = r, a, ns
    for M in (T.base, T.adjacent, T.gap, T.cratio, T.cangle, T.n_samples):
        M[np.tril_indices(n, -1)] = M.T[np.tril_indices(n, -1)]
    return T


def concave_angle_thresh(cfg: Config):
    if cfg.concave_angle_thresh >= 0:
        return cfg.concave_angle_thresh
    return load_cad_prior(cfg).theta_max + cfg.concave_angle_margin


def pair_rule(T: PairTable, k, l, cfg: Config):
    """断片対に適用される判定（分析・失敗の内訳用）．"""
    if not T.adjacent[k, l]:
        return "non_adjacent"
    g = T.gap[k, l]
    if np.isnan(g) or g >= cfg.bound_depth_thresh:
        return "depth_gap"
    if cfg.use_convexity:
        if T.n_samples[k, l] > 0 and T.cratio[k, l] > cfg.concave_ratio_thresh \
                and T.cangle[k, l] > concave_angle_thresh(cfg):
            return "concave"
        return "convex"
    return "connected" if not cfg.no_force_connect else "continuous"


def distance_from_table(T: PairTable, cfg: Config):
    """距離行列．

    既存（フラグなし）: 深度連続 → connected_dist，段差あり → base + adjacent_penalty，非隣接 → base
    no_force_connect   : 深度連続でも connected_dist に上書きせず base（凸性なしの場合）
    use_convexity      : 深度連続かつ凹 → base + concave_penalty，深度連続かつ凸・平坦 → max(base − β, connected_dist)
    """
    n = len(T.base)
    D = np.full((n, n), cfg.non_adjacent_dist)
    for k in range(n):
        for l in range(k + 1, n):
            rule = pair_rule(T, k, l, cfg)
            b = T.base[k, l]
            if rule == "non_adjacent":
                d = b
            elif rule == "depth_gap":
                d = b + cfg.adjacent_penalty
            elif rule == "connected":
                d = cfg.connected_dist
            elif rule == "continuous":
                d = b
            elif rule == "concave":
                d = b + cfg.concave_penalty
            else:  # convex
                d = max(b - cfg.convex_bonus, cfg.connected_dist)
            D[k, l] = D[l, k] = d
    return D


class SizeChecker:
    """統合後クラスタの大きさが CAD × (1+m) を超えるか（改善2）．"""

    def __init__(self, masks, pts, valid_idx, v_u, v_v, cfg: Config, cam):
        self.cfg = cfg
        self.prior = load_cad_prior(cfg)
        self.masks = masks
        rng = np.random.default_rng(0)
        self.frag_pts = []
        ek = np.ones((2 * cfg.size_erode_px + 1,) * 2, np.uint8) if cfg.size_erode_px > 0 else None
        for m in masks:
            if ek is not None:
                me = cv2.erode(m.astype(np.uint8), ek) > 0
                m = me if np.count_nonzero(me[v_v, v_u]) >= 10 else m
            p = pts[valid_idx[m[v_v, v_u]]]
            if len(p) > cfg.max_points_per_fragment:
                p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
            self.frag_pts.append(p)
        # 画素ごとの実面積 z²/(fx·fy) [m²]．欠損画素は 0（面積に数えない）
        self.area_img = None
        if cam is not None:
            fx, fy = cam[0], cam[1]
            self.area_img = np.zeros(masks[0].shape)
            self.area_img[v_v, v_u] = pts[valid_idx, 2] ** 2 / (fx * fy)
        lim = 1.0 + cfg.size_margin
        self.limits = dict(L1=lim * self.prior.L[0], L2=lim * self.prior.L[1], L3=lim * self.prior.L[2],
                           D=lim * self.prior.D, A=lim * self.prior.A)

    def measure(self, members):
        p = np.vstack([self.frag_pts[i] for i in members])
        q = self.cfg.size_percentile
        c = p - p.mean(axis=0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ Vt.T
        lo, hi = np.percentile(proj, q, axis=0), np.percentile(proj, 100 - q, axis=0)
        ext = np.sort(hi - lo)[::-1]
        ext = np.concatenate([ext, np.zeros(3 - len(ext))])
        inside = np.all((proj >= lo) & (proj <= hi), axis=1)
        d = max_distance(p[inside]) if inside.sum() >= 4 else 0.0
        out = dict(L1=ext[0], L2=ext[1], L3=ext[2], D=d)
        if self.area_img is not None:
            union = np.zeros(self.masks[0].shape, bool)
            for i in members:
                union |= self.masks[i]
            out["A"] = float(self.area_img[union].sum())
        return out

    def violations(self, members):
        m = self.measure(members)
        return [k for k in SIZE_KEYS if k in m and m[k] > self.limits[k]]

    def ratios(self, members):
        m = self.measure(members)
        base = dict(L1=self.prior.L[0], L2=self.prior.L[1], L3=self.prior.L[2], D=self.prior.D, A=self.prior.A)
        return {k: m[k] / base[k] for k in m}


def constrained_average_linkage(D, thresh, check=None):
    """平均連結の凝集型クラスタリング（cannot-link 制約付き）．

    距離行列には同点（connected_dist）が多く，同点の対をどの順に統合するかで結果が変わる．
    制約なしで scipy（linkage + fcluster）と同じ分割にするため，scipy と同じ
    nearest-neighbor chain の探索順で統合対（互いに最近傍の対）を見つける．
    「各ステップで平均距離最小の対」と同じ統合になる（平均連結は reducible なので，同点がなければ一致する）．

    統合対の距離が thresh 以下なら check(members) を確かめ，違反ならその対の距離を ∞（cannot-link）に
    して探索をやり直す．平均連結の更新式により ∞ は統合後のクラスタにも引き継がれる
    （大きくなったクラスタはさらに大きさを超えるため）．thresh を超える統合は scipy と同じ探索順を
    保つためにそのまま行い，最後に thresh 以下の統合だけで分割を作る（fcluster の distance 基準と同じ）．

    Returns: labels (n,), events [(members_a, members_b, distance, violations)]（thresh 以下の統合候補のみ）
    """
    n = len(D)
    M = D.astype(float).copy()
    size = np.ones(n)
    members = {i: [i] for i in range(n)}
    parent = list(range(n))   # thresh 以下の統合だけの union-find

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    events = []
    chain = []
    while np.count_nonzero(size) > 1:
        if not chain:
            chain = [int(np.flatnonzero(size)[0])]
        while True:
            x = chain[-1]
            if len(chain) > 1:
                y = chain[-2]
                cur = M[x, y]
            else:
                y, cur = -1, np.inf
            for i in range(n):
                if size[i] == 0 or i == x:
                    continue
                if M[x, i] < cur:
                    cur, y = M[x, i], i
            if y < 0 or not np.isfinite(cur):
                break
            if len(chain) > 1 and y == chain[-2]:
                break
            chain.append(y)
        if y < 0 or not np.isfinite(cur):
            # x はどのクラスタとも統合できない（すべて cannot-link）
            size[x] = 0
            chain = []
            continue
        chain = chain[:-2]
        x, y = min(x, y), max(x, y)
        if cur <= thresh:
            viol = check(members[x] + members[y]) if check is not None else []
            events.append((list(members[x]), list(members[y]), float(cur), viol))
            if viol:
                M[x, y] = M[y, x] = np.inf
                chain = []
                continue
            parent[find(x)] = find(y)
        nx, ny = size[x], size[y]
        size[x] = 0
        size[y] = nx + ny
        for i in range(n):
            if size[i] == 0 or i == y:
                continue
            M[i, y] = M[y, i] = (nx * M[i, x] + ny * M[i, y]) / (nx + ny)
        members[y] = members[x] + members[y]
        del members[x]
    roots = [find(i) for i in range(n)]
    uniq = {r: k + 1 for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([uniq[r] for r in roots]), events


def merge_constrained(masks, pts, valid_idx, v_u, v_v, cfg: Config, cam=None):
    """改善2・3の統合．既存の拡張（円筒軸・CAD寸法分割など）は凸性を使わない場合のみ距離行列に反映される．"""
    if cfg.use_convexity or cfg.no_force_connect:
        if cfg.use_cylinder_axis or cfg.hier_normal_check or cfg.hier_tiebreak or cfg.hier_overlap_link \
                or cfg.hier_model_check:
            raise NotImplementedError("凸性の距離行列は 7/31 の判定（深度段差）だけを置き換える．既存の追加判定とは併用不可")
        T = pair_table(masks, pts, valid_idx, v_u, v_v, cfg, convexity=cfg.use_convexity, cam=cam)
        D = distance_from_table(T, cfg)
    else:
        T = None
        D = distance_matrix(masks, pts, valid_idx, v_u, v_v, cfg)
    info = dict(D=D, table=T, stops=Counter(), n_stops=0, events=[])
    if len(masks) == 1:
        labels = np.array([1])
    elif cfg.use_size_constraint or cfg.own_linkage:
        if cfg.use_cad_constraint:
            raise NotImplementedError("use_cad_constraint（既存の分割方式）と use_size_constraint は併用しない")
        checker = SizeChecker(masks, pts, valid_idx, v_u, v_v, cfg, cam) if cfg.use_size_constraint else None
        labels, events = constrained_average_linkage(D, cfg.cluster_thresh,
                                                     checker.violations if checker else None)
        info["events"] = events
        for _, _, _, viol in events:
            if viol:
                info["n_stops"] += 1
                info["stops"].update(viol)
        if info["n_stops"]:
            log.info("size constraint stopped %d merges: %s", info["n_stops"], dict(info["stops"]))
    else:
        labels = fcluster(linkage(squareform(D, checks=False), method=cfg.linkage_method),
                          t=cfg.cluster_thresh, criterion="distance")
    info["labels"] = labels
    LAST_INFO.clear()
    LAST_INFO.update(info)
    merged = []
    for cid in np.unique(labels):
        inst = np.zeros(masks[0].shape, dtype=bool)
        for i in np.where(labels == cid)[0]:
            inst |= masks[i]
        merged.append(inst)
    return merged
