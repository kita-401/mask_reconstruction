"""Config.use_cpp=True のときの統合（距離行列・小断片の割り当て）．

手順・判定は merging.distance_matrix と graph_merging.attach_small_fragments とまったく同じで，
画像全体を走査していた重い計算（断片の点の取り出し，組ごとの隣接・深度差・法線角・重なり）だけを
C++（cpp_kernels.py）に置き換えている．乱数（点の間引き）の呼び出し順も同じ．
元の Python 版はそのまま残してある（use_cpp=False で使われる）．
"""
import numpy as np

from . import cpp_kernels as ck
from .config import Config
from .merging import POINT_CACHE, axis_segment_distance, cad_extent, cylinder_axis, depth_image, principal_extent


def _index_image(pts, valid_idx, v_u, v_v, shape):
    return POINT_CACHE.get(pts, valid_idx, ("index", shape), lambda: ck.index_image(valid_idx, v_u, v_v, shape))


def fragment_indices(masks, pts, valid_idx, v_u, v_v):
    """各マスクの点の番号（valid_idx の何番目か）．m[v_v, v_u] が真の番号と同じ．"""
    stack = ck.stack_masks(masks)
    return stack, ck.gather_indices(stack, _index_image(pts, valid_idx, v_u, v_v, masks[0].shape))


def distance_matrix_cpp(masks, pts, valid_idx, v_u, v_v, cfg: Config):
    """merging.distance_matrix と同じ距離行列．"""
    n = len(masks)
    shape = masks[0].shape
    stack, idx = fragment_indices(masks, pts, valid_idx, v_u, v_v)
    frag = [pts[valid_idx[ix]] for ix in idx]

    centers, axes = [], []
    for p in frag:
        c = np.mean(p, axis=0)
        _, _, Vt = np.linalg.svd(p - c, full_matrices=False)
        centers.append(c)
        axes.append(Vt[0])

    N = None
    if cfg.hier_normal_check or (cfg.use_cylinder_axis and cfg.fast_normals):
        from .graph_merging import _normal_image
        N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg)
    if cfg.use_cylinder_axis:
        rng = np.random.default_rng(0)
        if cfg.fast_normals:
            Nflat = N[v_v, v_u]
            cyl = [cylinder_axis(frag[k], cfg, rng, normals=Nflat[idx[k]]) for k in range(n)]
        else:
            cyl = [cylinder_axis(frag[k], cfg, rng) for k in range(n)]

    Z = depth_image(pts, valid_idx, v_u, v_v, shape)
    adjacent, gaps, nangle, overlap, area = ck.pair_features(
        stack, cfg.adjacency_kernel, Z, N if cfg.hier_normal_check else None)
    areas = [int(a) for a in area]

    D = np.full((n, n), cfg.non_adjacent_dist)
    for k in range(n):
        for l in range(k + 1, n):
            d_vec = centers[l] - centers[k]
            d_norm = np.linalg.norm(d_vec)
            d_hat = d_vec / d_norm if d_norm > 1e-6 else axes[k]
            angle_diff = 1.0 - np.abs(np.dot(axes[k], axes[l]))
            axis_align = 1.0 - max(abs(np.dot(d_hat, axes[k])), abs(np.dot(d_hat, axes[l])))

            is_adjacent = bool(adjacent[k, l])
            gap = None
            if is_adjacent and not np.isnan(gaps[k, l]):
                gap = float(gaps[k, l])

            connected = is_adjacent and gap is not None and gap < cfg.bound_depth_thresh
            if connected and cfg.use_cylinder_axis and min(areas[k], areas[l]) >= cfg.axis_min_area:
                connected = axis_segment_distance(cyl[k], cyl[l], cfg.axis_seg_ext) < cfg.axis_dist_thresh

            overlapped = False
            if cfg.hier_overlap_link and is_adjacent:
                overlapped = overlap[k, l] / min(areas[k], areas[l]) >= cfg.overlap_link_ratio

            ang = None
            if connected and cfg.hier_normal_check:
                ang = None if np.isnan(nangle[k, l]) else float(nangle[k, l])
                connected = ang is not None and ang < cfg.normal_angle_thresh

            if connected and cfg.hier_model_check and min(areas[k], areas[l]) >= cfg.model_check_min_area:
                from .model_fit import explained_ratios
                ratios = explained_ratios([frag[k], frag[l]], cfg)
                connected = min(ratios) >= cfg.model_check_min_ratio

            if overlapped and not connected:
                D[k, l] = cfg.connected_dist + cfg.tiebreak_scale * 2.0
            elif connected:
                D[k, l] = cfg.connected_dist
                if cfg.hier_tiebreak:
                    tb = gap / cfg.bound_depth_thresh
                    if ang is not None:
                        tb += ang / cfg.normal_angle_thresh
                    D[k, l] += cfg.tiebreak_scale * tb
            else:
                D[k, l] = (d_norm * cfg.w_center
                           + (angle_diff * cfg.w_angle + axis_align * cfg.w_axis_align) * cfg.L_cad)
                if is_adjacent:
                    D[k, l] += cfg.adjacent_penalty
            D[l, k] = D[k, l]
    return D


def fragment_points_subsampled(masks, pts, valid_idx, v_u, v_v, cfg: Config, rng):
    """merge_fragments の CAD 制約用の点（max_points_per_fragment 点まで間引き）．"""
    _, idx = fragment_indices(masks, pts, valid_idx, v_u, v_v)
    out = []
    for ix in idx:
        p = pts[valid_idx[ix]]
        if len(p) > cfg.max_points_per_fragment:
            p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
        out.append(p)
    return out


def attach_small_fragments_cpp(clusters, smalls, pts, valid_idx, v_u, v_v, cfg: Config):
    """graph_merging.attach_small_fragments と同じ割り当て．"""
    from .graph_merging import _normal_image
    clusters = [c.copy() for c in clusters]
    if not clusters:
        return list(smalls) if cfg.stage2_keep_unassigned else []
    shape = clusters[0].shape
    N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg)
    Z = depth_image(pts, valid_idx, v_u, v_v, shape)
    I = _index_image(pts, valid_idx, v_u, v_v, shape)
    limit = cad_extent(cfg)[:2] * cfg.cad_size_margin if cfg.use_cad_constraint else None
    rng = np.random.default_rng(0)

    def sub_points(m):
        p = pts[valid_idx[ck.gather_indices(ck.stack_masks([m]), I)[0]]]
        if len(p) > cfg.max_points_per_fragment:
            p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
        return p

    remaining = list(range(len(smalls)))
    while remaining:
        # 残っている小断片とまとまりの組の特徴をまとめて計算する（小断片 → まとまりの順に並べる）
        ns = len(remaining)
        stack = ck.stack_masks([smalls[si] for si in remaining] + clusters)
        adjacent, gaps, nangle, _, _ = ck.pair_features(stack, cfg.adjacency_kernel, Z, N)
        best = None
        for a, si in enumerate(remaining):
            for ci in range(len(clusters)):
                b = ns + ci
                if not adjacent[a, b]:
                    continue
                if np.isnan(gaps[a, b]) or gaps[a, b] >= cfg.stage2_depth_thresh:
                    continue
                gap = float(gaps[a, b])
                if np.isnan(nangle[a, b]) or nangle[a, b] >= cfg.stage2_normal_thresh:
                    continue
                ang = float(nangle[a, b])
                cost = gap / cfg.stage2_depth_thresh + ang / cfg.stage2_normal_thresh
                if best is not None and cost >= best[0]:
                    continue
                if limit is not None:
                    ext = principal_extent(np.vstack([sub_points(smalls[si]), sub_points(clusters[ci])]),
                                           cfg.extent_percentile)
                    if np.any(ext[:2] > limit):
                        continue
                best = (cost, si, ci)
        if best is None:
            break
        _, si, ci = best
        clusters[ci] = clusters[ci] | smalls[si]
        remaining.remove(si)

    if cfg.stage2_keep_unassigned:
        clusters += [smalls[si] for si in remaining]
    return clusters
