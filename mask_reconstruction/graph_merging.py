"""断片統合（グラフ＋検証方式）．

1. 断片ごとに点群・面積・円筒軸（法線PCA）を求める
2. 「同じ部品らしい」断片ペアを辺として列挙し，コスト（小さいほど同じ部品らしい）を付ける
   - 隣接辺   : 2Dで隣接 かつ 境界の深度差 < bound_depth_thresh
                 かつ 境界両側の平均法線の角度差 < normal_angle_thresh（use_normal_check）
                 かつ（graph_use_axis で，両方が大きい断片なら）円筒軸の線分間距離 < axis_dist_thresh
   - 隠れ辺   : 2Dで隣接していないが，円筒軸の線分（間隙ぶん延長）が近く，
                 両断片の間の領域が手前の物体で覆われている（隠れによる分断）
3. コストの小さい辺から順に，両端のまとまりを統合する．統合のたびに
   CAD寸法に収まるか（と，任意でテンプレートの当てはまり）を確認し，満たさなければ統合しない
"""
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .config import Config
from .merging import (POINT_CACHE, axis_segment_distance, boundary_depth_gap, cad_extent, cylinder_axis,
                      principal_extent)
from .template_fit import template_fitness


class _Fragment:
    def __init__(self, mask, pts, valid_idx, v_u, v_v, cfg, rng, kernel):
        self.mask = mask
        self.area = int(mask.sum())
        p = pts[valid_idx[mask[v_v, v_u]]]
        self.points = p
        sub = p if len(p) <= cfg.max_points_per_fragment else \
            p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
        self.sub = sub
        self.tree = cKDTree(sub)
        self.dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        self.axis = cylinder_axis(p, cfg, rng)
        self.axis_reliable = self.area >= cfg.axis_min_area


def _depth_image(pts, valid_idx, v_u, v_v, shape):
    Z = np.zeros(shape, dtype=np.float64)
    Z[v_v, v_u] = pts[valid_idx, 2]
    return Z


def _normal_image(pts, valid_idx, v_u, v_v, shape, cfg):
    """画素ごとの法線．同じ点群なら1回だけ計算する（読み取り専用で使うこと）．

    fast_normals=True なら，Open3D の近傍探索ではなく深度画像上の差分の外積で求める（convexity.py と同じ）．
    求められない画素は 0 ベクトル．
    """
    if cfg.fast_normals:
        def make_fast():
            from .convexity import organized_normals, point_image
            P, V = point_image(pts, valid_idx, v_u, v_v, shape)
            return organized_normals(P, V, cfg.fast_normal_window)[0]
        return POINT_CACHE.get(pts, valid_idx, ("normals_fast", shape, cfg.fast_normal_window), make_fast)

    def make():
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[valid_idx]))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.boundary_normal_radius, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
        N = np.zeros((*shape, 3))
        N[v_v, v_u] = np.asarray(pcd.normals)
        return N
    return POINT_CACHE.get(pts, valid_idx, ("normals", shape, cfg.boundary_normal_radius), make)


def boundary_normal_angle(fa, fb, N):
    """境界付近の両側の平均法線の角度差 [deg]．同じ部品なら表面が滑らかにつながり小さい．"""
    means = []
    for band in (fa.mask & fb.dilated, fb.mask & fa.dilated):
        n = N[band]
        n = n[np.linalg.norm(n, axis=1) > 0]
        if len(n) < 5:
            return None
        m = n.mean(axis=0)
        means.append(m / np.linalg.norm(m))
    return float(np.degrees(np.arccos(np.clip(means[0] @ means[1], -1, 1))))


def _occluded_between(fa, fb, depth_img, cam, cfg):
    """2断片の最近点を結ぶ線分上で，観測深度がその線分より手前にある（＝隠れている）割合．"""
    d, j = fb.tree.query(fa.sub)
    i = int(np.argmin(d))
    pa, pb = fa.sub[i], fb.sub[j[i]]
    fx, fy, cx, cy = cam
    h, w = depth_img.shape
    ts = np.linspace(0.1, 0.9, cfg.bridge_samples)
    q = pa[None] * (1 - ts[:, None]) + pb[None] * ts[:, None]
    u = np.round(fx * q[:, 0] / q[:, 2] + cx).astype(int)
    v = np.round(fy * q[:, 1] / q[:, 2] + cy).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not ok.any():
        return 0.0, float(d[i])
    z_obs = depth_img[v[ok], u[ok]]
    in_frag = fa.mask[v[ok], u[ok]] | fb.mask[v[ok], u[ok]]
    occluded = ((z_obs > 0) & (z_obs < q[ok, 2] - cfg.bridge_occlusion_tol)) & ~in_frag
    return float(occluded.mean()), float(d[i])


def candidate_edges(frags, pts, valid_idx, v_u, v_v, cam, cfg: Config):
    """辺のリスト [(cost, a, b, kind)]．"""
    edges = []
    shape = frags[0].mask.shape
    depth_img = _depth_image(pts, valid_idx, v_u, v_v, shape)
    N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg) if cfg.use_normal_check else None
    for a in range(len(frags)):
        fa = frags[a]
        for b in range(a + 1, len(frags)):
            fb = frags[b]
            if np.any(fa.dilated & fb.mask):
                gap = boundary_depth_gap(fa.mask, fb.mask, fa.dilated, fb.dilated,
                                         pts, valid_idx, v_u, v_v)
                if gap is None or gap >= cfg.bound_depth_thresh:
                    continue
                cost = gap / cfg.bound_depth_thresh
                if N is not None:
                    ang = boundary_normal_angle(fa, fb, N)
                    if ang is None or ang >= cfg.normal_angle_thresh:
                        continue
                    cost += ang / cfg.normal_angle_thresh
                large = fa.axis_reliable and fb.axis_reliable
                if large and cfg.graph_use_axis:
                    ad = axis_segment_distance(fa.axis, fb.axis, cfg.axis_seg_ext)
                    if ad >= cfg.axis_dist_thresh:
                        continue
                    cost += ad / cfg.axis_dist_thresh
                edges.append((cost, a, b, "adjacent" if large else "adjacent_small"))
            elif cfg.use_bridge and cam is not None and fa.axis_reliable and fb.axis_reliable:
                dmin = fb.tree.query(fa.sub, distance_upper_bound=cfg.bridge_max_gap)[0].min()
                if not np.isfinite(dmin):
                    continue
                ad = axis_segment_distance(fa.axis, fb.axis, cfg.axis_seg_ext + dmin)
                if ad >= cfg.axis_dist_thresh:
                    continue
                occ, _ = _occluded_between(fa, fb, depth_img, cam, cfg)
                if occ < cfg.bridge_min_occluded:
                    continue
                edges.append((ad / cfg.axis_dist_thresh + cfg.bridge_penalty, a, b, "bridge"))
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    return edges


class _Region:
    """2段目で使う，マスクと膨張マスクだけを持つ領域（boundary_normal_angle 用）．"""

    def __init__(self, mask, kernel):
        self.mask = mask
        self.dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0


def attach_small_fragments(clusters, smalls, pts, valid_idx, v_u, v_v, cfg: Config):
    """2段目：小さな断片を1段目のまとまりに割り当てる．

    - 割り当て先の条件：2Dで隣接，境界の深度差 < stage2_depth_thresh，
      境界両側の法線の角度差 < stage2_normal_thresh，（CAD制約ありなら）割り当て後もCAD寸法内
    - コスト（深度差/閾値 + 角度差/閾値）が最小の割り当てから順に確定し，まとまりを更新して繰り返す
      （割り当て済みの小断片を介して，さらに別の小断片がつながることもある）
    - どこにも割り当てられなかった小断片は stage2_keep_unassigned=False なら捨てる
    """
    kernel = np.ones((cfg.adjacency_kernel, cfg.adjacency_kernel), np.uint8)
    clusters = [c.copy() for c in clusters]
    if not clusters:
        return list(smalls) if cfg.stage2_keep_unassigned else []
    shape = clusters[0].shape
    N = _normal_image(pts, valid_idx, v_u, v_v, shape, cfg)
    limit = cad_extent(cfg)[:2] * cfg.cad_size_margin if cfg.use_cad_constraint else None
    rng = np.random.default_rng(0)

    def sub_points(m):
        p = pts[valid_idx[m[v_v, v_u]]]
        if len(p) > cfg.max_points_per_fragment:
            p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
        return p

    remaining = list(range(len(smalls)))
    s_regions = [_Region(m, kernel) for m in smalls]
    while remaining:
        c_regions = [_Region(c, kernel) for c in clusters]
        best = None
        for si in remaining:
            sr = s_regions[si]
            for ci, cr in enumerate(c_regions):
                if not np.any(sr.dilated & cr.mask):
                    continue
                gap = boundary_depth_gap(sr.mask, cr.mask, sr.dilated, cr.dilated,
                                         pts, valid_idx, v_u, v_v)
                if gap is None or gap >= cfg.stage2_depth_thresh:
                    continue
                ang = boundary_normal_angle(sr, cr, N)
                if ang is None or ang >= cfg.stage2_normal_thresh:
                    continue
                cost = gap / cfg.stage2_depth_thresh + ang / cfg.stage2_normal_thresh
                if best is not None and cost >= best[0]:
                    continue
                if limit is not None:
                    ext = principal_extent(np.vstack([sub_points(sr.mask), sub_points(cr.mask)]),
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


def graph_merge(masks, pts, valid_idx, v_u, v_v, cfg: Config, cam=None, return_log=False):
    if len(masks) == 0:
        return ([], []) if return_log else []
    if cfg.two_stage:
        from dataclasses import replace
        large = [m for m in masks if m.sum() >= cfg.stage1_min_area]
        small = [m for m in masks if m.sum() < cfg.stage1_min_area]
        out = graph_merge(large, pts, valid_idx, v_u, v_v, replace(cfg, two_stage=False), cam)
        out = attach_small_fragments(out, small, pts, valid_idx, v_u, v_v, cfg)
        return (out, []) if return_log else out
    rng = np.random.default_rng(0)
    kernel = np.ones((cfg.adjacency_kernel, cfg.adjacency_kernel), np.uint8)
    frags = [_Fragment(m, pts, valid_idx, v_u, v_v, cfg, rng, kernel) for m in masks]
    edges = candidate_edges(frags, pts, valid_idx, v_u, v_v, cam, cfg)

    limit = cad_extent(cfg)[:2] * cfg.cad_size_margin if cfg.use_cad_constraint else None
    label = list(range(len(frags)))
    members = {i: [i] for i in range(len(frags))}
    fit_cache = {}
    log = []

    def fitness_of(ids):
        key = tuple(sorted(ids))
        if key not in fit_cache:
            fit_cache[key] = template_fitness(np.vstack([frags[i].sub for i in ids]), cfg)
        return fit_cache[key]

    for cost, a, b, kind in edges:
        ca, cb = label[a], label[b]
        if ca == cb:
            continue
        merged = members[ca] + members[cb]
        reason = "merged"
        if limit is not None:
            ext = principal_extent(np.vstack([frags[i].sub for i in merged]), cfg.extent_percentile)
            if np.any(ext[:2] > limit):
                reason = "cad_size"
        if reason == "merged" and cfg.use_template_check:
            n_pts = sum(len(frags[i].sub) for i in merged)
            if n_pts >= cfg.template_min_points:
                f_m = fitness_of(merged)
                f_parts = max(fitness_of(members[ca]), fitness_of(members[cb]))
                if f_m < cfg.template_min_fitness or f_m < f_parts - cfg.template_max_drop:
                    reason = "template_fit"
        log.append((kind, a, b, cost, reason))
        if reason != "merged":
            continue
        for i in members[cb]:
            label[i] = ca
        members[ca] = merged
        del members[cb]

    out = []
    for ids in members.values():
        inst = np.zeros(masks[0].shape, dtype=bool)
        for i in ids:
            inst |= masks[i]
        out.append(inst)
    return (out, log) if return_log else out
