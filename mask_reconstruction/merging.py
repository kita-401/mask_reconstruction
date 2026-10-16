"""断片マスクの統合：3D形状（PCA主軸）と境界の深度連続性に基づく階層クラスタリング．"""
import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import Config


def fragment_pca(masks, pts, valid_idx, v_u, v_v):
    """各断片の3D点群の重心と第1主軸．"""
    centers, axes = [], []
    for m in masks:
        p = pts[valid_idx[m[v_v, v_u]]]
        c = np.mean(p, axis=0)
        _, _, Vt = np.linalg.svd(p - c, full_matrices=False)
        centers.append(c)
        axes.append(Vt[0])
    return centers, axes


class _PointCloudCache:
    """同じ点群（同じ配列オブジェクト）に対する画像形式の計算結果を使い回す（結果は変わらない）．

    点群の配列そのものを参照として持ち，`is` で比べるので，別の点群と取り違えることはない．
    """

    def __init__(self):
        self.key = None
        self.store = {}

    def get(self, pts, valid_idx, name, make):
        if self.key is None or self.key[0] is not pts or self.key[1] is not valid_idx:
            self.key = (pts, valid_idx)
            self.store = {}
        if name not in self.store:
            self.store[name] = make()
        return self.store[name]


POINT_CACHE = _PointCloudCache()


def depth_image(pts, valid_idx, v_u, v_v, shape):
    """点群を画素に戻した深度画像 [m]（点のない画素は 0）．"""
    def make():
        Z = np.zeros(shape, dtype=np.float64)
        Z[v_v, v_u] = pts[valid_idx, 2]
        return Z
    return POINT_CACHE.get(pts, valid_idx, ("depth", shape), make)


def boundary_depth_gap(mask_k, mask_l, dil_k, dil_l, pts, valid_idx, v_u, v_v):
    """隣接する2断片の境界付近の深度中央値の差 [m]．計算できなければ None．

    点は画素と1対1に対応するので，境界の画素の深度を深度画像から直接取る（全点群からの抽出より速い）．
    """
    Z = depth_image(pts, valid_idx, v_u, v_v, mask_k.shape)
    z_k = Z[mask_k & dil_l]
    z_l = Z[mask_l & dil_k]
    z_k = z_k[z_k > 0]
    z_l = z_l[z_l > 0]
    if len(z_k) == 0 or len(z_l) == 0:
        return None
    return abs(np.median(z_k) - np.median(z_l))


def cylinder_axis(p, cfg: Config, rng, normals=None):
    """法線の主成分分析で断片の円筒中心軸を求める．

    円筒面の法線は軸と直交するので，Σ n nᵀ の最小固有値の固有ベクトルが軸方向．
    各点を法線の逆向きに半径だけ移動すると軸上に乗るので，その中央値を軸の通過点とする．

    normals を渡すと（fast_normals：深度画像上で求めた法線），断片ごとの法線推定を省く．

    Returns: (方向 d, 通過点 c0, 軸上の範囲 t_min, t_max)
    """
    if normals is not None:
        ok = np.linalg.norm(normals, axis=1) > 0
        p, n = p[ok], normals[ok]
        if len(p) > cfg.axis_max_points:
            sel = rng.choice(len(p), cfg.axis_max_points, replace=False)
            p, n = p[sel], n[sel]
    else:
        import open3d as o3d
        if len(p) > cfg.axis_max_points:
            p = p[rng.choice(len(p), cfg.axis_max_points, replace=False)]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.normal_radius, max_nn=30))
        pc.orient_normals_towards_camera_location(np.zeros(3))
        n = np.asarray(pc.normals)
    _, V = np.linalg.eigh(n.T @ n / len(n))
    d = V[:, 0]
    c = p - cfg.pipe_radius * n
    c0 = np.median(c - np.outer(c @ d, d), axis=0)
    t = (c - c0) @ d
    return d, c0, np.percentile(t, 2), np.percentile(t, 98)


def axis_segment_distance(ax1, ax2, ext, n_samples=60):
    """2つの円筒軸を実在範囲（両端を ext 延長）の線分とみなしたときの最短距離 [m]．"""
    segs = []
    for d, c0, t0, t1 in (ax1, ax2):
        segs.append(c0 + np.outer(np.linspace(t0 - ext, t1 + ext, n_samples), d))
    return np.min(np.linalg.norm(segs[0][:, None] - segs[1][None], axis=2))


def distance_matrix(masks, pts, valid_idx, v_u, v_v, cfg: Config):
    """断片間の距離行列．

    - 2D で隣接し，境界の深度差が bound_depth_thresh 未満 → 同一部品（connected_dist）
      use_cylinder_axis=True なら，さらに円筒軸の線分間距離が axis_dist_thresh 未満であること
      hier_normal_check=True なら，さらに境界両側の法線の角度差が normal_angle_thresh 未満であること
      hier_tiebreak=True なら，connected_dist に深度差・角度差に応じた小さな連続値を足す
    - それ以外 → 重心間距離 + (主軸の角度差 + 主軸と重心方向のずれ) * L_cad
                 隣接していて深度が不連続なら adjacent_penalty を加算
    """
    n = len(masks)
    centers, axes = fragment_pca(masks, pts, valid_idx, v_u, v_v)
    if cfg.use_cylinder_axis:
        rng = np.random.default_rng(0)
        if cfg.fast_normals:
            from .graph_merging import _normal_image
            Nflat = _normal_image(pts, valid_idx, v_u, v_v, masks[0].shape, cfg)[v_v, v_u]
            cyl = []
            for m in masks:
                sel = np.flatnonzero(m[v_v, v_u])
                cyl.append(cylinder_axis(pts[valid_idx[sel]], cfg, rng, normals=Nflat[sel]))
        else:
            cyl = [cylinder_axis(pts[valid_idx[m[v_v, v_u]]], cfg, rng) for m in masks]
        areas = [int(m.sum()) for m in masks]
    kernel = np.ones((cfg.adjacency_kernel, cfg.adjacency_kernel), np.uint8)
    dilated = [cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in masks]
    areas_all = [int(m.sum()) for m in masks]
    if cfg.hier_normal_check:
        from .graph_merging import _Region, _normal_image, boundary_normal_angle
        N = _normal_image(pts, valid_idx, v_u, v_v, masks[0].shape, cfg)
        regions = [_Region(m, kernel) for m in masks]

    D = np.full((n, n), cfg.non_adjacent_dist)
    for k in range(n):
        for l in range(k + 1, n):
            d_vec = centers[l] - centers[k]
            d_norm = np.linalg.norm(d_vec)
            d_hat = d_vec / d_norm if d_norm > 1e-6 else axes[k]
            angle_diff = 1.0 - np.abs(np.dot(axes[k], axes[l]))
            axis_align = 1.0 - max(abs(np.dot(d_hat, axes[k])), abs(np.dot(d_hat, axes[l])))

            is_adjacent = np.any(dilated[k] & masks[l])
            gap = None
            if is_adjacent:
                gap = boundary_depth_gap(masks[k], masks[l], dilated[k], dilated[l],
                                         pts, valid_idx, v_u, v_v)

            connected = is_adjacent and gap is not None and gap < cfg.bound_depth_thresh
            # 小さい断片は軸推定が不安定なので，軸の判定を省き深度連続だけで判断する
            if connected and cfg.use_cylinder_axis and min(areas[k], areas[l]) >= cfg.axis_min_area:
                connected = axis_segment_distance(cyl[k], cyl[l], cfg.axis_seg_ext) < cfg.axis_dist_thresh

            # FastSAM の断片同士が大きく重なっていれば，FastSAM 自身が同じ物体とみなした証拠として結合する
            overlapped = False
            if cfg.hier_overlap_link and is_adjacent:
                inter = np.sum(masks[k] & masks[l])
                overlapped = inter / min(areas_all[k], areas_all[l]) >= cfg.overlap_link_ratio

            ang = None
            if connected and cfg.hier_normal_check:
                ang = boundary_normal_angle(regions[k], regions[l], N)
                connected = ang is not None and ang < cfg.normal_angle_thresh

            # モデル1個で両方の断片を説明できるか（大きい断片同士のみ，部品の種類に依存しない）
            if connected and cfg.hier_model_check and min(areas_all[k], areas_all[l]) >= cfg.model_check_min_area:
                from .model_fit import explained_ratios
                ratios = explained_ratios([pts[valid_idx[masks[x][v_v, v_u]]] for x in (k, l)], cfg)
                connected = min(ratios) >= cfg.model_check_min_ratio

            if overlapped and not connected:
                D[k, l] = cfg.connected_dist + cfg.tiebreak_scale * 2.0   # 深度・法線で結合したペアより後に
            elif connected:
                D[k, l] = cfg.connected_dist
                if cfg.hier_tiebreak:
                    # 同値（connected_dist）だと統合順が実装依存になるので，連続値を足して順序を決める
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


def principal_extent(p, percentile):
    """点群の主成分軸方向の寸法（大きい順，[percentile, 100-percentile] の幅）．"""
    c = p - p.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ Vt.T
    return np.percentile(proj, 100 - percentile, axis=0) - np.percentile(proj, percentile, axis=0)


def cad_extent(cfg: Config):
    """CADモデル点群の主軸方向寸法 [m]（大きい順）．"""
    import open3d as o3d
    return principal_extent(np.asarray(o3d.io.read_point_cloud(cfg.cad_model).points), 0.0)


def remerge_within_cad(clusters, D, frag_pts, limit, cfg: Config):
    """分割で生じた切れ端を，最も近い断片対（single linkage）の距離が閾値以下で，
    統合後もCAD寸法に収まるクラスタ同士で貪欲に再統合する．統合後の寸法が小さい組を優先．"""
    clusters = [list(c) for c in clusters]
    while True:
        cands = []
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = D[np.ix_(clusters[a], clusters[b])].min()
                if d > cfg.cluster_thresh:
                    continue
                ext = principal_extent(np.vstack([frag_pts[i] for i in clusters[a] + clusters[b]]),
                                       cfg.extent_percentile)
                if np.all(ext[:2] <= limit):
                    cands.append((d, ext[0], a, b))
        if not cands:
            return clusters
        _, _, a, b = min(cands)
        clusters[a] += clusters[b]
        del clusters[b]


def cad_constrained_cut(Z, D, frag_pts, cfg: Config):
    """デンドログラムを閾値で切ったあと，寸法がCADを超えるクラスタを子ノードへ再帰的に分割する．

    距離行列には同値（connected_dist）が多く，統合順の違いで結果が変わるため，
    独自に統合し直さず scipy のデンドログラムをそのまま使う．制約なしなら fcluster と同一．
    """
    n = len(frag_pts)
    limit = cad_extent(cfg)[:2] * cfg.cad_size_margin
    children = {n + j: (int(Z[j, 0]), int(Z[j, 1])) for j in range(len(Z))}
    height = {n + j: Z[j, 2] for j in range(len(Z))}

    def leaves(node):
        return [node] if node < n else leaves(children[node][0]) + leaves(children[node][1])

    def fits(node):
        ext = principal_extent(np.vstack([frag_pts[i] for i in leaves(node)]), cfg.extent_percentile)
        return np.all(ext[:2] <= limit)

    clusters = []

    def visit(node):
        if node < n or (height[node] <= cfg.cluster_thresh and fits(node)):
            clusters.append(leaves(node))
        else:
            visit(children[node][0])
            visit(children[node][1])

    visit(n + len(Z) - 1)
    if cfg.cad_remerge:
        clusters = remerge_within_cad(clusters, D, frag_pts, limit, cfg)
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        labels[members] = cid + 1
    return labels


def merge_fragments(masks, pts, valid_idx, v_u, v_v, cfg: Config, cam=None):
    """断片を統合し，部品ごとのマスクを返す．

    cam: (fx, fy, cx, cy)．merge_method="graph" の隠れ辺の判定に使う．
    """
    if len(masks) == 0:
        return []
    if cfg.post_close:
        # 2026-10-16 の改善5（postprocess.py）
        from dataclasses import replace
        from .postprocess import close_masks
        merged = merge_fragments(masks, pts, valid_idx, v_u, v_v, replace(cfg, post_close=False), cam)
        return close_masks(merged, pts, valid_idx, v_u, v_v, cfg)
    if cfg.use_size_constraint or cfg.own_linkage or cfg.use_convexity or cfg.no_force_connect:
        # 2026-10-16 の改善2・3（constrained_merging.py）．フラグがすべてオフならここは通らない
        from .constrained_merging import merge_constrained
        return merge_constrained(masks, pts, valid_idx, v_u, v_v, cfg, cam)
    if cfg.auto_model_shape:
        from dataclasses import replace
        from .model_shape import cylinder_info_from_file
        info = cylinder_info_from_file(cfg.cad_model)
        cfg = replace(cfg, auto_model_shape=False, pipe_radius=info.radius,
                      use_cylinder_axis=cfg.use_cylinder_axis and info.fraction >= cfg.cylinder_min_fraction)
    if cfg.two_stage and cfg.merge_method == "hierarchical":
        from dataclasses import replace
        from .graph_merging import attach_small_fragments
        large = [m for m in masks if m.sum() >= cfg.stage1_min_area]
        small = [m for m in masks if m.sum() < cfg.stage1_min_area]
        merged = merge_fragments(large, pts, valid_idx, v_u, v_v, replace(cfg, two_stage=False), cam) \
            if large else []
        return attach_small_fragments(merged, small, pts, valid_idx, v_u, v_v, cfg)
    if cfg.merge_method == "graph":
        from .graph_merging import graph_merge
        merged = graph_merge(masks, pts, valid_idx, v_u, v_v, cfg, cam)
        if cfg.absorb_small:
            merged = absorb_small_clusters(merged, pts, valid_idx, v_u, v_v, cfg)
        return merged
    D = distance_matrix(masks, pts, valid_idx, v_u, v_v, cfg)
    if len(masks) == 1:
        return [masks[0].copy()]
    Z = linkage(squareform(D, checks=False), method=cfg.linkage_method)
    if cfg.use_cad_constraint:
        rng = np.random.default_rng(0)
        frag_pts = []
        for m in masks:
            p = pts[valid_idx[m[v_v, v_u]]]
            if len(p) > cfg.max_points_per_fragment:
                p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
            frag_pts.append(p)
        labels = cad_constrained_cut(Z, D, frag_pts, cfg)
    else:
        labels = fcluster(Z, t=cfg.cluster_thresh, criterion="distance")
    merged = []
    for cid in np.unique(labels):
        inst = np.zeros(masks[0].shape, dtype=bool)
        for i in np.where(labels == cid)[0]:
            inst |= masks[i]
        merged.append(inst)
    if cfg.absorb_small:
        merged = absorb_small_clusters(merged, pts, valid_idx, v_u, v_v, cfg)
    return merged


def absorb_small_clusters(masks, pts, valid_idx, v_u, v_v, cfg: Config):
    """統合後の小さなまとまり（切れ端）を，隣接するまとまりのうち境界の深度が最も連続しているものに吸収する．

    - 面積 absorb_area 未満のまとまりを小さい順に処理する
    - 吸収先の条件：2D で隣接し，境界の深度差 < absorb_depth_thresh
    - use_cad_constraint=True なら，吸収後の寸法が CAD × cad_size_margin 以内であること
    - 条件を満たす吸収先がなければそのまま残す（削除はしない）
    """
    masks = [m.copy() for m in masks]
    kernel = np.ones((cfg.adjacency_kernel, cfg.adjacency_kernel), np.uint8)
    limit = cad_extent(cfg)[:2] * cfg.cad_size_margin if cfg.use_cad_constraint else None
    rng = np.random.default_rng(0)

    def points_of(m):
        p = pts[valid_idx[m[v_v, v_u]]]
        if len(p) > cfg.max_points_per_fragment:
            p = p[rng.choice(len(p), cfg.max_points_per_fragment, replace=False)]
        return p

    tried = set()
    while True:
        areas = [int(m.sum()) for m in masks]
        small = [i for i in np.argsort(areas) if areas[i] < cfg.absorb_area and i not in tried]
        if not small:
            return masks
        s = small[0]
        tried.add(s)
        dil_s = cv2.dilate(masks[s].astype(np.uint8), kernel) > 0
        best, best_gap = None, cfg.absorb_depth_thresh
        for j, m in enumerate(masks):
            if j == s or not np.any(dil_s & m):
                continue
            dil_j = cv2.dilate(m.astype(np.uint8), kernel) > 0
            gap = boundary_depth_gap(masks[s], m, dil_s, dil_j, pts, valid_idx, v_u, v_v)
            if gap is None or gap >= best_gap:
                continue
            if limit is not None:
                ext = principal_extent(np.vstack([points_of(masks[s]), points_of(m)]),
                                       cfg.extent_percentile)
                if np.any(ext[:2] > limit):
                    continue
            best, best_gap = j, gap
        if best is not None:
            masks[best] |= masks[s]
            del masks[s]
            # インデックスが変わるので，吸収先が変わった可能性のある切れ端を再挑戦させる
            tried = set()
