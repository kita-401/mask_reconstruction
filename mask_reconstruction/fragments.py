"""FastSAM による断片マスク生成と，背景断片の除去．"""
import cv2
import numpy as np

from .config import Config


def run_fastsam(model, rgb, cfg: Config, device):
    """FastSAM の全マスクのうち，面積 min_fragment_area 以上のものを返す．"""
    h, w = rgb.shape[:2]
    res = model.predict(rgb, device=device, retina_masks=True, conf=cfg.fastsam_conf,
                        iou=cfg.fastsam_iou, imgsz=cfg.fastsam_imgsz, verbose=False)
    masks = []
    if res[0].masks is not None:
        for m in res[0].masks.data.cpu().numpy():
            mr = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if np.sum(mr) > cfg.min_fragment_area:
                masks.append(mr)
    return masks


def run_yolo(model, rgb, device):
    """比較用：YOLO26-seg（ゼロショット）のマスク．"""
    h, w = rgb.shape[:2]
    res = model.predict(rgb, device=device, verbose=False)
    if res[0].masks is None:
        return []
    return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            for m in res[0].masks.data.cpu().numpy()]


def fit_plane(pts, cfg: Config):
    """点群で最も大きな平面 (n, d)（n·p + d = 0, |n| = 1）．乱数の種を固定して毎回同じ結果にする．"""
    import open3d as o3d
    o3d.utility.random.seed(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(cfg.plane_voxel)
    (a, b, c, d), _ = pcd.segment_plane(cfg.plane_dist_thresh, 3, cfg.plane_ransac_iter)
    n = np.array([a, b, c])
    return n / np.linalg.norm(n), d / np.linalg.norm(n)


def filter_fragments(masks, rgb, v_u, v_v, valid_idx, cfg: Config, pts=None):
    """黒い背景（トレイ）が主体の断片と，有効な3D点が少ない断片を除く．

    use_plane_removal=True なら，点の大半が最大平面（机・トレイの面）上にある断片も除く（pts が必要）．
    """
    t = cfg.black_thresh
    is_black = (rgb[:, :, 0] < t) & (rgb[:, :, 1] < t) & (rgb[:, :, 2] < t)
    on_plane = None
    if cfg.use_plane_removal:
        if pts is None:
            raise ValueError("use_plane_removal=True には filter_fragments(..., pts=点群) が必要")
        n, d = fit_plane(pts[valid_idx], cfg)
        on_plane = np.zeros(rgb.shape[:2], bool)
        on_plane[v_v, v_u] = np.abs(pts[valid_idx] @ n + d) < cfg.plane_dist_thresh
    if cfg.use_cpp and masks:   # 同じ判定を C++ の集計で行う
        from . import cpp_kernels as ck
        has_point = np.zeros(rgb.shape[:2], bool)
        has_point[v_v, v_u] = True
        imgs = [np.ones(rgb.shape[:2], bool), is_black, has_point] + ([on_plane] if on_plane is not None else [])
        cnt = ck.mask_counts(ck.stack_masks(masks), imgs)
        kept = []
        for m, c in zip(masks, cnt):
            if c[1] / c[0] > cfg.max_black_ratio or c[2] <= cfg.min_fragment_points:
                continue
            if on_plane is not None and c[3] / c[2] > cfg.plane_max_ratio:
                continue
            kept.append(m)
        return kept
    kept = []
    for m in masks:
        if np.sum(is_black[m]) / np.sum(m) > cfg.max_black_ratio:
            continue
        n_valid = len(valid_idx[m[v_v, v_u]])
        if n_valid <= cfg.min_fragment_points:
            continue
        if on_plane is not None and np.sum(on_plane[m]) / n_valid > cfg.plane_max_ratio:
            continue
        kept.append(m)
    return kept
