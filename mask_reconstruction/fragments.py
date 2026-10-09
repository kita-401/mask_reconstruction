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


def filter_fragments(masks, rgb, v_u, v_v, valid_idx, cfg: Config):
    """黒い背景（トレイ）が主体の断片と，有効な3D点が少ない断片を除く．"""
    t = cfg.black_thresh
    is_black = (rgb[:, :, 0] < t) & (rgb[:, :, 1] < t) & (rgb[:, :, 2] < t)
    kept = []
    for m in masks:
        if np.sum(is_black[m]) / np.sum(m) > cfg.max_black_ratio:
            continue
        if len(valid_idx[m[v_v, v_u]]) > cfg.min_fragment_points:
            kept.append(m)
    return kept
