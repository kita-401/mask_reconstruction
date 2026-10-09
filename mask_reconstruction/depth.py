"""深度の前処理と LingBot-Depth による深度補完，点群化．"""
import sys

import cv2
import numpy as np
import open3d as o3d
import torch

from .config import LINGBOT_DIR, Config


def load_lingbot(cfg: Config, device):
    if str(LINGBOT_DIR) not in sys.path:
        sys.path.append(str(LINGBOT_DIR))
    from mdm.model.v2 import MDMModel

    model = MDMModel.from_pretrained(cfg.lingbot_model).to(device)
    model.eval()
    return model


def remove_depth_outliers(depth, cfg: Config):
    """メディアンフィルタとの差が大きい画素（飛び値）を 0（欠損）にする．"""
    median = cv2.medianBlur(depth, cfg.median_ksize)
    diff = np.abs(depth.astype(np.float32) - median.astype(np.float32))
    out = depth.copy()
    out[(out > 0) & (diff > cfg.depth_outlier_thresh)] = 0
    return out


def refine_depth(model, sample, cfg: Config, device):
    """欠損を含む深度を LingBot-Depth で補完し，uint16 [mm] で返す．"""
    h, w = sample.shape
    intr = np.array([[sample.fx / w, 0, sample.cx / w],
                     [0, sample.fy / h, sample.cy / h],
                     [0, 0, 1]], dtype=np.float32)
    intr_t = torch.tensor(intr, dtype=torch.float32, device=device).unsqueeze(0)
    img_t = torch.tensor(cv2.cvtColor(sample.rgb, cv2.COLOR_BGR2RGB) / 255.0,
                         dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    depth_f = remove_depth_outliers(sample.depth, cfg)
    depth_t = torch.tensor(np.nan_to_num(depth_f.astype(np.float32) / 1000.0),
                           dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model.infer(img_t, depth_in=depth_t, apply_mask=True,
                          intrinsics=intr_t, enable_depth_mask=False)
    return (out["depth"].squeeze().cpu().numpy() * 1000.0).astype(np.uint16)


def depth_to_points(depth_mm, sample, cfg: Config):
    """深度画像を点群化し，各点の画素座標 (u, v) との対応も返す．

    Returns:
        pts: (N, 3) 点群
        valid_idx: 画像内に投影される点のインデックス
        v_u, v_v: valid_idx に対応する画素座標
    """
    h, w = sample.shape
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, sample.fx, sample.fy, sample.cx, sample.cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_mm), intr, depth_scale=1000.0, depth_trunc=cfg.depth_trunc)
    pts = np.asarray(pcd.points)

    z = pts[:, 2]
    u = np.round(sample.fx * pts[:, 0] / z + sample.cx).astype(int)
    v = np.round(sample.fy * pts[:, 1] / z + sample.cy).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0.001)
    valid_idx = np.where(valid)[0]
    return pts, valid_idx, u[valid_idx], v[valid_idx]
