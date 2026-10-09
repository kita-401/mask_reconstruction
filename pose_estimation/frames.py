"""RGB-D フレームとマスクファイルの読み込み．

マスクファイルの形式（セグメンテーション側との唯一の取り決め）:
    <mask_dir>/config{i}.npy : bool ndarray (N, H, W)  N = インスタンス数
H, W はセグメンテーション側の入力解像度（現状 476×840）．RGB-D はこの解像度に合わせて読み込む．
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR


@dataclass
class Frame:
    rgb: np.ndarray     # BGR (H, W, 3)
    depth: np.ndarray   # uint16 (H, W)
    K: np.ndarray       # (3, 3) 内部パラメータ（H, W に合わせてスケーリング済み）

    @property
    def shape(self):
        return self.depth.shape


def frame_paths(idx, data_dir=DATA_DIR):
    d = Path(data_dir)
    sfx = "" if idx == 1 else f"_{idx}"
    return d / f"real_rgb{sfx}.png", d / f"real_depth{sfx}.png", d / f"camera_intrinsics{sfx}.json"


def load_frame(idx, size_hw, data_dir=DATA_DIR) -> Frame:
    """配置 idx の RGB-D を size_hw=(H, W) にリサイズして読み込む．"""
    rgb_p, depth_p, intr_p = frame_paths(idx, data_dir)
    rgb = cv2.imread(str(rgb_p))
    depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"{rgb_p} / {depth_p}")
    with open(intr_p) as f:
        cam = json.load(f)

    h0, w0 = depth.shape
    h, w = size_hw
    sx, sy = w / w0, h / h0
    K = np.array([[cam["fx"] * sx, 0, cam["cx"] * sx],
                  [0, cam["fy"] * sy, cam["cy"] * sy],
                  [0, 0, 1]])
    return Frame(rgb=cv2.resize(rgb, (w, h)),
                 depth=cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST),
                 K=K)


def load_masks(mask_dir, idx):
    """マスクファイルを読み込む．存在しなければ None．"""
    p = Path(mask_dir) / f"config{idx}.npy"
    if not p.exists():
        return None
    m = np.load(p)
    return [mi.astype(bool) for mi in m] if m.ndim == 3 else []
