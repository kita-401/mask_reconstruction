"""データセットの読み込み（RGB・深度・内部パラメータ・COCOアノテーション）．"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, Config


@dataclass
class Sample:
    idx: int
    rgb: np.ndarray        # BGR, (h_in, w_in, 3)
    depth: np.ndarray      # uint16 [mm], (h_in, w_in)
    gt_masks: list         # list[bool ndarray]
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def shape(self):
        return self.rgb.shape[:2]


def sample_paths(idx, data_dir=DATA_DIR):
    """配置番号 idx (1〜10) のファイルパス．1番だけ接尾辞なし．"""
    d = Path(data_dir)
    sfx = "" if idx == 1 else f"_{idx}"
    return {
        "rgb": d / f"real_rgb{sfx}.png",
        "depth": d / f"real_depth{sfx}.png",
        "intrinsics": d / f"camera_intrinsics{sfx}.json",
        "annotation": d / f"annotation{idx}.json",
    }


def input_size(img_shape, patch_size=14):
    return (img_shape[0] // patch_size) * patch_size, (img_shape[1] // patch_size) * patch_size


def parse_coco(json_path, img_shape, patch_size=14):
    """COCOポリゴンを bool マスク (h_in, w_in) のリストに変換する．"""
    with open(json_path) as f:
        coco = json.load(f)
    if len(coco["images"]) > 0:
        img_shape = (coco["images"][0]["height"], coco["images"][0]["width"])
    h_in, w_in = input_size(img_shape, patch_size)

    gt_masks = []
    for ann in coco["annotations"]:
        if len(ann["segmentation"]) == 0:
            continue
        mask = np.zeros(img_shape, dtype=np.uint8)
        for poly in ann["segmentation"]:
            cv2.fillPoly(mask, [np.array(poly, np.int32).reshape((-1, 1, 2))], 1)
        gt_masks.append(cv2.resize(mask, (w_in, h_in), interpolation=cv2.INTER_NEAREST) > 0)
    return gt_masks, (h_in, w_in)


def load_sample(idx, cfg=Config(), data_dir=DATA_DIR):
    p = sample_paths(idx, data_dir)
    rgb = cv2.imread(str(p["rgb"]))
    depth = cv2.imread(str(p["depth"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"config {idx}: {p['rgb']} / {p['depth']}")

    if p["annotation"].exists():
        gt_masks, (h_in, w_in) = parse_coco(p["annotation"], rgb.shape[:2], cfg.patch_size)
    else:
        gt_masks, (h_in, w_in) = [], input_size(rgb.shape[:2], cfg.patch_size)

    with open(p["intrinsics"]) as f:
        cam = json.load(f)
    ref_w, ref_h = cfg.intrinsics_ref_size
    return Sample(
        idx=idx,
        rgb=cv2.resize(rgb, (w_in, h_in)),
        depth=cv2.resize(depth, (w_in, h_in), interpolation=cv2.INTER_NEAREST),
        gt_masks=gt_masks,
        fx=cam["fx"] * (w_in / ref_w), fy=cam["fy"] * (h_in / ref_h),
        cx=cam["cx"] * (w_in / ref_w), cy=cam["cy"] * (h_in / ref_h),
    )
