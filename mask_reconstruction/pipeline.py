"""提案手法のパイプライン全体．

FastSAM 断片 → LingBot-Depth 深度補完・点群化 → 背景断片除去 → 3D形状に基づく断片統合
"""
import torch
from ultralytics import FastSAM

from .config import Config
from .depth import depth_to_points, load_lingbot, refine_depth, remove_depth_outliers
from .fragments import filter_fragments, run_fastsam
from .merging import merge_fragments


def compute_depth(lingbot, sample, cfg: Config, device):
    """設定に応じた深度 [mm]：LingBot-Depth / 時間（空間）メディアン（改善1）/ 飛び値除去のみ．"""
    if cfg.depth_refine:
        return refine_depth(lingbot, sample, cfg, device)
    if cfg.depth_temporal:
        from .depth_temporal import prepare_depth
        return prepare_depth(sample, cfg)
    return remove_depth_outliers(sample.depth, cfg)


class MaskReconstructor:
    def __init__(self, cfg: Config = Config(), device=None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lingbot = load_lingbot(cfg, self.device)
        self.fastsam = FastSAM(cfg.fastsam_weights)

    def fragments(self, sample):
        """FastSAM の断片マスク（比較手法 "FastSAM" の出力でもある）．"""
        return run_fastsam(self.fastsam, sample.rgb, self.cfg, self.device)

    def reconstruct(self, sample, fragments=None):
        """マスクを再構築する．fragments を渡すと FastSAM の推論を省略する．"""
        if fragments is None:
            fragments = self.fragments(sample)
        depth_mm = compute_depth(self.lingbot, sample, self.cfg, self.device)
        pts, valid_idx, v_u, v_v = depth_to_points(depth_mm, sample, self.cfg)
        kept = filter_fragments(fragments, sample.rgb, v_u, v_v, valid_idx, self.cfg)
        cam = (sample.fx, sample.fy, sample.cx, sample.cy)
        return merge_fragments(kept, pts, valid_idx, v_u, v_v, self.cfg, cam=cam)
