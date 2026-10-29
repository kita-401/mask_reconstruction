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
        # LingBot-Depth を使わない設定（depth_refine=False）なら，モデルを読み込まない（GPU メモリと起動時間の節約）
        self.lingbot = load_lingbot(cfg, self.device) if cfg.depth_refine else None
        self.fastsam = FastSAM(cfg.fastsam_weights)

    def fragments(self, sample):
        """FastSAM の断片マスク（比較手法 "FastSAM" の出力でもある）．"""
        return run_fastsam(self.fastsam, sample.rgb, self.cfg, self.device)

    def reconstruct(self, sample, fragments=None):
        """マスクを再構築する．fragments を渡すと FastSAM の推論を省略する．

        各ステップの時間 [s] を self.last_times に残す（処理には影響しない）．
        """
        import time
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        times = {}
        t = time.perf_counter()
        if fragments is None:
            fragments = self.fragments(sample)
        sync()
        times["FastSAM"], t = time.perf_counter() - t, time.perf_counter()
        depth_mm = compute_depth(self.lingbot, sample, self.cfg, self.device)
        self.last_depth = depth_mm   # 統合に使った深度 [mm]（LingBot-Depth の補完後など）．姿勢推定側で使えるように残す
        sync()
        times["depth"], t = time.perf_counter() - t, time.perf_counter()
        if self.cfg.use_gpu_ops:   # 同じ点群を GPU で計算する（gpu_ops.py）
            from .gpu_ops import depth_to_points as depth_to_points_gpu
            pts, valid_idx, v_u, v_v = depth_to_points_gpu(depth_mm, sample, self.cfg)
        else:
            pts, valid_idx, v_u, v_v = depth_to_points(depth_mm, sample, self.cfg)
        times["points"], t = time.perf_counter() - t, time.perf_counter()
        kept = filter_fragments(fragments, sample.rgb, v_u, v_v, valid_idx, self.cfg, pts=pts)
        times["filter"], t = time.perf_counter() - t, time.perf_counter()
        cam = (sample.fx, sample.fy, sample.cx, sample.cy)
        merged = merge_fragments(kept, pts, valid_idx, v_u, v_v, self.cfg, cam=cam)
        times["merge"] = time.perf_counter() - t
        times["n_fragments"], times["n_kept"] = len(fragments), len(kept)
        self.last_times = times
        return merged
