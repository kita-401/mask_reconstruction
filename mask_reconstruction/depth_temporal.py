"""改善1：LingBot-Depth を使わない深度の前処理．

静止シーンを N フレーム撮影し，画素ごとに時間方向のメディアンを取る（値 0 のフレームは除外）．
有効フレームが depth_temporal_min_valid 未満の画素は欠損（0）とし，補完しない．
フレームが1枚しかない場合は，有効画素だけを使った空間メディアンのみを行う（欠損は欠損のまま）．
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import Config


def temporal_median(frames, cfg: Config):
    """frames: uint16 [mm] の (H, W) 配列のリスト．"""
    stack = np.stack([f.astype(np.float32) for f in frames[:cfg.depth_temporal_frames]])
    stack[stack == 0] = np.nan
    n_valid = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    med[n_valid < cfg.depth_temporal_min_valid * len(stack)] = 0
    return np.nan_to_num(med).round().astype(np.uint16)


def spatial_median_valid(depth, ksize):
    """有効画素（>0）だけで計算する空間メディアン．欠損画素は 0 のまま．"""
    r = ksize // 2
    d = depth.astype(np.float32)
    d[d == 0] = np.nan
    win = sliding_window_view(np.pad(d, r, constant_values=np.nan), (ksize, ksize))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(win.reshape(*depth.shape, -1), axis=2)
    out = np.where(depth > 0, med, 0)
    return np.nan_to_num(out).round().astype(np.uint16)


def prepare_depth(sample, cfg: Config):
    """LingBot-Depth なしの深度 [mm]．sample.depth_frames があれば時間メディアン．"""
    frames = getattr(sample, "depth_frames", None) or [sample.depth]
    if cfg.depth_temporal_frames > 1 and len(frames) > 1:
        return temporal_median(frames, cfg)
    return spatial_median_valid(frames[0], cfg.depth_spatial_ksize)
