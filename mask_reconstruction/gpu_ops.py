"""深度の前処理と点群化の GPU 版（Config.use_gpu_ops / 実機ノードの --gpu で使う）．

どの関数も CPU 版と同じ値を返すように書いてある（tests/test_gpu_ops.py で確認）．
  - temporal_median_depth : 時間方向の中央値（live_pose_node.temporal_median_depth と同じ）
  - remove_depth_outliers : cv2.medianBlur（端は延長）との差が大きい画素を 0 にする（depth.remove_depth_outliers と同じ）
  - depth_to_points       : Open3D の create_from_depth_image と同じ計算で点群化（depth.depth_to_points と同じ）
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_gpu_u16(a):
    # uint16 は torch で演算できないので int32 にして送る
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.int32)).to(_DEV, non_blocking=True)


def temporal_median_depth(frames, min_valid=0.5):
    """深度（uint16，0 は欠損）の画素ごとの中央値．欠損のフレームは除き，有効が min_valid 未満の画素は 0．"""
    st, _ = torch.sort(torch.stack([_to_gpu_u16(f) for f in frames]), dim=0)
    n = st.shape[0]
    c = (st != 0).sum(dim=0)
    k = n - c
    hi = torch.gather(st, 0, torch.clamp(k + c // 2, max=n - 1)[None])[0].double()
    lo = torch.gather(st, 0, torch.clamp(k + (c - 1) // 2, 0, n - 1)[None])[0].double()
    med = torch.where(c % 2 == 1, hi, (lo + hi) / 2.0)
    med[c < min_valid * n] = 0
    return torch.round(med).to(torch.int32).cpu().numpy().astype(np.uint16)


def remove_depth_outliers(depth, cfg: Config):
    """メディアンフィルタ（ksize×ksize，端は延長）との差が depth_outlier_thresh を超える画素を 0 にする．"""
    k = cfg.median_ksize
    d = _to_gpu_u16(depth)
    padded = F.pad(d[None, None].float(), (k // 2,) * 4, mode="replicate")[0, 0]
    win = padded.unfold(0, k, 1).unfold(1, k, 1).reshape(*d.shape, k * k)
    med = win.kthvalue(k * k // 2 + 1, dim=-1).values          # 奇数個の中央値（uint16 の値は float32 で正確に表せる）
    diff = (d.float() - med).abs()
    out = d.clone()
    out[(out > 0) & (diff > cfg.depth_outlier_thresh)] = 0
    return out.cpu().numpy().astype(np.uint16)


def depth_to_points(depth_mm, sample, cfg: Config):
    """depth.depth_to_points と同じ結果（pts, valid_idx, v_u, v_v）を GPU で計算する．

    Open3D と同じく，z は float32 に丸めた depth/1000（depth_trunc を超えたら 0），x = (u - cx) * z / fx（倍精度）．
    点は行優先の順（z > 0 の画素）．画像内に投影し直せて z > 0.001 の点を valid とする．
    """
    h, w = sample.shape
    d = _to_gpu_u16(depth_mm).double()
    # GPU では「スカラーで割る」が「逆数を掛ける」に置き換えられて丸めが変わるので，割る数を配列にして本当に割る
    z = (d / torch.full_like(d, 1000.0)).float()
    z[z > cfg.depth_trunc] = 0
    v, u = torch.nonzero(z > 0, as_tuple=True)                # 行優先の順
    zz = z[v, u].double()
    x = (u.double() - sample.cx) * zz / torch.full_like(zz, sample.fx)
    y = (v.double() - sample.cy) * zz / torch.full_like(zz, sample.fy)
    pts = torch.stack([x, y, zz], dim=1)
    uu = torch.round(sample.fx * x / zz + sample.cx).long()
    vv = torch.round(sample.fy * y / zz + sample.cy).long()
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h) & (zz > 0.001)
    valid_idx = torch.nonzero(ok, as_tuple=True)[0]
    return (pts.cpu().numpy(), valid_idx.cpu().numpy(), uu[valid_idx].cpu().numpy(), vv[valid_idx].cpu().numpy())
