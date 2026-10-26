"""姿勢推定用の点群化の GPU 版（実機ノードの --gpu で使う）．pointcloud.py の mask_to_points + preprocess と同じ処理．

  1. マスクを mask_erode_px だけ縮め（cv2.erode と同じく画像の外は縮めない），有効な深度の画素を3次元点にする
  2. ボクセルで間引く（Open3D の voxel_down_sample と同じ：最小点 − voxel/2 を原点に格子を切り，ボクセル内の点の平均）
  3. 統計的外れ値除去（Open3D の remove_statistical_outlier と同じ：自分を含む近傍 nb 点までの平均距離が，
     全点の平均 + std_ratio × 標準偏差（N−1 で割る）未満の点を残す）
Open3D は間引いた点を内部の順序で並べるので，点の並び順は CPU 版と異なる（点の集合は同じ．tests/test_pose_gpu.py）．
点の平均の足し算の順序が GPU では異なるので，座標は最後の桁（1e-16 程度）だけ違うことがある．
"""
import torch
import torch.nn.functional as F

from .config import PoseConfig

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def erode(mask_t, r):
    """正方形（一辺 2r+1）での収縮．画像の外は「マスクの内側」とみなす（cv2.erode の既定と同じ）．"""
    if r <= 0:
        return mask_t
    inv = (~mask_t).float()[None, None]
    return F.max_pool2d(inv, 2 * r + 1, stride=1, padding=r)[0, 0] == 0


def mask_points(mask_t, depth_t, K, cfg: PoseConfig):
    """マスク（GPU の bool 画像）内の点 (N, 3) [m]．depth_t は GPU の深度画像（mm，int32）．"""
    m = erode(mask_t, cfg.mask_erode_px)
    v, u = torch.nonzero(m, as_tuple=True)
    z = depth_t[v, u].double()
    z = z / torch.full_like(z, cfg.depth_scale)
    ok = (z > cfg.depth_min) & (z < cfg.depth_max)
    u, v, z = u[ok].double(), v[ok].double(), z[ok]
    x = (u - K[0, 2]) * z / torch.full_like(z, K[0, 0])
    y = (v - K[1, 2]) * z / torch.full_like(z, K[1, 1])
    return torch.stack([x, y, z], dim=1)


def voxel_down_sample(p, voxel):
    if len(p) == 0:
        return p
    minb = p.min(dim=0).values - voxel * 0.5
    idx = torch.floor((p - minb) / torch.full_like(p, voxel)).long()
    dims = idx.max(dim=0).values + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]   # 3次元の番号を1つの整数に（1次元の unique は速い）
    _, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    sums = torch.zeros((len(counts), 3), dtype=p.dtype, device=p.device).index_add_(0, inv, p)
    return sums / counts[:, None].to(p.dtype)


def remove_statistical_outlier(p, nb, std_ratio, margin=4):
    """近傍は，単精度で多めの候補（nb + margin 点）を選び，候補だけ倍精度で距離を計算し直して近い nb 点を決める
    （RTX 系の GPU は倍精度の総当たりが遅いため．候補に本当の近傍が入っていれば結果は倍精度の総当たりと同じ）．"""
    k = min(nb, len(p))
    kc = min(nb + margin, len(p))
    c = p - p.mean(dim=0)
    cand = torch.cdist(c.float(), c.float()).topk(kc, dim=1, largest=False).indices      # (N, kc)
    d = torch.linalg.norm(p[cand] - p[:, None, :], dim=2)                                # 倍精度で計算し直す
    avg = d.topk(k, dim=1, largest=False).values.mean(dim=1)   # 自分（距離 0）を含む
    valid = avg > 0
    n = int(valid.sum())
    if n < 2:
        return p
    mean = avg[valid].mean()
    std = torch.sqrt(((avg[valid] - mean) ** 2).sum() / (n - 1))
    return p[valid & (avg < mean + std_ratio * std)]


def preprocess(p, cfg: PoseConfig):
    p = voxel_down_sample(p, cfg.voxel_size)
    if cfg.outlier_nb > 0 and len(p) > cfg.outlier_nb:
        p = remove_statistical_outlier(p, cfg.outlier_nb, cfg.outlier_std)
    return p


def batch_mask_points_preprocess(masks, depth_np, K, cfg: PoseConfig):
    """全部品のマスクをまとめて処理する．戻り値：(マスク内の点のリスト, 間引き・外れ値除去後の点のリスト)（numpy）．

    min_points 未満の部品は，間引き後の点を None にする（CPU 版と同じく推定しない）．
    """
    import numpy as np
    if not masks:
        return [], []
    M = torch.from_numpy(np.stack(masks)).to(_DEV)
    if cfg.mask_erode_px > 0:
        r = cfg.mask_erode_px
        M = F.max_pool2d((~M).float()[:, None], 2 * r + 1, stride=1, padding=r)[:, 0] == 0
    mid, v, u = torch.nonzero(M, as_tuple=True)              # 部品ごと，行優先の順（部品ごとに処理した場合と同じ順）
    dt = torch.from_numpy(depth_np.astype(np.int32)).to(_DEV)
    z = dt[v, u].double()
    z = z / torch.full_like(z, cfg.depth_scale)
    ok = (z > cfg.depth_min) & (z < cfg.depth_max)
    mid, u, v, z = mid[ok], u[ok].double(), v[ok].double(), z[ok]
    x = (u - K[0, 2]) * z / torch.full_like(z, K[0, 0])
    y = (v - K[1, 2]) * z / torch.full_like(z, K[1, 1])
    P = torch.stack([x, y, z], dim=1)
    counts = torch.bincount(mid, minlength=len(masks)).cpu().numpy()
    raw = list(np.split(P.cpu().numpy(), np.cumsum(counts)[:-1]))
    out = []
    starts = np.concatenate([[0], np.cumsum(counts)])
    for k in range(len(masks)):
        if counts[k] < cfg.min_points:
            out.append(None)
            continue
        out.append(preprocess(P[starts[k]:starts[k + 1]], cfg).cpu().numpy())
    return raw, out


def to_gpu_depth(depth_np):
    return torch.from_numpy(depth_np.astype("int32")).to(_DEV)


def to_gpu_mask(mask_np):
    return torch.from_numpy(mask_np).to(_DEV)
