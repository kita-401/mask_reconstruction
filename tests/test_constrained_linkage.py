"""制約なしの自前平均連結が scipy（linkage + fcluster）と同じ分割になることの確認．

    python tests/test_constrained_linkage.py            # 乱数の距離行列（同点なし）
    python tests/test_constrained_linkage.py --real     # 実データ10配置の距離行列（同点あり）も確認
"""
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction.constrained_merging import constrained_average_linkage  # noqa: E402


def same_partition(a, b):
    """ラベルの付け方によらず，同じ分割か．"""
    pairs_a = {(x, y) for x in range(len(a)) for y in range(x + 1, len(a)) if a[x] == a[y]}
    pairs_b = {(x, y) for x in range(len(b)) for y in range(x + 1, len(b)) if b[x] == b[y]}
    return pairs_a == pairs_b


def scipy_labels(D, t):
    return fcluster(linkage(squareform(D, checks=False), method="average"), t=t, criterion="distance")


def test_random(n_trials=300):
    rng = np.random.default_rng(0)
    for trial in range(n_trials):
        n = int(rng.integers(2, 40))
        X = rng.normal(size=(n, 3)) * rng.uniform(0.5, 3)
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        t = float(rng.uniform(0.3, 3.0))
        ours, _ = constrained_average_linkage(D, t)
        assert same_partition(ours, scipy_labels(D, t)), f"trial {trial} differs"
    print(f"random: {n_trials} trials identical")


def test_real():
    from mask_reconstruction import Config, MaskReconstructor, load_sample
    from mask_reconstruction.depth import depth_to_points
    from mask_reconstruction.fragments import filter_fragments
    from mask_reconstruction.merging import distance_matrix
    from mask_reconstruction.pipeline import compute_depth

    for cfg_name, cfg in [("7/31(LingBot)", Config(intrinsics_ref_size=(848, 480))),
                          ("改善1", Config(intrinsics_ref_size=(848, 480), depth_refine=False,
                                         depth_temporal=True))]:
        recon = MaskReconstructor(cfg)
        n_same = 0
        for i in range(1, 11):
            s = load_sample(i, cfg)
            d = compute_depth(recon.lingbot, s, cfg, recon.device)
            pts, vi, vu, vv = depth_to_points(d, s, cfg)
            kept = filter_fragments(recon.fragments(s), s.rgb, vu, vv, vi, cfg)
            D = distance_matrix(kept, pts, vi, vu, vv, cfg)
            ours, _ = constrained_average_linkage(D, cfg.cluster_thresh)
            ok = same_partition(ours, scipy_labels(D, cfg.cluster_thresh))
            n_same += ok
            if not ok:
                print(f"  {cfg_name} config {i}: differs (同点の統合順の違い)")
        print(f"real {cfg_name}: {n_same}/10 configs identical")


if __name__ == "__main__":
    test_random()
    if "--real" in sys.argv:
        test_real()
