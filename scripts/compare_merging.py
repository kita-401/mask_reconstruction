"""断片統合の方式を比較する．深度補完と FastSAM は配置ごとに1回だけ計算し，統合方法だけを変える．

    python scripts/compare_merging.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.depth import depth_to_points, refine_depth, remove_depth_outliers  # noqa: E402
from mask_reconstruction.evaluation import METRICS  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.merging import cad_extent, merge_fragments  # noqa: E402

LEGACY = Config()
FIXED = Config(intrinsics_ref_size=(848, 480))


def cad(cfg, m):
    return replace(cfg, use_cad_constraint=True, cad_size_margin=m, cad_remerge=True)


def axis(t, min_area=0):
    return replace(FIXED, use_cylinder_axis=True, axis_dist_thresh=t, axis_min_area=min_area)


def absorb(cfg, area, depth=0.008):
    return replace(cfg, absorb_small=True, absorb_area=area, absorb_depth_thresh=depth)


CURRENT = cad(axis(0.008, 5000), 1.2)   # 現在の推奨設定
RECALL = cad(axis(0.008), 1.1)          # Recall 重視

def graph(margin=1.2, bridge=True, template=False, **kw):
    return replace(FIXED, merge_method="graph", use_cad_constraint=True, cad_size_margin=margin,
                   **{"axis_min_area": 5000, "use_bridge": bridge, "use_template_check": template, **kw})


GRAPH = graph(1.2, bridge=False, normal_angle_thresh=20)


def two(area=3000, depth=0.008, normal=20.0, keep=False, base=GRAPH):
    return replace(base, two_stage=True, stage1_min_area=area, stage2_depth_thresh=depth,
                   stage2_normal_thresh=normal, stage2_keep_unassigned=keep)


# 7/31 の手法（階層クラスタリング）に改良を1つずつ足していく
H_NORMAL = replace(FIXED, hier_normal_check=True, normal_angle_thresh=20)
H_TIE = replace(H_NORMAL, hier_tiebreak=True)
H_CAD = replace(H_TIE, use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True)
H_2STAGE = replace(H_CAD, two_stage=True, stage1_min_area=3000)
H_2STAGE_AXIS = replace(H_2STAGE, use_cylinder_axis=True, axis_min_area=5000)
G_2STAGE = two(3000)


def raw(cfg):
    return replace(cfg, depth_refine=False)


HA = H_2STAGE_AXIS   # 現在の提案手法（7/31 + 全部入り）

HA = H_2STAGE_AXIS   # 現在の提案手法（7/31 + 全部入り）
HA12 = replace(HA, axis_dist_thresh=0.012)

HA = H_2STAGE_AXIS   # 現在の提案手法（7/31 + 全部入り）
HA12O = replace(HA, axis_dist_thresh=0.012, hier_overlap_link=True, overlap_link_ratio=0.5)


def mc(cfg, t):
    return replace(cfg, hier_model_check=True, model_check_min_ratio=t)


# 半径などを手で設定せず，モデル点群から自動で求める
VARIANTS = {
    "提案(軸12mm+重なり) 半径=手入力14.6mm": HA12O,
    "提案(軸12mm+重なり) 半径=モデルから自動": replace(HA12O, auto_model_shape=True),
    "提案手法 半径=手入力": HA,
    "提案手法 半径=自動": replace(HA, auto_model_shape=True),
}


def main():
    print("CAD extent [mm]:", np.round(cad_extent(FIXED) * 1000, 1))
    recon = MaskReconstructor(LEGACY)
    scores = {k: [] for k in VARIANTS}
    times = {k: [] for k in VARIANTS}

    for i in range(1, 11):
        cache = {}
        for name, cfg in VARIANTS.items():
            key = (cfg.intrinsics_ref_size, cfg.depth_refine)
            if key not in cache:
                s = load_sample(i, cfg)
                if "frags" not in cache:
                    cache["frags"] = recon.fragments(s)
                d = refine_depth(recon.lingbot, s, cfg, recon.device) if cfg.depth_refine \
                    else remove_depth_outliers(s.depth, cfg)
                pts, vi, vu, vv = depth_to_points(d, s, cfg)
                kept = filter_fragments(cache["frags"], s.rgb, vu, vv, vi, cfg)
                cache[key] = (s, kept, pts, vi, vu, vv)
            s, kept, pts, vi, vu, vv = cache[key]
            t0 = time.time()
            masks = merge_fragments(kept, pts, vi, vu, vv, cfg, cam=(s.fx, s.fy, s.cx, s.cy))
            times[name].append(time.time() - t0)
            scores[name].append({"n": len(masks), **evaluate_masks(masks, s.gt_masks)})
        print(f"config {i} done", flush=True)

    print(f"\n{'variant':>34} {'mIoU%':>6} {'Prec%':>6} {'Rec%':>6} {'over':>5} {'under':>5} "
          f"{'#inst':>6} {'merge[s]':>9}")
    summary = {}
    for name, rows in scores.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in METRICS + ("n",)}
        m["merge_time"] = float(np.mean(times[name]))
        summary[name] = {"mean": m, "per_config": rows}
        print(f"{name:>34} {m['miou']*100:6.2f} {m['prec']*100:6.2f} {m['rec']*100:6.2f} "
              f"{m['over']:5.1f} {m['under']:5.1f} {m['n']:6.1f} {m['merge_time']:9.3f}")

    out = RESULTS_DIR / "compare_merging_auto.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
