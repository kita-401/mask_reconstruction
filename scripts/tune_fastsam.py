"""FastSAM の設定を変えて，断片の数・統合の上限・統合後の精度を比較する．

    python scripts/tune_fastsam.py

上限（oracle）: 各断片を最も重なる正解部品に割り当ててまとめた場合の評価値．
断片が粗くなりすぎて複数部品にまたがると，この値が下がる．
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.depth import depth_to_points, refine_depth  # noqa: E402
from mask_reconstruction.fragments import filter_fragments, run_fastsam  # noqa: E402
from mask_reconstruction.merging import merge_fragments  # noqa: E402

FIXED = Config(intrinsics_ref_size=(848, 480))
CURRENT = replace(FIXED, use_cylinder_axis=True, axis_min_area=5000,
                  use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True)
GRAPH = replace(FIXED, merge_method="graph", normal_angle_thresh=20, axis_min_area=5000,
                use_cad_constraint=True, cad_size_margin=1.2)

SETTINGS = {
    "default (640, c0.3, iou0.8, a1000)": {},
    "512 a1000": dict(fastsam_imgsz=512),
    **{f"640 a{a}": dict(min_fragment_area=a) for a in (2000, 3000, 4000)},
    **{f"512 a{a}": dict(fastsam_imgsz=512, min_fragment_area=a) for a in (2000, 3000, 4000)},
}
MERGERS = {"baseline": FIXED, "graph": GRAPH}


def oracle_merge(frags, gts):
    groups = {}
    for f in frags:
        ov = [np.sum(f & g) / np.sum(f) for g in gts]
        j = int(np.argmax(ov))
        if ov[j] > 0.5:
            groups.setdefault(j, []).append(f)
    return [np.any(np.stack(v), axis=0) for v in groups.values()]


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def main():
    recon = MaskReconstructor(FIXED)
    res = {name: {"frag": [], "small": [], "oracle": [], **{m: [] for m in MERGERS}} for name in SETTINGS}
    for i in range(1, 11):
        s = load_sample(i, FIXED)
        d = refine_depth(recon.lingbot, s, FIXED, recon.device)
        pts, vi, vu, vv = depth_to_points(d, s, FIXED)
        cam = (s.fx, s.fy, s.cx, s.cy)
        for name, kw in SETTINGS.items():
            fcfg = replace(FIXED, **kw)
            frags = run_fastsam(recon.fastsam, s.rgb, fcfg, recon.device)
            kept = filter_fragments(frags, s.rgb, vu, vv, vi, fcfg)
            r = res[name]
            r["frag"].append(len(kept))
            r["small"].append(sum(int(m.sum()) < 5000 for m in kept))
            r["oracle"].append(evaluate_masks(oracle_merge(kept, s.gt_masks), s.gt_masks))
            for mname, mcfg in MERGERS.items():
                masks = merge_fragments(kept, pts, vi, vu, vv, replace(mcfg, **kw), cam=cam)
                r[mname].append({"n": len(masks), **evaluate_masks(masks, s.gt_masks)})
        print(f"config {i} done", flush=True)

    print(f"\n{'setting':>36} {'#frag':>6} {'#small':>6} | {'oracle P/R/IoU':>16} | "
          f"{'baseline P/R/IoU (n)':>22} | {'graph P/R/IoU (n)':>22}")
    summary = {}
    for name, r in res.items():
        o = r["oracle"]
        line = (f"{name:>36} {np.mean(r['frag']):6.1f} {np.mean(r['small']):6.1f} | "
                f"{mean(o, 'prec')*100:4.0f}/{mean(o, 'rec')*100:3.0f}/{mean(o, 'miou')*100:3.0f}     | ")
        summary[name] = {"frag": float(np.mean(r["frag"])), "small": float(np.mean(r["small"])),
                         "oracle": {k: mean(o, k) for k in ("prec", "rec", "miou")}}
        for m in MERGERS:
            x = r[m]
            line += (f"{mean(x, 'prec')*100:5.1f}/{mean(x, 'rec')*100:5.1f}/{mean(x, 'miou')*100:5.1f}"
                     f" ({mean(x, 'n'):4.1f}) | ")
            summary[name][m] = {k: mean(x, k) for k in ("prec", "rec", "miou", "over", "under", "n")}
            summary[name][m]["per_config"] = x
        print(line)
    out = RESULTS_DIR / "tune_fastsam_baseline.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
