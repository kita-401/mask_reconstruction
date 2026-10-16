"""改善1〜5（2026-10-16）のアブレーション評価．10配置の平均±標準偏差．

    python scripts/eval_improvements.py                      # 全構成
    python scripts/eval_improvements.py --variants base imp1 # 一部だけ

FastSAM は配置ごとに1回だけ実行し，深度・統合の設定だけを変える．
時間は 深度[ms]（LingBot-Depth またはメディアン）と 統合[ms]（後処理を含む）を分けて記録する．
出力: results/eval_improvements/summary.json, table.md
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.depth import depth_to_points  # noqa: E402
from mask_reconstruction.evaluation import METRICS  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.merging import merge_fragments  # noqa: E402
from mask_reconstruction.pipeline import compute_depth  # noqa: E402

# 点群のスケールを CAD と比べるため，内部パラメータは正しいスケール（848x480）で統一する
BASE = Config(intrinsics_ref_size=(848, 480))
IMP1 = replace(BASE, depth_refine=False, depth_temporal=True)
IMP2 = dict(use_size_constraint=True)
IMP3 = dict(use_convexity=True, no_force_connect=True)
IMP5 = dict(post_close=True)

VARIANTS = {
    "base": ("1. ベースライン（既存手法）", BASE),
    "imp1": ("2. +改善1（LingBotなし・メディアン）", IMP1),
    "imp12": ("3. +改善1+2（サイズ制約）", replace(IMP1, **IMP2)),
    "imp13": ("4. +改善1+3（凸性・強制0.1廃止）", replace(IMP1, **IMP3)),
    "imp123": ("5. +改善1+2+3（全部）", replace(IMP1, **IMP2, **IMP3)),
    "imp1235": ("6. +改善1+2+3+5（後処理）", replace(IMP1, **IMP2, **IMP3, **IMP5)),
}


def run(variant_keys, configs):
    recon = MaskReconstructor(BASE)
    rows = {k: [] for k in variant_keys}
    for i in configs:
        frags = None
        depth_cache = {}
        for key in variant_keys:
            cfg = VARIANTS[key][1]
            s = load_sample(i, cfg)
            if frags is None:
                frags = recon.fragments(s)
            dkey = (cfg.depth_refine, cfg.depth_temporal)
            if dkey not in depth_cache:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                d = compute_depth(recon.lingbot, s, cfg, recon.device)
                torch.cuda.synchronize()
                depth_cache[dkey] = (d, (time.perf_counter() - t0) * 1000)
            d, t_depth = depth_cache[dkey]
            pts, vi, vu, vv = depth_to_points(d, s, cfg)
            kept = filter_fragments(frags, s.rgb, vu, vv, vi, cfg)
            t0 = time.perf_counter()
            masks = merge_fragments(kept, pts, vi, vu, vv, cfg, cam=(s.fx, s.fy, s.cx, s.cy))
            t_merge = (time.perf_counter() - t0) * 1000
            rows[key].append({"config": i, "n": len(masks), "n_frag": len(kept),
                              "depth_ms": t_depth, "merge_ms": t_merge,
                              **evaluate_masks(masks, s.gt_masks)})
        print(f"config {i} done", flush=True)
    return rows


def table(rows):
    lines = ["| 構成 | mIoU [%] | Precision [%] | Recall [%] | 過分割 | 過統合 | マスク数 | 深度 [ms] | 統合 [ms] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, rs in rows.items():
        def ms(k, scale=1.0, fmt="{:.1f}"):
            v = np.array([r[k] for r in rs]) * scale
            return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"
        lines.append(f"| {VARIANTS[key][0]} | {ms('miou', 100)} | {ms('prec', 100)} | {ms('rec', 100)} | "
                     f"{ms('over')} | {ms('under')} | {ms('n')} | {ms('depth_ms', fmt='{:.0f}')} | "
                     f"{ms('merge_ms', fmt='{:.0f}')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="eval_improvements")
    args = ap.parse_args()
    rows = run(args.variants, args.configs)
    out = RESULTS_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    md = table(rows)
    print(md)
    (out / "table.md").write_text(md + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
