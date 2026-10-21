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
from mask_reconstruction import constrained_merging as cm  # noqa: E402
from mask_reconstruction.pipeline import compute_depth  # noqa: E402

# 点群のスケールを CAD と比べるため，内部パラメータは正しいスケール（848x480）で統一する
BASE = Config(intrinsics_ref_size=(848, 480))
IMP1 = replace(BASE, depth_refine=False, depth_temporal=True)
IMP2 = dict(use_size_constraint=True)
IMP3 = dict(use_convexity=True, no_force_connect=True)
IMP5 = dict(post_close=True)
# 凹判定の閾値：既定は CAD 由来（τ_a = θ_cad_max + 10°）．LOSO は analyze_pairs.py で選んだ値
LOSO = dict(concave_ratio_thresh=0.5, concave_angle_thresh=35.0)

VARIANTS = {
    "base": ("1. ベースライン（既存手法）", BASE),
    "imp1": ("2. +改善1（LingBotなし・メディアン）", IMP1),
    "imp12": ("3. +改善1+2（サイズ制約）", replace(IMP1, **IMP2)),
    "imp13": ("4. +改善1+3（凸性・強制0.1廃止）", replace(IMP1, **IMP3)),
    "imp123": ("5. +改善1+2+3（全部）", replace(IMP1, **IMP2, **IMP3)),
    "imp1235": ("6. +改善1+2+3+5（後処理）", replace(IMP1, **IMP2, **IMP3, **IMP5)),
    "imp13_loso": ("4'. +改善1+3（τ: LOSO）", replace(IMP1, **IMP3, **LOSO)),
    "imp123_loso": ("5'. +改善1+2+3（τ: LOSO）", replace(IMP1, **IMP2, **IMP3, **LOSO)),
    "imp1235_loso": ("6'. +改善1+2+3+5（τ: LOSO）", replace(IMP1, **IMP2, **IMP3, **IMP5, **LOSO)),
    "imp12_e3": ("3e. +改善1+2（寸法は3px収縮）", replace(IMP1, **IMP2, size_erode_px=3)),
    "imp123_e3": ("5e. +改善1+2+3（寸法は3px収縮）", replace(IMP1, **IMP2, **IMP3, size_erode_px=3)),
    "imp123_loso_e3": ("5e'. +改善1+2+3（τ: LOSO，3px収縮）",
                       replace(IMP1, **IMP2, **IMP3, **LOSO, size_erode_px=3)),
    # LingBot-Depth ありで改善2・3を使う場合
    "lb2": ("L2. LingBot +改善2", replace(BASE, **IMP2)),
    "lb3": ("L3. LingBot +改善3", replace(BASE, **IMP3)),
    "lb23": ("L23. LingBot +改善2+3", replace(BASE, **IMP2, **IMP3)),
    "lb23_loso": ("L23'. LingBot +改善2+3（τ: LOSO）", replace(BASE, **IMP2, **IMP3, **LOSO)),
    "lb23_e3": ("L23e. LingBot +改善2+3（寸法は3px収縮）", replace(BASE, **IMP2, **IMP3, size_erode_px=3)),
    # 高速化 B：hier_improved_v2 の法線を深度画像上の差分で求める
    "v2": ("hier_improved_v2", None),
    "v2_fast5": ("hier_improved_v2 + 高速な法線（窓 5px）", dict(fast_normals=True, fast_normal_window=5)),
    "v2_fast7": ("hier_improved_v2 + 高速な法線（窓 7px）", dict(fast_normals=True, fast_normal_window=7)),
    "v2_fast9": ("hier_improved_v2 + 高速な法線（窓 9px）", dict(fast_normals=True, fast_normal_window=9)),
}


VARIANTS["v2f"] = ("hier_improved_v2_fast（C++，LingBot-Depth あり）", dict(use_cpp=True))
VARIANTS["v2f_raw"] = ("hier_improved_v2_fast（C++，LingBot-Depth なし）", dict(use_cpp=True, depth_refine=False))
VARIANTS["v2f_raw_noaxis"] = ("hier_improved_v2_fast（C++，LingBot-Depth なし，円筒軸なし）",
                              dict(use_cpp=True, depth_refine=False, use_cylinder_axis=False))


def _v2_variants():
    from export_masks import METHODS
    for key in ("v2", "v2_fast5", "v2_fast7", "v2_fast9"):
        name, kw = VARIANTS[key]
        VARIANTS[key] = (name, replace(METHODS["hier_improved_v2"], **(kw or {})))
    for key in ("v2f", "v2f_raw", "v2f_raw_noaxis"):
        name, kw = VARIANTS[key]
        VARIANTS[key] = (name, replace(METHODS["hier_improved_v2_fast"], **kw))


sys.path.insert(0, str(Path(__file__).resolve().parent))
_v2_variants()


def fragment_labels(kept, masks):
    """各断片が含まれる統合マスクの番号（後処理で画素が増えても，断片を最も多く含むマスク）．"""
    return np.array([int(np.argmax([np.sum(f & m) for m in masks])) for f in kept])


def breakdown(kept, masks, gts, T, cfg, events, overlap=0.2):
    """残った過分割・過統合が，どの判定で起きたかの内訳．

    過分割（GT が2つ以上の予測に面積20%超ずつ覆われる）：その GT の断片で別々の予測に入った対のうち，
      距離最小の対に適用された判定（サイズ制約で止めた対は size:<条件>，隣接しない対は non_adjacent）
    過統合（予測が2つ以上の GT にまたがる）：1つの断片が2つ以上の GT にまたがれば fastsam_fragment，
      そうでなければ，別々の GT の断片で同じ予測に入った対のうち距離最小の対の判定
    """
    from collections import Counter
    D = cm.distance_from_table(T, cfg)
    lab = fragment_labels(kept, masks)
    gts = [g for g in gts if g.sum() > 0]
    fg = [int(np.argmax([np.sum(f & g) for g in gts])) if any(np.sum(f & g) for g in gts) else -1 for f in kept]
    blocked = {}
    for a, b, _, viol in events:
        if viol:
            for x in a:
                for y in b:
                    blocked[(min(x, y), max(x, y))] = "size:" + "+".join(viol)
    over, under = Counter(), Counter()
    for gi, g in enumerate(gts):
        cover = [j for j, m in enumerate(masks) if np.sum(g & m) / g.sum() > overlap]
        if len(cover) < 2:
            continue
        cand = [(D[x, y], x, y) for x in range(len(kept)) for y in range(x + 1, len(kept))
                if fg[x] == gi and fg[y] == gi and lab[x] != lab[y] and lab[x] in cover and lab[y] in cover]
        if not cand:
            over["other(断片なし)"] += 1
            continue
        _, x, y = min(cand)
        over[blocked.get((x, y)) or cm.pair_rule(T, x, y, cfg)] += 1
    for j, m in enumerate(masks):
        hit = [gi for gi, g in enumerate(gts) if np.sum(g & m) / m.sum() > overlap]
        if len(hit) < 2:
            continue
        members = [x for x in range(len(kept)) if lab[x] == j]
        if any(sum(np.sum(kept[x] & gts[gi]) / kept[x].sum() > overlap for gi in hit) >= 2 for x in members):
            under["fastsam_fragment"] += 1
            continue
        cand = [(D[x, y], x, y) for x in members for y in members if x < y and fg[x] != fg[y]
                and fg[x] in hit and fg[y] in hit]
        if not cand:
            under["other"] += 1
            continue
        _, x, y = min(cand)
        under[cm.pair_rule(T, x, y, cfg)] += 1
    return dict(over), dict(under)


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
            kept = filter_fragments(frags, s.rgb, vu, vv, vi, cfg, pts=pts)
            tkey = ("table",) + dkey
            if tkey not in depth_cache:   # 内訳用の断片対の特徴（深度ごとに1回）
                depth_cache[tkey] = cm.pair_table(kept, pts, vi, vu, vv, replace(cfg, use_convexity=True),
                                                  convexity=True)
            cm.LAST_INFO.clear()
            t0 = time.perf_counter()
            masks = merge_fragments(kept, pts, vi, vu, vv, cfg, cam=(s.fx, s.fy, s.cx, s.cy))
            t_merge = (time.perf_counter() - t0) * 1000
            if "D" in cm.LAST_INFO:   # 内訳に使う距離行列が実際の統合と同じであることの確認
                assert np.allclose(cm.LAST_INFO["D"], cm.distance_from_table(depth_cache[tkey], cfg))
            over_c, under_c = breakdown(kept, masks, s.gt_masks, depth_cache[tkey], cfg,
                                        cm.LAST_INFO.get("events", []))
            rows[key].append({"config": i, "n": len(masks), "n_frag": len(kept),
                              "depth_ms": t_depth, "merge_ms": t_merge,
                              "size_stops": dict(cm.LAST_INFO.get("stops", {})),
                              "over_causes": over_c, "under_causes": under_c,
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
    for key, rs in rows.items():
        tot = {}
        for r in rs:
            for k, v in r.get("size_stops", {}).items():
                tot[k] = tot.get(k, 0) + v
        if tot:
            lines.append(f"\n{VARIANTS[key][0]}：サイズ制約で止めた統合（10配置合計，1回で複数条件あり）: {tot}")
    lines.append("\n残った過分割・過統合の原因（10配置合計）")
    lines.append("| 構成 | 過分割の原因 | 過統合の原因 |\n|---|---|---|")
    for key, rs in rows.items():
        agg = []
        for f in ("over_causes", "under_causes"):
            tot = {}
            for r in rs:
                for k, v in r[f].items():
                    tot[k] = tot.get(k, 0) + v
            agg.append(", ".join(f"{k} {v}" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))
        lines.append(f"| {VARIANTS[key][0]} | {agg[0]} | {agg[1]} |")
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
