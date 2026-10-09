"""CAD仮説によるセグメンテーション（cad_hypothesis）を評価し，2段階グラフ統合と比較する．

    python scripts/eval_cad_hypothesis.py

既存コード（mask_reconstruction）は読み込み・断片作成・比較対象の統合にだけ使う．
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_hypothesis import CADHypothesisSegmenter, HypConfig  # noqa: E402
from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.depth import depth_to_points, refine_depth  # noqa: E402
from mask_reconstruction.evaluation import METRICS  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.merging import merge_fragments  # noqa: E402

FIXED = Config(intrinsics_ref_size=(848, 480))
TWO_STAGE = replace(FIXED, merge_method="graph", normal_angle_thresh=20, axis_min_area=5000,
                    use_cad_constraint=True, cad_size_margin=1.2, two_stage=True, stage1_min_area=3000)
BASE = HypConfig()
# 選択のパラメータだけを変える（仮説の生成・評価に関わる explain_tol などはここでは変えない）
VARIANTS = {
    "cad-hyp (default)": BASE,
    "cad-hyp min_expl 4000": replace(BASE, min_explained=4000),
    "cad-hyp min_expl 6000": replace(BASE, min_explained=6000),
    "cad-hyp viol<0.3": replace(BASE, max_violation_ratio=0.3),
}
HYBRID_OVERLAP = 0.5   # 既存手法のまとまりのうち，CAD仮説のマスクと重なる割合がこれ未満のものを補う


def hybrid(hyp_masks, clusters):
    """CAD仮説で得たマスクを優先し，説明されなかった領域は既存手法（2段階グラフ統合）のまとまりで補う．"""
    if not hyp_masks:
        return list(clusters)
    covered = np.any(np.stack(hyp_masks), axis=0)
    extra = [c for c in clusters if np.sum(c & covered) / c.sum() < HYBRID_OVERLAP]
    return list(hyp_masks) + extra


def iou(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else 0.0


def main():
    recon = MaskReconstructor(FIXED)
    seg = CADHypothesisSegmenter(BASE)
    scores = {"2stage graph": [], **{k: [] for k in VARIANTS}, **{k + " +hybrid": [] for k in VARIANTS}}
    diag = {"gt": 0, "gt_with_good_hyp": 0, "n_hyp": [], "time": []}
    for i in range(1, 11):
        s = load_sample(i, FIXED)
        frags = recon.fragments(s)
        d = refine_depth(recon.lingbot, s, FIXED, recon.device)
        pts, vi, vu, vv = depth_to_points(d, s, FIXED)
        kept = filter_fragments(frags, s.rgb, vu, vv, vi, FIXED)
        cam = (s.fx, s.fy, s.cx, s.cy)
        clusters = merge_fragments(kept, pts, vi, vu, vv, TWO_STAGE, cam=cam)
        scores["2stage graph"].append({"n": len(clusters), **evaluate_masks(clusters, s.gt_masks)})

        seeds = clusters + [f for f in kept if f.sum() >= BASE.seed_min_area]
        t0 = time.time()
        res = seg(kept, seeds, pts, vi, vu, vv, cam)
        diag["time"].append(time.time() - t0)
        diag["n_hyp"].append(res.n_hypotheses)
        for name, cfg in VARIANTS.items():
            seg.cfg = cfg
            r = res if cfg == BASE else seg(kept, seeds, pts, vi, vu, vv, cam, hyps=res.hypotheses)
            scores[name].append({"n": len(r.masks), **evaluate_masks(r.masks, s.gt_masks)})
            hm = hybrid(r.masks, clusters)
            scores[name + " +hybrid"].append({"n": len(hm), **evaluate_masks(hm, s.gt_masks)})
        seg.cfg = BASE

        # 仮説の網羅性：各正解部品について，説明画素が IoU>=0.5 で一致する仮説があるか
        for g in s.gt_masks:
            diag["gt"] += 1
            if any(iou(h.explained, g) >= 0.5 for h in res.hypotheses):
                diag["gt_with_good_hyp"] += 1
        print(f"config {i}: {res.n_hypotheses} hyps, selected {len(res.selected)}, "
              f"{diag['time'][-1]:.1f}s", flush=True)

    print(f"\n仮説数/画像 {np.mean(diag['n_hyp']):.0f}, 処理時間/画像 {np.mean(diag['time']):.1f}s, "
          f"良い仮説（説明画素と正解の IoU>=0.5）がある正解部品 {diag['gt_with_good_hyp']}/{diag['gt']}")
    print(f"{'method':>30} {'mIoU%':>6} {'Prec%':>6} {'Rec%':>6} {'over':>5} {'under':>5} {'#inst':>6}")
    summary = {}
    for name, rows in scores.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in METRICS + ("n",)}
        summary[name] = {"mean": m, "per_config": rows}
        print(f"{name:>30} {m['miou']*100:6.2f} {m['prec']*100:6.2f} {m['rec']*100:6.2f} "
              f"{m['over']:5.1f} {m['under']:5.1f} {m['n']:6.1f}")
    out = RESULTS_DIR / "cad_hypothesis_eval.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "diag": diag}, f, indent=1)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
