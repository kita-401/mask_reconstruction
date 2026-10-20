"""改善4：閾値決定のための断片対の分析（GT のある10配置）．

    python scripts/analyze_pairs.py

1. 隣接する断片対を「同一 GT 部品」「別 GT 部品」に分類する（断片の GT は最大重なりで決める）
2. 段差・concave_ratio・concave_angle・統合後サイズ比（l_i/L_i, d/D_cad, A/A_cad）・基本コストの
   ヒストグラムと散布図を保存する
3. τ_r, τ_a（凹判定），m（サイズ制約），β（凸の減算量）を leave-one-scene-out で選ぶ
   基準は2クラスの balanced accuracy（同一部品の正解率と別部品の正解率の平均）

出力: results/pair_analysis/{pairs.csv, loso.json, *.png}
"""
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, load_sample  # noqa: E402
from mask_reconstruction.cad_prior import load_cad_prior  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.constrained_merging import SizeChecker, pair_table  # noqa: E402
from mask_reconstruction.depth import depth_to_points  # noqa: E402
from mask_reconstruction.fragments import filter_fragments  # noqa: E402
from mask_reconstruction.pipeline import compute_depth  # noqa: E402

CFG = Config(intrinsics_ref_size=(848, 480), depth_refine=False, depth_temporal=True, use_convexity=True)
OUT = RESULTS_DIR / "pair_analysis"
SAME, DIFF = "#2a78d6", "#eb6834"   # 同一部品 / 別部品
SIZE_KEYS = ("L1", "L2", "L3", "D", "A")


def assign_gt(masks, gts):
    """断片ごとの GT 番号（最大重なり，重ならなければ -1）と，その重なり率．"""
    out = []
    for m in masks:
        ov = [np.sum(m & g) for g in gts]
        j = int(np.argmax(ov)) if gts else -1
        out.append((j if gts and ov[j] > 0 else -1, ov[j] / m.sum() if gts and ov[j] > 0 else 0.0))
    return out


def collect():
    recon = MaskReconstructor(CFG)
    rows = []
    for i in range(1, 11):
        s = load_sample(i, CFG)
        d = compute_depth(recon.lingbot, s, CFG, recon.device)
        pts, vi, vu, vv = depth_to_points(d, s, CFG)
        kept = filter_fragments(recon.fragments(s), s.rgb, vu, vv, vi, CFG)
        T = pair_table(kept, pts, vi, vu, vv, CFG, convexity=True)
        sc = SizeChecker(kept, pts, vi, vu, vv, CFG, (s.fx, s.fy, s.cx, s.cy))
        gt = assign_gt(kept, s.gt_masks)
        for k in range(len(kept)):
            for l in range(k + 1, len(kept)):
                if not T.adjacent[k, l] or gt[k][0] < 0 or gt[l][0] < 0:
                    continue
                r = sc.ratios([k, l])
                rows.append(dict(scene=i, k=k, l=l, same=int(gt[k][0] == gt[l][0]),
                                 purity=min(gt[k][1], gt[l][1]),
                                 area_min=int(min(kept[k].sum(), kept[l].sum())),
                                 gap_mm=float(T.gap[k, l] * 1000), base=float(T.base[k, l]),
                                 cratio=float(T.cratio[k, l]), cangle=float(T.cangle[k, l]),
                                 n_samples=int(T.n_samples[k, l]),
                                 **{f"r_{key}": float(r.get(key, np.nan)) for key in SIZE_KEYS}))
        print(f"config {i}: {len(kept)} fragments", flush=True)
    return rows


def bal_acc(pred_diff, same):
    same = np.asarray(same, bool)
    tpr = np.mean(pred_diff[~same]) if (~same).any() else np.nan   # 別部品を「別」と判定
    tnr = np.mean(~pred_diff[same]) if same.any() else np.nan      # 同一部品を「同じ」と判定
    return 0.5 * (tpr + tnr), tnr, tpr


def loso(rows, grid, rule):
    """leave-one-scene-out．rule(rows_sub, params) → 「別部品」と判定する bool 配列．"""
    scenes = sorted({r["scene"] for r in rows})
    folds = []
    for s in scenes:
        train = [r for r in rows if r["scene"] != s]
        test = [r for r in rows if r["scene"] == s]
        y_tr = np.array([r["same"] for r in train])
        best = max(grid, key=lambda p: (bal_acc(rule(train, p), y_tr)[0], -grid.index(p)))
        y_te = np.array([r["same"] for r in test])
        ba, tnr, tpr = bal_acc(rule(test, best), y_te) if len(test) else (np.nan,) * 3
        folds.append(dict(scene=s, params=best, test_bal_acc=ba, test_same_kept=tnr, test_diff_split=tpr,
                          n_test=len(test)))
    y = np.array([r["same"] for r in rows])
    best_all = max(grid, key=lambda p: (bal_acc(rule(rows, p), y)[0], -grid.index(p)))
    ba, tnr, tpr = bal_acc(rule(rows, best_all), y)
    return dict(folds=folds, all_data=dict(params=best_all, bal_acc=ba, same_kept=tnr, diff_split=tpr),
                loso_mean_bal_acc=float(np.nanmean([f["test_bal_acc"] for f in folds])))


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def plot_hist(rows, key, label, path, bins):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for same, color, name in ((1, SAME, "同一部品"), (0, DIFF, "別部品")):
        v = col([r for r in rows if r["same"] == same], key)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, histtype="step", linewidth=2, color=color, label=f"{name} (n={len(v)})",
                density=True)
    ax.set_xlabel(label)
    ax.set_ylabel("密度")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(rows, kx, ky, lx, ly, path, lines=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    for same, color, name, mk in ((1, SAME, "同一部品", "o"), (0, DIFF, "別部品", "^")):
        sub = [r for r in rows if r["same"] == same]
        ax.scatter(col(sub, kx), col(sub, ky), s=16, c=color, marker=mk, alpha=0.6, label=name,
                   edgecolors="white", linewidths=0.5)
    for axis, v, style in (lines or []):
        (ax.axvline if axis == "x" else ax.axhline)(v, color="0.4", linestyle=style, linewidth=1)
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "TakaoGothic", "DejaVu Sans"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    prior = load_cad_prior(CFG)
    gap_thr = CFG.bound_depth_thresh * 1000
    cont = [r for r in rows if np.isfinite(r["gap_mm"]) and r["gap_mm"] < gap_thr and r["n_samples"] > 0]

    # --- 図 ---
    plot_hist(rows, "gap_mm", "境界の段差 [mm]", OUT / "hist_gap.png", np.linspace(0, 60, 31))
    plot_hist(cont, "cratio", "concave_ratio（段差 < 8mm の対）", OUT / "hist_cratio.png", np.linspace(0, 1, 21))
    plot_hist([r for r in cont if r["cangle"] > 0], "cangle", "concave_angle [deg]（段差 < 8mm, 凹あり）",
              OUT / "hist_cangle.png", np.linspace(0, 180, 19))
    plot_hist(cont, "base", "基本コスト（段差 < 8mm の対）", OUT / "hist_base.png", np.linspace(0, 15, 31))
    for key in SIZE_KEYS:
        plot_hist(rows, f"r_{key}", f"2断片を統合したときのサイズ比 {key}", OUT / f"hist_size_{key}.png",
                  np.linspace(0, 2.5, 26))
    plot_scatter(cont, "cratio", "cangle", "concave_ratio", "concave_angle [deg]", OUT / "scatter_convexity.png",
                 [("y", prior.theta_max + CFG.concave_angle_margin, "--")])
    plot_scatter(rows, "r_L1", "r_D", "l1 / L1", "d / D_cad", OUT / "scatter_size.png",
                 [("x", 1 + CFG.size_margin, "--"), ("y", 1 + CFG.size_margin, "--")])
    plot_scatter(cont, "base", "cratio", "基本コスト", "concave_ratio", OUT / "scatter_base_cratio.png",
                 [("x", CFG.cluster_thresh, "--")])

    # --- leave-one-scene-out ---
    def rule_concave(rs, p):
        tr, ta = p
        return (col(rs, "cratio") > tr) & (col(rs, "cangle") > ta)

    def rule_size(rs, m):
        R = np.stack([col(rs, f"r_{k}") for k in SIZE_KEYS], axis=1)
        return np.any(np.nan_to_num(R) > 1 + m, axis=1)

    def rule_beta(rs, beta):
        # 凸・平坦とみなした対：base − β > cluster_thresh なら（その対だけでは）統合されない
        return col(rs, "base") - beta > CFG.cluster_thresh

    grid_c = [(tr, ta) for tr in np.round(np.arange(0.0, 1.0, 0.1), 2) for ta in range(15, 181, 5)]
    grid_m = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    grid_b = list(np.round(np.arange(0.0, 12.01, 0.25), 2))
    res = dict(
        n_pairs=len(rows), n_same=int(sum(r["same"] for r in rows)),
        n_continuous=len(cont), n_continuous_same=int(sum(r["same"] for r in cont)),
        cad=dict(L_mm=[x * 1000 for x in prior.L], D_mm=prior.D * 1000, A_cm2=prior.A * 1e4,
                 theta_max=prior.theta_max),
        concave=loso(cont, grid_c, rule_concave),
        concave_cad_default=dict(zip(("bal_acc", "same_kept", "diff_split"), bal_acc(
            rule_concave(cont, (CFG.concave_ratio_thresh, prior.theta_max + CFG.concave_angle_margin)),
            col(cont, "same").astype(bool)))),
        size=loso(rows, grid_m, rule_size),
        beta=loso(cont, grid_b, rule_beta),
    )
    with open(OUT / "loso.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for key in ("concave", "size", "beta"):
        r = res[key]
        print(f"{key}: 全データでの最良 {r['all_data']}  LOSO の平均 balanced acc {r['loso_mean_bal_acc']:.3f}")
        print("   折ごとの選択:", [f["params"] for f in r["folds"]])
    print("凹判定（CAD由来の既定 τ_r=%.1f, τ_a=%.1f）:" % (CFG.concave_ratio_thresh,
                                                     prior.theta_max + CFG.concave_angle_margin),
          res["concave_cad_default"])
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
