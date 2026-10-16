"""セグメンテーションのマスクごとに，MaskNet でモデル点群を計測点群に合わせて削ってから位置合わせし，姿勢を推定する．

    python scripts/run_masknet_pose.py                         # hier_improved_v2 のマスク
    python scripts/run_masknet_pose.py --masks hier_improved_v2 --configs 1 9

手法（どれも同じマスク・同じ計測点群を使う）
  svd_icp       : 既存の SVD+ICP（カメラ側半分のモデル，MaskNet なし）．比較の基準
  masknet_svd   : MaskNet でモデルを削る → 削ったモデルと計測点群の主成分で回転（符号探索）→ ICP
                  （masksvd_exp の「中心化＋半径1に拡大＋符号探索＋ICP 1cm」）
  masknet_refine: 既存 SVD+ICP の姿勢でモデルを重ねる → MaskNet で削る → 削ったモデルで ICP をかけ直す

評価（真値姿勢がないため，masksvd_exp と同じ基準）
  FSV  : 全周モデルのうちカメラ側を向く面の点が，計測深度より 10mm 以上手前に来る割合（<5% 成功，>20% 失敗）
  fitness : 計測点のうち，推定姿勢のモデル表面から 3mm 以内の割合
  ADD-S : 同じ手法で正解マスクから求めた姿勢とのずれ（セグメンテーションの影響）

入力: results/masks/<masks>/config{i}.npy, results/masks/gt/config{i}.npy
出力: results/pose_masknet/<masks>/<手法>/config{i}.png, summary.json
既存コード（mask_reconstruction / pose_estimation / masksvd_exp / vision_ws）は読み込みにだけ使う．
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from masksvd_exp.evaluation import fitness, free_space_violation  # noqa: E402
from masksvd_exp.masksvd import MaskSVD, MaskSVDConfig  # noqa: E402
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.config import MASKS_DIR, POSE_RESULTS_DIR  # noqa: E402
from pose_estimation.evaluation import add_s, match_masks  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.visualization import draw_poses  # noqa: E402

BASE = MaskSVDConfig()
DIRECT = replace(BASE, normalize_scale=True, center_inputs=True, flip_mode="search", icp_threshold=0.01)
REFINE = replace(BASE, normalize_scale=True, icp_threshold=0.01, icp_iter=30)
EVAL_MODEL = BASE.model_path      # 評価は全手法とも同じ全周モデルで行う
METHODS = ("svd_icp", "masknet_svd", "masknet_refine")


def estimate_all(masks, frame, pc, svd_icp, half, direct, refine):
    """マスクごとに3手法の姿勢．戻り値: {手法: [(T or None, 時間[s], 計測点群)]}"""
    out = {m: [] for m in METHODS}
    for mk in masks:
        p = mask_to_points(mk, frame, pc)
        if len(p) < pc.min_points:
            for m in METHODS:
                out[m].append((None, 0.0, None))
            continue
        p = preprocess(p, pc)
        t0 = time.perf_counter()
        T0 = svd_icp(half, p).T_model_to_cam
        t_svd = time.perf_counter() - t0
        out["svd_icp"].append((T0, t_svd, p))
        t0 = time.perf_counter()
        T1 = direct.register(p)
        out["masknet_svd"].append((T1, time.perf_counter() - t0, p))
        t0 = time.perf_counter()
        T2 = refine.refine(p, T0)
        out["masknet_refine"].append((T2, t_svd + time.perf_counter() - t0, p))   # SVD+ICP の時間を含む
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="hier_improved_v2")
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    args = ap.parse_args()

    pc = PoseConfig()
    svd_icp = SVDICPRegistration(pc)
    half = load_model(pc)
    direct, refine = MaskSVD(DIRECT), MaskSVD(REFINE)
    full_vis = np.asarray(o3d.io.read_point_cloud(EVAL_MODEL).voxel_down_sample(0.003).points)
    full_adds = np.asarray(o3d.io.read_point_cloud(EVAL_MODEL).voxel_down_sample(0.002).points)
    out_root = POSE_RESULTS_DIR.parent / "pose_masknet" / args.masks
    rows = {m: [] for m in METHODS}

    for i in args.configs:
        masks = load_masks(MASKS_DIR / args.masks, i)
        gt = load_masks(MASKS_DIR / "gt", i)
        frame = load_frame(i, masks[0].shape)
        depth_m = frame.depth.astype(np.float64) / 1000.0
        est = estimate_all(masks, frame, pc, svd_icp, half, direct, refine)
        est_gt = estimate_all(gt, frame, pc, svd_icp, half, direct, refine) if gt is not None else None
        match = match_masks(masks, gt) if gt is not None else [-1] * len(masks)
        for m in METHODS:
            poses = []
            for k, (T, t, p) in enumerate(est[m]):
                poses.append(T)
                if T is None:
                    continue
                r = dict(config=i, mask=k, time_ms=t * 1000, gt=int(match[k]),
                         fsv=free_space_violation(EVAL_MODEL, T, depth_m, frame.K),
                         fitness=fitness(EVAL_MODEL, T, p), T=T.tolist())
                if match[k] >= 0 and est_gt[m][match[k]][0] is not None:
                    r["add_s_mm"] = add_s(full_adds, T, est_gt[m][match[k]][0]) * 1000
                rows[m].append(r)
            d = out_root / m
            d.mkdir(parents=True, exist_ok=True)
            vis_model = load_model(pc) if m == "svd_icp" else full_vis
            cv2.imwrite(str(d / f"config{i}.png"), draw_poses(frame, vis_model, poses, masks))
        print(f"config {i}: {len(masks)} masks", flush=True)

    names = {"svd_icp": "既存 SVD+ICP（半分モデル）", "masknet_svd": "MaskNet で削る → SVD → ICP",
             "masknet_refine": "SVD+ICP → MaskNet で削る → ICP"}
    lines = [f"マスク: {args.masks}（{len(rows['svd_icp'])} 個）",
             "| 手法 | 成功 FSV<5% | 失敗 FSV>20% | fitness | ADD-S 中央値 [mm] | ADD-S<5mm | 時間 [ms/個] |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for m in METHODS:
        rs = rows[m]
        fsv = np.array([r["fsv"] for r in rs])
        adds = np.array([r["add_s_mm"] for r in rs if "add_s_mm" in r])
        s = dict(n=len(rs), success=float(np.nanmean(fsv < 0.05)), fail=float(np.nanmean(fsv > 0.2)),
                 fitness=float(np.mean([r["fitness"] for r in rs])),
                 add_s_median=float(np.median(adds)) if len(adds) else None,
                 add_s_lt5=float(np.mean(adds < 5)) if len(adds) else None,
                 time_ms=float(np.mean([r["time_ms"] for r in rs])))
        summary[m] = s
        lines.append(f"| {names[m]} | {s['success']*100:.0f}% | {s['fail']*100:.0f}% | {s['fitness']:.3f} | "
                     f"{s['add_s_median']:.2f} | {s['add_s_lt5']*100:.0f}% | {s['time_ms']:.1f} |")
    md = "\n".join(lines)
    print(md)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "table.md").write_text(md + "\n")
    with open(out_root / "summary.json", "w") as f:
        json.dump(dict(summary=summary, per_mask=rows), f, indent=1, ensure_ascii=False)
    print(f"Saved: {out_root}")


if __name__ == "__main__":
    main()
