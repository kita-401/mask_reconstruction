"""セグメンテーションのマスクごとに，MaskNet でモデル点群を計測点群に合わせて削ってから位置合わせし，姿勢を推定する．

    python scripts/run_masknet_pose.py                         # hier_improved_v2 のマスク
    python scripts/run_masknet_pose.py --masks hier_improved_v2 --configs 1 9

手法（どれも同じマスクを使う．位置合わせの前に，マスク領域の点群とモデル点群を同じ辺長 VOXEL のボクセルで間引く）
  svd_icp_2mm  : 参考．既存の SVD+ICP（カメラ側半分のモデル，ボクセル 2mm の既定設定）
  svd_icp      : 既存の SVD+ICP（カメラ側半分のモデル），ボクセル VOXEL（--voxel，既定 5mm）
  full_svd     : MaskNet なし．全周モデルのまま SVD（符号探索）→ ICP（MaskNet の効果を切り分けるため）
  masknet_svd  : MaskNet で全周モデルを計測点群に合わせて削る → SVD（符号探索）→ ICP
                 （masksvd_exp の「中心化＋半径1に拡大＋符号探索＋ICP 1cm」）

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

VOXEL = 0.005                     # [m] 計測点群・モデル点群に共通のボクセルの辺長
BASE = MaskSVDConfig()
DIRECT = replace(BASE, normalize_scale=True, center_inputs=True, flip_mode="search", icp_threshold=0.01)
EVAL_MODEL = BASE.model_path      # 評価は全手法とも同じ全周モデルで行う
METHODS = ("svd_icp_2mm", "svd_icp", "full_svd", "masknet_svd")
STATS = {"n_measured": [], "n_model": [], "measured_ge_model": 0}


def estimate_all(masks, frame, pcs, svd_icp, halves, masknet):
    """マスクごとに各手法の姿勢．戻り値: {手法: [(T or None, 時間[s], 計測点群)]}"""
    out = {m: [] for m in METHODS}
    full = np.asarray(masknet.model.voxel_down_sample(VOXEL).points)
    for mk in masks:
        p_raw = mask_to_points(mk, frame, pcs["v"])
        if len(p_raw) < pcs["v"].min_points:
            for m in METHODS:
                out[m].append((None, 0.0, None))
            continue
        p2 = preprocess(p_raw, pcs["2mm"])
        p = preprocess(p_raw, pcs["v"])
        STATS["n_measured"].append(len(p))
        STATS["n_model"].append(len(full))
        STATS["measured_ge_model"] += int(len(p) >= len(full))

        def run(name, fn, pts):
            t0 = time.perf_counter()
            T = fn()
            out[name].append((T, time.perf_counter() - t0, pts))

        run("svd_icp_2mm", lambda: svd_icp["2mm"](halves["2mm"], p2).T_model_to_cam, p2)
        run("svd_icp", lambda: svd_icp["v"](halves["v"], p).T_model_to_cam, p)
        run("full_svd", lambda: masknet.register(p, template=full), p)
        run("masknet_svd", lambda: masknet.register(p), p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="hier_improved_v2")
    ap.add_argument("--voxel", type=float, default=0.005, help="計測・モデル点群に共通のボクセルの辺長 [m]")
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    args = ap.parse_args()

    global VOXEL
    VOXEL = args.voxel
    pcs = {"2mm": PoseConfig(), "v": replace(PoseConfig(), voxel_size=VOXEL)}
    svd_icp = {k: SVDICPRegistration(c) for k, c in pcs.items()}
    halves = {k: load_model(c) for k, c in pcs.items()}
    pc = pcs["v"]
    masknet = MaskSVD(replace(DIRECT, model_voxel=VOXEL))
    full_vis = np.asarray(o3d.io.read_point_cloud(EVAL_MODEL).voxel_down_sample(0.003).points)
    full_adds = np.asarray(o3d.io.read_point_cloud(EVAL_MODEL).voxel_down_sample(0.002).points)
    out_root = POSE_RESULTS_DIR.parent / "pose_masknet" / args.masks
    if VOXEL != 0.005:
        out_root = out_root.with_name(f"{args.masks}_voxel{VOXEL * 1000:g}mm")
    rows = {m: [] for m in METHODS}

    for i in args.configs:
        masks = load_masks(MASKS_DIR / args.masks, i)
        gt = load_masks(MASKS_DIR / "gt", i)
        frame = load_frame(i, masks[0].shape)
        depth_m = frame.depth.astype(np.float64) / 1000.0
        est = estimate_all(masks, frame, pcs, svd_icp, halves, masknet)
        est_gt = estimate_all(gt, frame, pcs, svd_icp, halves, masknet) if gt is not None else None
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
            vis_model = halves["2mm"] if m.startswith("svd_icp") else full_vis
            cv2.imwrite(str(d / f"config{i}.png"), draw_poses(frame, vis_model, poses, masks))
        print(f"config {i}: {len(masks)} masks", flush=True)

    names = {"svd_icp_2mm": "参考：既存 SVD+ICP（半分モデル，ボクセル 2mm）",
             "svd_icp": "既存 SVD+ICP（半分モデル，{v}）",
             "full_svd": "MaskNet なし：全周モデル → SVD → ICP（{v}）",
             "masknet_svd": "MaskNet で削る → SVD → ICP（{v}）"}
    v = f"{VOXEL * 1000:g}mm"
    names = {k: t.format(v=v) for k, t in names.items()}
    nm, nd = np.array(STATS["n_measured"]), np.array(STATS["n_model"])
    lines = [f"マスク: {args.masks}（{len(rows['svd_icp'])} 個）．{v} ボクセル後の点数：計測 中央値 {np.median(nm):.0f}，"
             f"モデル {nd[0]}．計測点数 ≥ モデル点数（MaskNet の入力で計測点を間引く）: "
             f"{STATS['measured_ge_model']}/{len(nm)}（正解マスク分を含む）",
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
