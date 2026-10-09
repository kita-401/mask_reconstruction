"""マスクファイルから各インスタンスの3次元姿勢を SVD + ICP で推定する．

    python scripts/run_pose_estimation.py --masks gt proposed
    python scripts/run_pose_estimation.py --masks proposed --model full

入力: results/masks/<name>/config{i}.npy（scripts/export_masks.py で作成）
出力: results/pose/<name>[_<model>]/config{i}.png, poses.json
--masks に gt を含めると，正解マスクでの姿勢を基準に他のマスクでの姿勢のずれ（ADD-S）も出す．
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation import PoseConfig, SVDICPRegistration  # noqa: E402
from pose_estimation.config import MASKS_DIR, PKG_DIR, POSE_RESULTS_DIR  # noqa: E402
from pose_estimation.evaluation import add_s, fit_quality, free_space_violation, match_masks  # noqa: E402
from pose_estimation.frames import load_frame, load_masks  # noqa: E402
from pose_estimation.pointcloud import load_model, mask_to_points, preprocess  # noqa: E402
from pose_estimation.visualization import draw_poses  # noqa: E402


def estimate(masks, frame, model, reg, cfg):
    rows, poses = [], []
    for k, m in enumerate(masks):
        pts = mask_to_points(m, frame, cfg)
        row = {"id": k, "n_points": int(len(pts))}
        if len(pts) < cfg.min_points:
            poses.append(None)
            rows.append({**row, "status": "too_few_points"})
            continue
        pts = preprocess(pts, cfg)
        t0 = time.time()
        r = reg(model, pts)
        fit, rmse = fit_quality(model, pts, r.T_model_to_cam, cfg.eval_threshold)
        poses.append(r.T_model_to_cam)
        rows.append({**row, "status": "ok", "time": time.time() - t0,
                     "svd_fitness": r.svd_fitness, "icp_fitness": r.icp_fitness,
                     "icp_rmse": r.icp_rmse, "fitness": fit, "rmse": rmse,
                     "fsv": free_space_violation(model, r.T_model_to_cam, frame, cfg.depth_scale),
                     "T": r.T_model_to_cam.tolist()})
    return rows, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["gt", "proposed"],
                    help="results/masks/ 以下のフォルダ名")
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--model", choices=["half", "full"], default="half")
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--hypotheses", choices=["original", "all_perms"], default="original")
    ap.add_argument("--icp-all", action="store_true", help="全初期候補で ICP し最良を採用")
    args = ap.parse_args()

    cfg = PoseConfig(model_path=str(PKG_DIR / "models" / f"t_pipe_{args.model}.pcd"))
    if args.voxel:
        cfg = replace(cfg, voxel_size=args.voxel)
    cfg = replace(cfg, hypotheses=args.hypotheses, icp_all_candidates=args.icp_all)
    reg = SVDICPRegistration(cfg)
    model = load_model(cfg)
    suffix = "" if args.model == "half" else f"_{args.model}"

    all_rows = {name: {} for name in args.masks}
    for i in args.configs:
        loaded = {name: load_masks(MASKS_DIR / name, i) for name in args.masks}
        shape = next(m[0].shape for m in loaded.values() if m)
        frame = load_frame(i, shape)
        poses_by = {}
        for name, masks in loaded.items():
            if masks is None:
                print(f"[skip] {MASKS_DIR / name}/config{i}.npy がありません")
                continue
            rows, poses = estimate(masks, frame, model, reg, cfg)
            poses_by[name] = (masks, poses)
            all_rows[name][i] = rows
            out = POSE_RESULTS_DIR / f"{name}{suffix}"
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"config{i}.png"), draw_poses(frame, model, poses, masks))

        # 正解マスクでの姿勢を基準にした ADD-S
        if "gt" in poses_by:
            gt_masks, gt_poses = poses_by["gt"]
            for name, (masks, poses) in poses_by.items():
                if name == "gt":
                    continue
                for row, j, T in zip(all_rows[name][i], match_masks(masks, gt_masks), poses):
                    row["gt_match"] = j
                    if j >= 0 and T is not None and gt_poses[j] is not None:
                        row["add_s_vs_gt"] = add_s(model, T, gt_poses[j])
        print(f"config {i} done")

    # --- 集計 ---
    print(f"\nmodel={args.model} voxel={cfg.voxel_size*1000:.0f}mm  "
          f"(fitness/RMSE: 計測点のうちモデルから {cfg.eval_threshold*1000:.0f}mm 以内の割合とその RMSE / "
          f"FSV: モデル点が計測深度より10mm以上手前に来る割合)")
    print(f"{'masks':>16} {'#inst':>6} {'#ok':>5} {'fitness':>8} {'RMSE[mm]':>9} {'FSV<5%':>7} {'FSV>20%':>8} {'time[ms]':>9} "
          f"{'matched':>8} {'ADD-S med[mm]':>14} {'ADD-S<5mm':>10}")
    summary = {}
    for name, per_cfg in all_rows.items():
        rows = [r for rs in per_cfg.values() for r in rs]
        ok = [r for r in rows if r["status"] == "ok"]
        adds = np.array([r["add_s_vs_gt"] for r in ok if "add_s_vs_gt" in r])
        s = {"n": len(rows), "ok": len(ok),
             "fitness": float(np.mean([r["fitness"] for r in ok])) if ok else None,
             "rmse": float(np.nanmean([r["rmse"] for r in ok])) if ok else None,
             "time": float(np.mean([r["time"] for r in ok])) if ok else None}
        fsv = np.array([r["fsv"] for r in ok])
        s.update(fsv_ok=float(np.mean(fsv < 0.05)), fsv_bad=float(np.mean(fsv > 0.2)))
        line = (f"{name:>16} {s['n']:>6} {s['ok']:>5} {s['fitness']:>8.3f} {s['rmse']*1000:>9.2f} "
                f"{s['fsv_ok']*100:>6.0f}% {s['fsv_bad']*100:>7.0f}% {s['time']*1000:>9.1f}")
        if len(adds):
            s.update(matched=int(len(adds)), add_s_median=float(np.median(adds)),
                     add_s_lt5mm=float(np.mean(adds < 0.005)))
            line += f" {len(adds):>8} {np.median(adds)*1000:>14.2f} {np.mean(adds < 0.005)*100:>9.0f}%"
        summary[name] = s
        print(line)
        out = POSE_RESULTS_DIR / f"{name}{suffix}"
        with open(out / "poses.json", "w") as f:
            json.dump({"config": vars(cfg), "summary": s, "per_config": per_cfg}, f, indent=1)
    print(f"Saved to {POSE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
