"""YOLO26 / FastSAM / 提案手法 を全配置で評価し，平均±標準偏差を出力する．

    python scripts/run_batch_evaluation.py               # 配置 1〜10
    python scripts/run_batch_evaluation.py --configs 3 9
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402

from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.evaluation import METRICS  # noqa: E402
from mask_reconstruction.fragments import run_yolo  # noqa: E402

METHODS = ("yolo", "fastsam", "proposed")


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "batch_evaluation.json")
    args = ap.parse_args()

    cfg = Config()
    print("Loading models...")
    recon = MaskReconstructor(cfg)
    yolo = YOLO(cfg.yolo_weights)

    per_config = []
    for i in args.configs:
        print(f"--- Processing config {i} ---")
        s = load_sample(i, cfg)

        m_yolo, t_yolo = timed(run_yolo, yolo, s.rgb, recon.device)
        m_fs, t_fs = timed(recon.fragments, s)
        # 提案手法の時間は FastSAM 推論を含まない（元実装と同じ定義）．合計は fastsam + proposed．
        m_prop, t_prop = timed(recon.reconstruct, s, fragments=m_fs)

        row = {"config": i}
        for name, masks, t in (("yolo", m_yolo, t_yolo), ("fastsam", m_fs, t_fs),
                               ("proposed", m_prop, t_prop)):
            row[name] = {"time": t, "n_masks": len(masks), **evaluate_masks(masks, s.gt_masks)}
        per_config.append(row)
        print("  " + "  ".join(f"{m}: P={row[m]['prec']*100:.1f}% IoU={row[m]['miou']*100:.1f}%"
                               for m in METHODS))

    print("\n" + "=" * 90)
    print(f"BATCH EXPERIMENT RESULTS (Average of {len(per_config)} Configs)")
    print("=" * 90)
    print(f"{'Method':>9} {'Time (s)':>15} {'mIoU (%)':>14} {'Precision (%)':>14} "
          f"{'Recall (%)':>14} {'Over-seg':>10} {'Under-seg':>10}")
    summary = {}
    for name in METHODS:
        vals = {k: np.array([r[name][k] for r in per_config]) for k in ("time",) + METRICS}
        summary[name] = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in vals.items()}
        pct = lambda k: f"{vals[k].mean()*100:6.2f} ± {vals[k].std()*100:5.2f}"  # noqa: E731
        print(f"{name:>9} {vals['time'].mean():6.3f} ± {vals['time'].std():5.3f} "
              f"{pct('miou')} {pct('prec')} {pct('rec')} "
              f"{vals['over'].mean():4.1f} ± {vals['over'].std():3.1f} "
              f"{vals['under'].mean():4.1f} ± {vals['under'].std():3.1f}")

    best = max(per_config, key=lambda r: r["proposed"]["prec"])
    print(f"\nBest config for proposed (Precision): {best['config']} "
          f"({best['proposed']['prec']*100:.2f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_config": per_config}, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
