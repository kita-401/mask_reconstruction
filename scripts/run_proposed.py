"""提案手法だけを実行し，配置ごとのマスク重畳画像と評価値を保存する．

    python scripts/run_proposed.py                # 配置 1〜10
    python scripts/run_proposed.py --configs 3 9
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, evaluate_masks, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.visualization import mask_overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "proposed")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recon = MaskReconstructor(Config())
    print(f"{'config':>6} {'frag':>5} {'inst':>5} {'GT':>4} {'time[s]':>8} "
          f"{'mIoU%':>6} {'Prec%':>6} {'Rec%':>6} {'over':>5} {'under':>5}")
    for i in args.configs:
        s = load_sample(i)
        t0 = time.time()
        frags = recon.fragments(s)
        masks = recon.reconstruct(s, fragments=frags)
        t = time.time() - t0

        ev = evaluate_masks(masks, s.gt_masks) if s.gt_masks else None
        cv2.imwrite(str(args.out_dir / f"config{i}.png"), mask_overlay(s.rgb, masks))
        np.save(args.out_dir / f"config{i}_masks.npy", np.array(masks))

        line = f"{i:>6} {len(frags):>5} {len(masks):>5} {len(s.gt_masks):>4} {t:>8.3f}"
        if ev:
            line += (f" {ev['miou']*100:>6.1f} {ev['prec']*100:>6.1f} {ev['rec']*100:>6.1f}"
                     f" {ev['over']:>5} {ev['under']:>5}")
        print(line)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
