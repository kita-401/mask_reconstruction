"""指定した配置について GT / YOLO26 / FastSAM / 提案手法 の比較図を保存する．

    python scripts/visualize.py --config 9
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402

from mask_reconstruction import Config, MaskReconstructor, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402
from mask_reconstruction.fragments import run_yolo  # noqa: E402
from mask_reconstruction.visualization import save_comparison  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or RESULTS_DIR / f"comparison_config{args.config}.png"

    cfg = Config()
    recon = MaskReconstructor(cfg)
    s = load_sample(args.config, cfg)

    m_fs = recon.fragments(s)
    results = {
        "YOLO26 (Zero-shot)": run_yolo(YOLO(cfg.yolo_weights), s.rgb, recon.device),
        "FastSAM (Over-seg)": m_fs,
        "Proposed Method": recon.reconstruct(s, fragments=m_fs),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_comparison(s, results, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
