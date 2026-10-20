"""セグメンテーション結果を姿勢推定用のマスクファイルとして書き出す．

    python scripts/export_masks.py                       # gt / baseline / proposed を全配置
    python scripts/export_masks.py --methods proposed --configs 1 9

出力: results/masks/<method>/config{i}.npy  (bool, (N, H, W))
姿勢推定（pose_estimation）はこのファイルだけを読む．セグメンテーションの設定を追加・変更したら
METHODS に登録してここで書き出せばよい．
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mask_reconstruction import Config, MaskReconstructor, load_sample  # noqa: E402
from mask_reconstruction.config import RESULTS_DIR  # noqa: E402

_FIXED = Config(intrinsics_ref_size=(848, 480))
METHODS = {
    "baseline": _FIXED,
    "proposed": replace(_FIXED, use_cylinder_axis=True, axis_min_area=5000,
                        use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True),
    "proposed_recall": replace(_FIXED, use_cylinder_axis=True,
                               use_cad_constraint=True, cad_size_margin=1.1, cad_remerge=True),
    "graph": replace(_FIXED, merge_method="graph", normal_angle_thresh=20, axis_min_area=5000,
                     use_cad_constraint=True, cad_size_margin=1.2),
    # 現在の提案手法：7/31 の手法（階層クラスタリング）+ 法線・同点解消・CAD寸法・2段階化・円筒軸
    "hier_improved": replace(_FIXED, hier_normal_check=True, normal_angle_thresh=20, hier_tiebreak=True,
                             use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True,
                             two_stage=True, stage1_min_area=3000,
                             use_cylinder_axis=True, axis_min_area=5000),
    # 推奨（2026-10-09）：上記 + 円筒軸 12mm + FastSAM の重なり + 円筒の半径をモデル点群から自動推定
    "hier_improved_v2": replace(_FIXED, hier_normal_check=True, normal_angle_thresh=20, hier_tiebreak=True,
                                use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True,
                                two_stage=True, stage1_min_area=3000,
                                use_cylinder_axis=True, axis_min_area=5000, axis_dist_thresh=0.012,
                                hier_overlap_link=True, overlap_link_ratio=0.5, auto_model_shape=True),
    # hier_improved_v2 の法線を深度画像上の差分で求める高速版（2026-10-16，統合 1197 → 約 670ms，精度は同等以上）
    "hier_improved_v2_fast": replace(_FIXED, hier_normal_check=True, normal_angle_thresh=20, hier_tiebreak=True,
                                     use_cad_constraint=True, cad_size_margin=1.2, cad_remerge=True,
                                     two_stage=True, stage1_min_area=3000,
                                     use_cylinder_axis=True, axis_min_area=5000, axis_dist_thresh=0.012,
                                     hier_overlap_link=True, overlap_link_ratio=0.5, auto_model_shape=True,
                                     fast_normals=True, fast_normal_window=9),
    # 参考：2段階グラフ統合（大きな断片で統合 → 小さな断片を割り当て）
    "graph_2stage": replace(_FIXED, merge_method="graph", normal_angle_thresh=20, axis_min_area=5000,
                            use_cad_constraint=True, cad_size_margin=1.2,
                            two_stage=True, stage1_min_area=3000),
}


def save(masks, path, shape):
    arr = np.array(masks, dtype=bool) if masks else np.zeros((0, *shape), dtype=bool)
    np.save(path, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["gt", "baseline", "proposed"],
                    help="gt または METHODS のキー: " + ", ".join(METHODS))
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "masks")
    args = ap.parse_args()

    recon = None
    for i in args.configs:
        s = load_sample(i, _FIXED)
        for name in args.methods:
            out = args.out_dir / name
            out.mkdir(parents=True, exist_ok=True)
            if name == "gt":
                masks = s.gt_masks
            else:
                if recon is None:
                    recon = MaskReconstructor(_FIXED)
                recon.cfg = METHODS[name]
                # FastSAM の設定が手法ごとに違うことがあるので，断片は手法ごとに作り直す
                masks = recon.reconstruct(s)
            save(masks, out / f"config{i}.npy", s.shape)
            print(f"config {i} {name}: {len(masks)} masks")


if __name__ == "__main__":
    main()
