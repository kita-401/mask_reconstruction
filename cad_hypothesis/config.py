"""CAD仮説によるセグメンテーション（render-and-compare）のパラメータ．

既存の mask_reconstruction / pose_estimation とは独立した実験用パッケージ．
"""
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


@dataclass
class HypConfig:
    # --- モデル ---
    render_model: str = str(PKG_DIR / "models" / "t_pipe_dense.pcd")  # 描画用（全周，0.5mm 間隔）
    icp_model: str = str(PKG_DIR / "models" / "t_pipe_half.pcd")      # 位置合わせ用（カメラ側半分）
    render_splat: int = 1             # [px] 描画時に各点を (2r+1)^2 画素に広げる

    # --- 仮説の生成 ---
    seed_min_area: int = 3000         # [px] これ以上の断片・まとまりを仮説の種にする
    voxel: float = 0.002              # [m] ICP 用ダウンサンプリング
    icp_threshold: float = 0.01       # [m]
    icp_iter: int = 30
    use_axis_perms: bool = True       # 主軸の対応の並べ替え（隠れ対策）も初期値に含める
    top_k_per_seed: int = 3           # 種ごとに残す仮説数（ICP fitness 上位）
    use_svd_hyp: bool = True          # SVD 主軸による仮説（hypotheses.py）
    use_axis_hyp: bool = False        # 円筒軸による仮説（axis_hypotheses.py）
    full_model: str = str(PKG_DIR / "models" / "t_pipe_full.pcd")

    # --- 仮説の評価（描画した深度と観測深度の比較） ---
    explain_tol: float = 0.004        # [m] |観測 - 描画| がこれ未満なら「説明できた」画素
    violation_tol: float = 0.010      # [m] 観測が描画より これ以上奥なら「写るはずの所に無い」違反画素
    violation_weight: float = 2.0     # スコア = 説明画素数 - weight × 違反画素数
    min_explained: int = 2500         # [px] 採用する仮説の最小説明画素数（未説明画素のみで数える）
    max_violation_ratio: float = 0.15  # 違反画素 / 描画画素 の上限

    # --- 断片の割り当て ---
    assign_min_ratio: float = 0.5     # 断片のこの割合以上が仮説で説明されていれば割り当てる
    keep_unassigned_min_area: int = 0  # 割り当てられなかった断片のうち，これ以上の面積は単独で残す（0: 残さない）
