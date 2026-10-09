"""姿勢推定のパラメータ．

セグメンテーション側（mask_reconstruction パッケージ）には依存しない．
入力はマスクファイル（.npy）と RGB-D データのみ．
"""
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MASKS_DIR = PROJECT_ROOT / "results" / "masks"
POSE_RESULTS_DIR = PROJECT_ROOT / "results" / "pose"


@dataclass
class PoseConfig:
    # --- モデル点群 ---
    # t_pipe_half.pcd: カメラ側の半分（過去の ROS パッケージで使用していたもの）
    # t_pipe_full.pcd: 全周
    model_path: str = str(PKG_DIR / "models" / "t_pipe_half.pcd")

    # --- 計測点群の作成 ---
    depth_scale: float = 1000.0         # depth [uint16] / depth_scale = [m]
    depth_min: float = 0.1              # [m]
    depth_max: float = 2.0              # [m]
    mask_erode_px: int = 3              # マスク境界の飛び点を避けるため収縮させる
    min_points: int = 100               # これ未満のインスタンスは推定しない
    outlier_nb: int = 20                # 統計的外れ値除去（0 で無効）
    outlier_std: float = 2.0

    # --- ダウンサンプリング（モデル・計測とも） ---
    voxel_size: float = 0.002           # [m]

    # --- SVD + ICP（pose_estimation_demo/scripts/Registration_SVD.py と同じ既定値） ---
    flip_check_threshold: float = 0.005  # [m] 軸反転パターン選択時の fitness 判定距離
    icp_threshold: float = 0.01          # [m] ICP の対応点距離
    icp_max_iteration: int = 30
    # 初期姿勢候補の作り方（既定は元実装と同じ）
    hypotheses: str = "original"         # "original": 符号反転のみ / "all_perms": 軸の並べ替えも試す
    icp_all_candidates: bool = False     # True: 全候補で ICP し ICP 後 fitness 最大を採用

    # --- 評価 ---
    eval_threshold: float = 0.003        # [m] fitness / RMSE の判定距離
