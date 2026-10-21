"""提案手法のパラメータ一覧．

値はすべて 3Dto2D/run_batch_evaluation.py（2026-07-31 最終バッチ評価）と同一．
"""
from dataclasses import dataclass
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PKG_ROOT / "data"
WEIGHTS_DIR = PKG_ROOT / "weights"
RESULTS_DIR = PKG_ROOT / "results"
LINGBOT_DIR = PKG_ROOT.parent / "lingbot-depth"


@dataclass
class Config:
    # --- モデル ---
    fastsam_weights: str = str(WEIGHTS_DIR / "FastSAM-s.pt")
    yolo_weights: str = str(WEIGHTS_DIR / "yolo26n-seg.pt")
    lingbot_model: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

    # --- 入力画像 ---
    # LingBot-Depth (ViT, patch 14) に合わせて縦横を14の倍数に切り下げる
    patch_size: int = 14
    # 元実装では内部パラメータを 1280x720 基準でスケーリングしている．
    # 実データは 848x480 なので厳密には誤差があるが，既存結果との再現性のため踏襲．
    intrinsics_ref_size: tuple = (1280, 720)

    # --- FastSAM ---
    fastsam_conf: float = 0.3
    fastsam_iou: float = 0.8
    fastsam_imgsz: int = 640
    min_fragment_area: int = 1000      # [px] これ未満の断片は捨てる

    # --- 深度前処理 ---
    median_ksize: int = 5
    depth_outlier_thresh: float = 20   # [mm] メディアンとの差がこれを超える画素を欠損扱い
    depth_trunc: float = 2.0           # [m]

    # --- 断片フィルタ ---
    black_thresh: int = 60             # BGR全チャネルがこれ未満 → 黒（背景トレイ）
    max_black_ratio: float = 0.5
    min_fragment_points: int = 50

    # --- 断片統合（3Dモデル参照の階層クラスタリング） ---
    L_cad: float = 0.530               # CADモデル由来の形状重み
    adjacency_kernel: int = 11         # [px] 隣接判定の膨張カーネル
    bound_depth_thresh: float = 0.008  # [m] 境界の深度差がこれ未満なら同一部品とみなす
    connected_dist: float = 0.1        # 上記を満たす断片ペアの距離
    w_center: float = 115.0            # 重心間距離の重み
    w_angle: float = 5.0               # 主軸角度差の重み
    w_axis_align: float = 5.0          # 主軸と重心方向のずれの重み
    adjacent_penalty: float = 10.0     # 隣接しているが深度が不連続な場合のペナルティ
    non_adjacent_dist: float = 1000.0  # 距離行列の初期値
    linkage_method: str = "average"
    cluster_thresh: float = 1.5

    # --- 円筒軸による同一部品判定（深度連続に加えて，法線PCAで求めた円筒軸が近いことを要求） ---
    # 点群スケールが正しいこと（intrinsics_ref_size=(848, 480)）が前提
    use_cylinder_axis: bool = False
    pipe_radius: float = 0.0146        # [m] CADモデルの第3主軸寸法（外径 29.2mm）の半分（auto_model_shape=True なら自動）
    # モデル点群（cad_model）から形状特徴を自動で求める（model_shape.py）．部品専用の値を手で設定しない
    #   - 円筒の半径を推定して pipe_radius に使う
    #   - 円筒でできている割合が cylinder_min_fraction 未満の部品では，円筒軸の判定を使わない
    auto_model_shape: bool = False
    cylinder_min_fraction: float = 0.2
    normal_radius: float = 0.006       # [m] 法線推定の近傍半径
    axis_dist_thresh: float = 0.008    # [m] 円筒軸の線分間距離がこれ未満なら同一部品
    axis_seg_ext: float = 0.010        # [m] 軸の線分を両端に延長する長さ
    axis_max_points: int = 4000
    axis_min_area: int = 0             # [px] どちらかの断片がこれ未満なら軸の判定を省く

    # --- CAD寸法制約（統合後の大きさがCADモデルを超える統合を禁止） ---
    # 有効にする場合は intrinsics_ref_size=(848, 480) にして点群のスケールを正しくすること
    use_cad_constraint: bool = False
    cad_model: str = str(DATA_DIR / "template_pipe.pcd")
    cad_size_margin: float = 1.3       # CADの主軸方向寸法の何倍まで許すか
    extent_percentile: float = 2.0     # 寸法は [p, 100-p] パーセンタイル幅（飛び点対策）
    max_points_per_fragment: int = 2000
    cad_remerge: bool = False          # 分割後の切れ端をCAD寸法内で再統合する

    # --- 7/31 の手法（階層クラスタリング）への追加の判定 ---
    hier_normal_check: bool = False     # 「同一部品」の判定に境界の法線のつながり（normal_angle_thresh）を加える
    hier_tiebreak: bool = False         # 「同一部品」の距離に連続値を足して同点を解消する
    hier_overlap_link: bool = False     # FastSAM の断片同士が大きく重なっていれば結合する
    overlap_link_ratio: float = 0.3     # 重なり / 小さい方の面積 がこれ以上
    tiebreak_scale: float = 0.1
    # 「モデル1個で説明できるか」の判定（model_fit.py）．モデル点群のみ使用し，形状パラメータは不要
    hier_model_check: bool = False
    model_check_model: str = str(DATA_DIR / "template_pipe.pcd")
    model_check_min_area: int = 3000    # [px] 両方の断片がこれ以上のときだけ判定する
    model_check_min_ratio: float = 0.2  # 両断片の説明率の小さい方の下限
    model_check_voxel: float = 0.002    # [m]
    model_check_tol: float = 0.003      # [m]
    model_check_ransac_iter: int = 100000         # 加える値の大きさ（connected_dist + scale × (深度差/閾値 + 角度差/閾値)）

    # --- 深度 ---
    depth_refine: bool = True           # False: LingBot-Depth を使わず生の深度（飛び値除去のみ）を使う

    # --- 統合方式 ---
    # "hierarchical": 距離行列 + 階層クラスタリング（元手法，上の各拡張はこちら用）
    # "graph"       : 辺のコスト順に統合し，統合ごとに検証する（graph_merging.py）
    #                 円筒軸・CAD寸法のパラメータは上と共通（use_cylinder_axis は不要）
    merge_method: str = "hierarchical"
    use_normal_check: bool = True       # 境界両側の法線のつながりも見る
    normal_angle_thresh: float = 15.0   # [deg]
    boundary_normal_radius: float = 0.004  # [m] 法線推定の近傍半径
    graph_use_axis: bool = True         # 大きい断片同士では円筒軸の近さも要求する
    use_bridge: bool = False            # 隠れをはさんで分断された断片もつなぐ（現状は効果なし）
    bridge_max_gap: float = 0.045       # [m] 断片間の最近点距離の上限（交差するパイプの直径 + 余裕）
    bridge_samples: int = 15
    bridge_occlusion_tol: float = 0.005  # [m] 観測深度がこれ以上手前なら「隠れている」
    bridge_min_occluded: float = 0.6    # 間の領域のうち隠れている割合の下限
    bridge_penalty: float = 1.0         # 隣接辺を優先するためのコスト加算
    # 2段階統合：1段目は大きな断片だけでグラフ統合，2段目で小さな断片をまとまりに割り当てる
    two_stage: bool = False
    stage1_min_area: int = 3000         # [px] これ以上を「大きな断片」として1段目に使う
    stage2_depth_thresh: float = 0.008  # [m]
    stage2_normal_thresh: float = 20.0  # [deg]
    stage2_keep_unassigned: bool = False  # 割り当てられなかった小断片を残すか（False: 捨てる）
    use_template_check: bool = False    # 統合時にテンプレートの当てはまりも確認する
    template_model: str = str(DATA_DIR / "template_pipe_half.pcd")
    template_voxel: float = 0.002
    template_icp_threshold: float = 0.01
    template_icp_iter: int = 30
    template_fit_threshold: float = 0.003
    template_min_points: int = 300      # まとまりの点数がこれ未満なら当てはまりは確認しない
    template_min_fitness: float = 0.5
    template_max_drop: float = 0.1      # 統合後の fitness が統合前の最大値からこれ以上下がったら統合しない

    # --- 切れ端の吸収（統合後の後処理．どの統合方式とも組み合わせ可） ---
    absorb_small: bool = False
    absorb_area: int = 5000            # [px] これ未満のまとまりを吸収の対象にする
    absorb_depth_thresh: float = 0.008  # [m] 吸収先との境界の深度差の上限

    # ===== 2026-10-16 追加の改善（既定はすべてオフ．オフなら上の挙動と同一） =====
    # --- 改善1：LingBot-Depth を使わない深度（depth_refine=False のときだけ有効） ---
    # True : 時間方向メディアン（複数フレームがあれば）または有効画素だけの空間メディアン．欠損は補完しない
    # False: 既存の remove_depth_outliers（飛び値除去のみ）
    depth_temporal: bool = False
    depth_temporal_frames: int = 5      # 取得・使用するフレーム数 N（N=1 なら空間メディアンのみ）
    depth_temporal_min_valid: float = 0.5  # 有効フレームがこの割合未満の画素は欠損
    depth_spatial_ksize: int = 5        # N=1 のときの空間メディアンの窓 [px]（有効画素だけで計算）

    # --- 改善2：CADサイズによる cannot-link（制約付き平均連結クラスタリング，constrained_merging.py） ---
    use_size_constraint: bool = False
    size_margin: float = 0.15           # m．(1+m)×CAD値を超える統合を禁止
    size_percentile: float = 2.0        # 寸法・最大径は [p, 100-p] パーセンタイルの範囲で計算
    size_erode_px: int = 0              # 寸法・最大径の点群を取る前にマスクを収縮する [px]（縁の他物体の点対策．面積には使わない）
    cad_proj_rotations: int = 500       # A_cad を求めるランダム回転の数
    cad_proj_seed: int = 0
    # 自前の平均連結を使う（制約オフでも True にすれば自前実装．テストで scipy と比較する）
    own_linkage: bool = False

    # --- 改善3：法線の凸性（LCCP型）による境界判定 ---
    use_convexity: bool = False
    no_force_connect: bool = False      # 「段差 < bound_depth_thresh → connected_dist に上書き」をやめる
    normal_window: int = 5              # [px] 深度画像上の法線の差分窓
    convex_samples: int = 30            # 境界上のサンプル数 K
    convex_inset: int = 3               # [px] 境界から内側へずらす距離
    flat_angle: float = 15.0            # [deg] 法線のなす角がこれ未満なら平坦（凸扱い）
    concave_ratio_thresh: float = 0.5   # τ_r
    concave_angle_thresh: float = -1.0  # τ_a [deg]．負なら CAD の最大凹角 θ_cad_max + concave_angle_margin
    concave_angle_margin: float = 10.0  # [deg]
    # β．凸・平坦な境界の距離から引く量．scripts/analyze_pairs.py の leave-one-scene-out で
    # 全10折とも 5.0 が選ばれた（基本コストの分布：同一部品の中央値 ≈5，別部品 ≈7.5，cluster_thresh 1.5）
    convex_bonus: float = 5.0
    concave_penalty: float = 10.0       # 凹と判定した境界に足す量（既存の adjacent_penalty と同じ）
    cad_concave_pair_dist: tuple = (0.004, 0.010)  # [m] CAD上で凹凸を調べる点対の距離（境界をはさむ2点の間隔）
    cad_concave_percentile: float = 99.0  # θ_cad_max として使う凹角分布のパーセンタイル

    # --- 改善5：統合マスクの後処理（深度が連続な範囲だけで closing と穴埋め） ---
    post_close: bool = False
    post_close_ksize: int = 7           # [px]
    post_depth_thresh: float = 0.005    # [m] 局所平均深度との差がこれ未満の画素だけ加える
    cad_cache_dir: str = str(RESULTS_DIR / "cad_prior")

    # --- 高速化：統合で使う法線（境界の法線・円筒軸）を深度画像上の差分の外積で求める ---
    # False なら従来どおり Open3D の近傍探索（全点群 boundary_normal_radius，円筒軸は断片ごとに normal_radius）
    fast_normals: bool = False
    # [px] 差分をとる窓（convexity.organized_normals）．hier_improved_v2 の10配置で 5/7/9 を比べ，9 が従来と同等以上
    fast_normal_window: int = 9

    # --- 高速化：重い基本処理を C++（cpp/mr_kernels.cpp）で計算する．結果は Python 版と同じ ---
    # 断片の点の取り出し，断片の組ごとの隣接・深度差・法線角・重なり，深度画像上の法線，断片フィルタの集計
    use_cpp: bool = False

    # --- 統合後の仕上げ：1つの部品が2つ以上のマスクに分かれたものをまとめる（final_merge.py） ---
    final_merge: bool = False
    final_merge_margin: float = 0.2     # まとめた後の大きさが CAD の (1+この値) 倍以内なら許す

    # --- 統合前の断片の切り分け：2つの部品にまたがった断片を，深度・法線の途切れで分ける（split_fragments.py） ---
    split_fragments: bool = False
    # 既定値は LingBot-Depth なし（生の深度）の10配置で 4mm/20°/300px，8mm/20°/300px，8mm/30°/1000px，
    # 12mm/30°/1000px，8mm/45°/1000px を比べて選んだ（厳しいと生の深度の雑音で部品の中まで切れる）
    split_depth_thresh: float = 0.008   # [m] 隣り合う画素の深度差がこれ以上なら途切れ
    split_normal_thresh: float = 30.0   # [deg] 隣り合う画素の法線の角度差がこれ以上なら途切れ
    split_min_area: int = 1000          # [px] これ未満の切れ端は捨てる

    # --- 平面除去による背景断片の除去（黒いトレイ以外の背景用．断片フィルタで使う） ---
    # 点群で最も大きな平面（机・トレイの面）を RANSAC で求め，点の plane_max_ratio 超が平面上にある断片を捨てる
    use_plane_removal: bool = False
    plane_dist_thresh: float = 0.005    # [m] 平面からこの距離以内の点を「平面上」とする
    plane_max_ratio: float = 0.5
    plane_voxel: float = 0.005          # [m] 平面推定の前に点群を間引く
    plane_ransac_iter: int = 1000
