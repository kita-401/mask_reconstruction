"""静止またはゆっくり動く物体の姿勢を，フレームをまたいで平滑化するカルマンフィルタ．

状態は姿勢そのもの（位置 p，回転 R）で，動きのモデルは「姿勢は変わらない（ゆっくりした変化はプロセスノイズで吸収）」．
回転は SO(3) のまま持ち，誤差を回転ベクトル（3 次元）で表す誤差状態カルマンフィルタ（乗法的 EKF）にする
（クォータニオンの成分をそのまま足し引きすると，符号の反転や 180° 付近で破綻するため）．
位置 3 成分・回転誤差 3 成分の共分散は対角で持つ．

対称な物体（symmetries）では，観測を入れる前に，対称で同じ形になる姿勢のうち今の推定に最も近いものに置き換える
（T字パイプの幹の軸まわりの 180° など．形からは区別できない姿勢の間の飛びを外れ値として捨てないため）．

観測の外れ値（180° の取り違えなど）は捨てる：回転の差が gate_deg を超えるか位置の差が gate_m を超える観測は使わない．
ただし reset_after 回続けて外れたら，物体が動いた（または初期値が誤っていた）とみなして観測に合わせ直す．

canonical_ref（カメラ座標の 3x3）を与えると，対称で同じ形になる姿勢のうちどれを出すかを基準の向きに揃える：
推定を基準に最も近い姿勢で始め，以後は同じ姿勢を使い続け，別の対称な姿勢の方が基準に switch_deg 以上近くなったときだけ切り替える
（基準から同じくらい離れた姿勢の間で出力が飛ばないように）．
"""
import numpy as np
from scipy.spatial.transform import Rotation


class PoseKalman:
    def __init__(self, pos_std=0.002, rot_std_deg=6.0, pos_rate=0.005, rot_rate_deg=5.0,
                 gate_deg=30.0, gate_m=0.02, reset_after=3, max_gap=3.0, symmetries=(),
                 canonical_ref=None, switch_deg=20.0):
        """pos_std / rot_std_deg: 観測ノイズの標準偏差（1 フレームの推定の揺れ）[m] / [deg]
        pos_rate / rot_rate_deg: プロセスノイズ（1 秒あたりに許す姿勢の変化の標準偏差）[m/√s] / [deg/√s]
        gate_deg / gate_m: これより大きく外れた観測は捨てる
        reset_after: この回数続けて外れたら観測に合わせ直す
        max_gap: これ [s] より長く観測がなければ，次の観測で初期化し直す
        symmetries: 物体の対称性（モデル座標での 4x4 変換 S のリスト．T と T @ S が同じ形になる）
        canonical_ref / switch_deg: 対称な姿勢のうちどれを出すかの基準の向き（カメラ座標の 3x3）と，切り替えの余裕 [deg]
        """
        self.canonical_ref = None if canonical_ref is None else np.asarray(canonical_ref, float)
        self.switch_rad = np.deg2rad(switch_deg)
        self.symmetries = [np.asarray(S, float) for S in symmetries]
        self.r = np.array([pos_std ** 2] * 3 + [np.deg2rad(rot_std_deg) ** 2] * 3)
        self.q = np.array([pos_rate ** 2] * 3 + [np.deg2rad(rot_rate_deg) ** 2] * 3)
        self.gate_rad, self.gate_m = np.deg2rad(gate_deg), gate_m
        self.reset_after, self.max_gap = reset_after, max_gap
        self.T = None
        self.P = None
        self.t = None
        self.n_reject = 0
        self.last_rejected = False

    def _reset(self, T, t):
        self.T, self.P, self.t, self.n_reject = T.copy(), self.r.copy(), t, 0
        self._canonicalize(margin=0.0)

    def _ref_dist(self, T):
        return Rotation.from_matrix(T[:3, :3] @ self.canonical_ref.T).magnitude()

    def _canonicalize(self, margin):
        """推定を，対称な姿勢のうち基準の向きに最も近いものに置き換える（今の姿勢より margin [rad] 以上近いときだけ）．"""
        if self.canonical_ref is None or not self.symmetries:
            return
        best = min((self.T @ S for S in self.symmetries), key=self._ref_dist)
        if self._ref_dist(best) < self._ref_dist(self.T) - margin:
            self.T = best

    def update(self, T_meas, t):
        """観測 T_meas（4x4，モデル → カメラ）を入れ，平滑化した姿勢を返す．t は時刻 [s]．"""
        self.last_rejected = False
        if self.T is None or t - self.t > self.max_gap:
            self._reset(T_meas, t)
            return self.T.copy()
        self.P = self.P + self.q * max(t - self.t, 0.0)      # 予測（姿勢は変えず，不確かさだけ増やす）
        self.t = t
        R = self.T[:3, :3]
        if self.symmetries:   # 対称で同じ形になる姿勢のうち，今の推定に最も近いものを観測とする
            cands = [T_meas] + [T_meas @ S for S in self.symmetries]
            T_meas = min(cands, key=lambda C: Rotation.from_matrix(R.T @ C[:3, :3]).magnitude())
        d_rot = Rotation.from_matrix(R.T @ T_meas[:3, :3]).as_rotvec()   # モデル座標での回転のずれ
        d_pos = T_meas[:3, 3] - self.T[:3, 3]
        if np.linalg.norm(d_rot) > self.gate_rad or np.linalg.norm(d_pos) > self.gate_m:
            self.n_reject += 1
            self.last_rejected = True
            if self.n_reject >= self.reset_after:            # 続けて外れた → 動いた（または初期値が誤り）とみなす
                self._reset(T_meas, t)
            return self.T.copy()
        self.n_reject = 0
        K = self.P / (self.P + self.r)
        T = self.T.copy()
        T[:3, 3] = self.T[:3, 3] + K[:3] * d_pos
        T[:3, :3] = R @ Rotation.from_rotvec(K[3:] * d_rot).as_matrix()
        self.T = T
        self.P = (1.0 - K) * self.P
        self._canonicalize(margin=self.switch_rad)
        return self.T.copy()


def half_turn_symmetry(model_points, axis_index=1):
    """モデル点群の第 axis_index+1 主軸（重心を通る）まわりの 180° 回転（モデル座標の 4x4）と，
    その回転でモデルが自分に重なるかの目安（対応点までの距離の中央値 [m]）．
    T字パイプは第2主軸（幹の軸）まわりに 180° 回すと同じ形になる（t_pipe_full.pcd で 0.9 mm）．"""
    from scipy.spatial import cKDTree
    c = model_points.mean(0)
    _, _, vt = np.linalg.svd(model_points - c, full_matrices=False)
    R = Rotation.from_rotvec(np.pi * vt[axis_index]).as_matrix()
    S = np.eye(4)
    S[:3, :3] = R
    S[:3, 3] = c - R @ c
    err = float(np.median(cKDTree(model_points).query((model_points - c) @ R.T + c)[0]))
    return S, err


class PoseFilterBank:
    """ID ごとの PoseKalman．しばらく観測のない ID のフィルタは捨てる．"""

    def __init__(self, drop_after=5.0, **kw):
        self.kw, self.drop_after = kw, drop_after
        self.filters = {}

    def update(self, key, T_meas, t):
        f = self.filters.get(key)
        if f is None:
            f = self.filters[key] = PoseKalman(**self.kw)
        T = f.update(T_meas, t)
        for k in [k for k, f in self.filters.items() if t - f.t > self.drop_after]:
            del self.filters[k]
        return T, f.last_rejected
