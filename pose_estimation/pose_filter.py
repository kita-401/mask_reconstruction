"""静止またはゆっくり動く物体の姿勢を，フレームをまたいで平滑化するカルマンフィルタ．

状態は姿勢そのもの（位置 p，回転 R）で，動きのモデルは「姿勢は変わらない（ゆっくりした変化はプロセスノイズで吸収）」．
回転は SO(3) のまま持ち，誤差を回転ベクトル（3 次元）で表す誤差状態カルマンフィルタ（乗法的 EKF）にする
（クォータニオンの成分をそのまま足し引きすると，符号の反転や 180° 付近で破綻するため）．
位置 3 成分・回転誤差 3 成分の共分散は対角で持つ．

観測の外れ値（180° の取り違えなど）は捨てる：回転の差が gate_deg を超えるか位置の差が gate_m を超える観測は使わない．
ただし reset_after 回続けて外れたら，物体が動いた（または初期値が誤っていた）とみなして観測に合わせ直す．
"""
import numpy as np
from scipy.spatial.transform import Rotation


class PoseKalman:
    def __init__(self, pos_std=0.002, rot_std_deg=6.0, pos_rate=0.005, rot_rate_deg=5.0,
                 gate_deg=30.0, gate_m=0.02, reset_after=3, max_gap=3.0):
        """pos_std / rot_std_deg: 観測ノイズの標準偏差（1 フレームの推定の揺れ）[m] / [deg]
        pos_rate / rot_rate_deg: プロセスノイズ（1 秒あたりに許す姿勢の変化の標準偏差）[m/√s] / [deg/√s]
        gate_deg / gate_m: これより大きく外れた観測は捨てる
        reset_after: この回数続けて外れたら観測に合わせ直す
        max_gap: これ [s] より長く観測がなければ，次の観測で初期化し直す
        """
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

    def update(self, T_meas, t):
        """観測 T_meas（4x4，モデル → カメラ）を入れ，平滑化した姿勢を返す．t は時刻 [s]．"""
        self.last_rejected = False
        if self.T is None or t - self.t > self.max_gap:
            self._reset(T_meas, t)
            return self.T.copy()
        self.P = self.P + self.q * max(t - self.t, 0.0)      # 予測（姿勢は変えず，不確かさだけ増やす）
        self.t = t
        R = self.T[:3, :3]
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
        return self.T.copy()


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
