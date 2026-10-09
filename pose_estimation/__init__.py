"""セグメンテーションマスクからの3次元姿勢推定（SVD + ICP）．

セグメンテーション側（mask_reconstruction パッケージ）には依存しない．
入力はマスクファイル（results/masks/<手法名>/config{i}.npy）と RGB-D データ．
"""
from .config import PoseConfig
from .registration_svd import RegistrationResult, SVDICPRegistration

__all__ = ["PoseConfig", "SVDICPRegistration", "RegistrationResult"]
