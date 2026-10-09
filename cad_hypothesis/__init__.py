"""CAD仮説によるセグメンテーション（render-and-compare）の実験用パッケージ．

既存の mask_reconstruction / pose_estimation には手を加えず，それらの関数を呼び出して使うだけ．
不要になったらこのフォルダと scripts/eval_cad_hypothesis.py を削除すればよい．
"""
from .config import HypConfig
from .segment import CADHypothesisSegmenter

__all__ = ["HypConfig", "CADHypothesisSegmenter"]
