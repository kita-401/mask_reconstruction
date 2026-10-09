"""CAD仮説によるセグメンテーション（render-and-compare）．

1. 種（大きな断片と，既存手法で統合したまとまり）ごとに，T字管の姿勢の仮説を複数作る
2. 各仮説のモデルを深度画像に描画し，観測深度と比べる
     説明画素 : 描画あり かつ |観測 - 描画| < explain_tol
     違反画素 : 描画あり かつ 観測が描画より violation_tol 以上奥（写るはずの所に何も無い）
     （観測が描画より手前の画素は，手前の物体による隠れなので中立）
3. 「まだ説明されていない画素の説明数 - weight × 違反数」が最大の仮説から貪欲に採用する
4. FastSAM の断片を，その断片を最も多く説明する採用仮説に割り当て，仮説ごとにまとめてマスクにする
"""
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .config import HypConfig
from .hypotheses import pose_hypotheses
from .render import render_depth


@dataclass
class Hypothesis:
    T: np.ndarray
    seed: int
    explained: np.ndarray = None      # bool (H, W)
    violation: int = 0
    rendered: int = 0
    depth: np.ndarray = None


@dataclass
class SegResult:
    masks: list
    selected: list = field(default_factory=list)   # 採用した仮説
    n_hypotheses: int = 0
    hypotheses: list = field(default_factory=list)  # 生成した全仮説（パラメータ比較での再利用用）


def _load(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


class CADHypothesisSegmenter:
    def __init__(self, cfg: HypConfig = HypConfig()):
        self.cfg = cfg
        self.render_model = _load(cfg.render_model)
        self.icp_model = _load(cfg.icp_model)
        self.axis_gen = None
        if cfg.use_axis_hyp:
            from .axis_hypotheses import AxisHypothesisGenerator
            self.axis_gen = AxisHypothesisGenerator(_load(cfg.full_model))

    def evaluate(self, h: Hypothesis, Z, cam):
        cfg = self.cfg
        D = render_depth(self.render_model, h.T, cam, Z.shape, cfg.render_splat)
        drawn = D > 0
        valid = drawn & (Z > 0)
        h.depth = D
        h.explained = valid & (np.abs(Z - D) < cfg.explain_tol)
        h.violation = int(np.sum(valid & (Z > D + cfg.violation_tol)))
        h.rendered = int(drawn.sum())
        return h

    def generate(self, seeds, pts, valid_idx, v_u, v_v, cam, Z):
        hyps = []
        for si, sm in enumerate(seeds):
            if sm.sum() < self.cfg.seed_min_area:
                continue
            sp = pts[valid_idx[sm[v_v, v_u]]]
            poses = pose_hypotheses(sp, self.icp_model, self.cfg) if self.cfg.use_svd_hyp else []
            if self.axis_gen is not None:
                poses += self.axis_gen(sp, pts[valid_idx])
            for T in poses:
                hyps.append(self.evaluate(Hypothesis(T=T, seed=si), Z, cam))
        return hyps

    def __call__(self, fragments, seeds, pts, valid_idx, v_u, v_v, cam, hyps=None):
        """fragments: FastSAM の断片マスク / seeds: 種にするマスク（断片やまとまり）
        hyps: 生成済みの仮説（同じ画像で選択のパラメータだけ変えて比較するとき）"""
        cfg = self.cfg
        shape = fragments[0].shape if fragments else seeds[0].shape
        Z = np.zeros(shape)
        Z[v_v, v_u] = pts[valid_idx, 2]

        # 1-2. 仮説の生成と評価
        if hyps is None:
            hyps = self.generate(seeds, pts, valid_idx, v_u, v_v, cam, Z)

        # 3. 貪欲な採用
        covered = np.zeros(shape, dtype=bool)
        selected = []
        cand = [h for h in hyps if h.rendered > 0 and h.violation / h.rendered <= cfg.max_violation_ratio]
        while cand:
            gains = [np.sum(h.explained & ~covered) - cfg.violation_weight * h.violation for h in cand]
            b = int(np.argmax(gains))
            if np.sum(cand[b].explained & ~covered) < cfg.min_explained:
                break
            h = cand.pop(b)
            selected.append(h)
            covered |= h.explained

        # 4. 断片の割り当て
        groups = {i: [] for i in range(len(selected))}
        leftovers = []
        for f in fragments:
            area = f.sum()
            ratios = [np.sum(f & h.explained) / area for h in selected]
            if ratios and max(ratios) >= cfg.assign_min_ratio:
                groups[int(np.argmax(ratios))].append(f)
            elif cfg.keep_unassigned_min_area and area >= cfg.keep_unassigned_min_area:
                leftovers.append(f)
        masks = [np.any(np.stack(g), axis=0) for g in groups.values() if g] + leftovers
        return SegResult(masks=masks, selected=selected, n_hypotheses=len(hyps), hypotheses=hyps)
