"""推定姿勢のモデル点群を画像に投影して可視化する．"""
import cv2
import numpy as np

from .evaluation import transform


def _color(i, n):
    hsv = np.uint8([[[int(180 * i / max(n, 1)), 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def project(points, K):
    z = points[:, 2]
    u = K[0, 0] * points[:, 0] / z + K[0, 2]
    v = K[1, 1] * points[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def draw_poses(frame, model_pts, poses, masks=None, axis_len=0.03):
    """poses: list of (4,4) or None．マスクの輪郭・投影モデル点・座標軸を描く．"""
    img = (frame.rgb * 0.55).astype(np.uint8)
    n = len(poses)
    for i, T in enumerate(poses):
        col = _color(i, n)
        if masks is not None:
            cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 1)
        if T is None:
            continue
        uv, z = project(transform(model_pts, T), frame.K)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        for u, v in uv[ok].astype(int):
            cv2.circle(img, (u, v), 1, col, -1)
        # 座標軸（X: 赤, Y: 緑, Z: 青）
        pts = np.vstack([T[:3, 3]] + [T[:3, 3] + T[:3, k] * axis_len for k in range(3)])
        uv_ax, _ = project(pts, frame.K)
        o = tuple(uv_ax[0].astype(int))
        for e, c in zip(uv_ax[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, o, tuple(e.astype(int)), c, 2)
    return img
