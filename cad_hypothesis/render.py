"""点群モデルの深度描画（z-buffer + 点の広げ描き）．"""
import numpy as np


def render_depth(model_pts, T, cam, shape, splat=1):
    """姿勢 T (4x4, モデル→カメラ) のモデルを深度画像に描画する．描画されない画素は 0．"""
    fx, fy, cx, cy = cam
    h, w = shape
    p = model_pts @ T[:3, :3].T + T[:3, 3]
    z = p[:, 2]
    ok = z > 1e-3
    p, z = p[ok], z[ok]
    u = np.round(fx * p[:, 0] / z + cx).astype(np.int64)
    v = np.round(fy * p[:, 1] / z + cy).astype(np.int64)
    D = np.full(h * w, np.inf)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(D, vv[m] * w + uu[m], z[m])
    D[np.isinf(D)] = 0.0
    return D.reshape(h, w)
