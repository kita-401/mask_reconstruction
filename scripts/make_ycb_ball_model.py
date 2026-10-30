"""YCB の野球ボール（055_baseball，google_16k）のテクスチャ付きメッシュから，球全体の色付きモデル点群を作る．

    curl -O https://ycb-benchmarks.s3.amazonaws.com/data/google/055_baseball_google_16k.tgz  （weights/ycb/ に展開）
    python scripts/make_ycb_ball_model.py
    → pose_estimation/models/ball_ycb.pcd（原点は当てはめた球の中心，単位 m）

YCB Object and Model Set（CC BY 4.0）：B. Calli et al., "The YCB object and model set", 2015．http://www.ycbbenchmarks.com/
メッシュ表面に面積に比例して点を置き，UV 座標からテクスチャの色を読む（Open3D の読み込みではテクスチャが読めないため）．
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "weights/ycb/055_baseball/google_16k"))
ap.add_argument("--out", default=str(ROOT / "pose_estimation/models/ball_ycb.pcd"))
ap.add_argument("--voxel", type=float, default=0.0005)
ap.add_argument("--samples", type=int, default=400000)
args = ap.parse_args()

mesh = o3d.io.read_triangle_mesh(str(Path(args.src) / "textured.obj"), False)
V = np.asarray(mesh.vertices)
F = np.asarray(mesh.triangles)
UV = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)          # 三角形ごとの 3 頂点の UV
tex = cv2.imread(str(Path(args.src) / "texture_map.png"))[:, :, ::-1]   # RGB
H, W = tex.shape[:2]
rng = np.random.default_rng(0)
a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
tri = rng.choice(len(F), args.samples, p=area / area.sum())
r1, r2 = np.sqrt(rng.random(args.samples)), rng.random(args.samples)
w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)              # 一様な重心座標
pts = (w[:, :, None] * np.stack([a[tri], b[tri], c[tri]], 1)).sum(1)
uv = (w[:, :, None] * UV[tri]).sum(1)
col = tex[np.clip(((1 - uv[:, 1]) * H).astype(int), 0, H - 1), np.clip((uv[:, 0] * W).astype(int), 0, W - 1)] / 255.0
A = np.hstack([2 * V, np.ones((len(V), 1))])
x, *_ = np.linalg.lstsq(A, (V ** 2).sum(1), rcond=None)
center = x[:3]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts - center))
pcd.colors = o3d.utility.Vector3dVector(col)
pcd = pcd.voxel_down_sample(args.voxel)
o3d.io.write_point_cloud(args.out, pcd)
p = np.asarray(pcd.points)
print(f"保存: {args.out}（{len(p)} 点，半径 {np.linalg.norm(p, axis=1).mean()*1000:.1f} mm，色の範囲 {np.asarray(pcd.colors).min():.2f}〜{np.asarray(pcd.colors).max():.2f}）")
