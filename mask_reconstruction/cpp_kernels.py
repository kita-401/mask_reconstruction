"""C++ の基本処理（cpp/mr_kernels.cpp）を ctypes で呼ぶ（Config.use_cpp=True のとき使う）．

ビルド: bash mask_reconstruction/cpp/build.sh（未ビルドなら初回の呼び出しで自動でビルドする）
どの関数も Python 版と同じ値を返す（tests/test_cpp_kernels.py で確認）．
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent / "cpp"
_LIB_PATH = _DIR / "build" / "libmrkernels.so"
_lib = None

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
_u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_int = ctypes.c_int


def lib():
    global _lib
    if _lib is None:
        src = _DIR / "mr_kernels.cpp"
        if not _LIB_PATH.exists() or _LIB_PATH.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["bash", str(_DIR / "build.sh")], check=True, capture_output=True)
        L = ctypes.CDLL(str(_LIB_PATH))
        L.mr_point_image.argtypes = [_f64, _i64, _i64, ctypes.c_int64, _int, _int, _f64, _u8]
        L.mr_organized_normals.argtypes = [_f64, _u8, _int, _int, _int, _f64, _u8]
        L.mr_gather_indices.argtypes = [_u8, _int, _int, _int, _i64, _i64, ctypes.c_void_p, _int]
        L.mr_mask_counts.argtypes = [_u8, _int, _int, _int, _u8, _int, _i64]
        L.mr_pair_features.argtypes = [_u8, _int, _int, _int, _int, _f64, ctypes.c_void_p,
                                       _u8, _f64, _f64, _i64, _i64]
        L.mr_principal_extent.argtypes = [_f64, ctypes.c_int64, ctypes.c_double, _f64]
        L.mr_fragment_pca.argtypes = [_f64, _i64, _int, _f64, _f64]
        L.mr_fragment_pca.restype = None
        L.mr_pair_features_ab.argtypes = [_u8, _int, _u8, _int, _int, _int, _int, _f64, ctypes.c_void_p,
                                          _u8, _f64, _f64]
        for f in (L.mr_point_image, L.mr_organized_normals, L.mr_gather_indices, L.mr_mask_counts,
                  L.mr_pair_features, L.mr_principal_extent, L.mr_pair_features_ab):
            f.restype = None
        _lib = L
    return _lib


def stack_masks(masks):
    """bool マスクのリスト → (n, H, W) uint8 の連続配列．"""
    return np.ascontiguousarray(np.stack(masks)).view(np.uint8)


def point_image(pts, valid_idx, v_u, v_v, shape):
    H, W = shape
    P = np.empty((H, W, 3))
    V = np.empty((H, W), np.uint8)
    lib().mr_point_image(np.ascontiguousarray(pts[valid_idx], dtype=np.float64),
                         np.ascontiguousarray(v_v, dtype=np.int64), np.ascontiguousarray(v_u, dtype=np.int64),
                         len(valid_idx), H, W, P, V)
    return P, V.view(bool)


def organized_normals(P, V, window):
    H, W = V.shape
    N = np.empty((H, W, 3))
    ok = np.empty((H, W), np.uint8)
    lib().mr_organized_normals(np.ascontiguousarray(P), np.ascontiguousarray(V).view(np.uint8), H, W,
                               window, N, ok)
    return N, ok.view(bool)


def index_image(valid_idx, v_u, v_v, shape):
    """各画素の点の番号（valid_idx の何番目か）．点のない画素は -1．"""
    I = np.full(shape, -1, np.int64)
    I[v_v, v_u] = np.arange(len(valid_idx))
    return I


def gather_indices(stack, index_img):
    """各マスクに含まれる点の番号（valid_idx の何番目か）のリスト．m[v_v, v_u] が真の番号を昇順に並べたもの．"""
    n, H, W = stack.shape
    offsets = np.zeros(n + 1, np.int64)
    lib().mr_gather_indices(stack, n, H, W, index_img, offsets, None, 1)
    out = np.empty(offsets[-1], np.int64)
    lib().mr_gather_indices(stack, n, H, W, index_img, offsets, out.ctypes.data_as(ctypes.c_void_p), 0)
    return [out[offsets[k]:offsets[k + 1]] for k in range(n)]


def mask_counts(stack, images):
    """out[k, j] = sum(mask_k & images[j])．"""
    n, H, W = stack.shape
    imgs = np.ascontiguousarray(np.stack(images)).view(np.uint8)
    out = np.empty((n, len(images)), np.int64)
    lib().mr_mask_counts(stack, n, H, W, imgs, len(images), out)
    return out


def pair_features(stack, ksize, Z, N=None):
    """断片の組ごとの (adjacent, gap, nangle, overlap) と各断片の面積．計算できない値は NaN．"""
    n, H, W = stack.shape
    adjacent = np.zeros((n, n), np.uint8)
    gap = np.full((n, n), np.nan)
    nangle = np.full((n, n), np.nan)
    overlap = np.zeros((n, n), np.int64)
    area = np.zeros(n, np.int64)
    Np = None
    if N is not None:
        N = np.ascontiguousarray(N, dtype=np.float64)
        Np = N.ctypes.data_as(ctypes.c_void_p)
    lib().mr_pair_features(stack, n, H, W, ksize, np.ascontiguousarray(Z, dtype=np.float64), Np,
                           adjacent, gap, nangle, overlap, area)
    return adjacent.view(bool), gap, nangle, overlap, area


def principal_extent(p, percentile):
    """merging.principal_extent と同じ（主成分の方向ごとの [p, 100-p] パーセンタイル幅，分散の大きい順）．"""
    out = np.empty(3)
    lib().mr_principal_extent(np.ascontiguousarray(p, dtype=np.float64), len(p), float(percentile), out)
    return out


def pair_features_ab(stack_a, stack_b, ksize, Z, N=None):
    """集合 A の各マスクと集合 B の各マスクの組（nA×nB）の (adjacent, gap, nangle)．"""
    na, H, W = stack_a.shape
    nb = stack_b.shape[0]
    adjacent = np.zeros((na, nb), np.uint8)
    gap = np.full((na, nb), np.nan)
    nangle = np.full((na, nb), np.nan)
    Np = None
    if N is not None:
        N = np.ascontiguousarray(N, dtype=np.float64)
        Np = N.ctypes.data_as(ctypes.c_void_p)
    lib().mr_pair_features_ab(stack_a, na, stack_b, nb, H, W, ksize, np.ascontiguousarray(Z, dtype=np.float64), Np,
                              adjacent, gap, nangle)
    return adjacent.view(bool), gap, nangle


def fragment_pca(frag_points):
    """断片ごとの点群の重心と第1主軸（merging.fragment_pca と同じ．主軸の符号は不定）．"""
    n = len(frag_points)
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in frag_points])]).astype(np.int64)
    centers, axes = np.empty((n, 3)), np.empty((n, 3))
    lib().mr_fragment_pca(np.ascontiguousarray(np.concatenate(frag_points), dtype=np.float64), offsets, n, centers, axes)
    return list(centers), list(axes)
