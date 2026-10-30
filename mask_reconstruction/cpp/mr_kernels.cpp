// mask_reconstruction の重い基本処理の C++ 実装（Python からは cpp_kernels.py が ctypes で呼ぶ）．
//
// どの関数も，Python（numpy）版と同じ値を返すように書いてある：
//   - 画像全体を走査せず，断片を囲む矩形（膨張分を含む）の中だけを走査する
//   - 浮動小数点の演算順は numpy にそろえる（ビルド時に -ffp-contract=off で積和の融合を止める）
// ビルド: bash mask_reconstruction/cpp/build.sh
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace {

struct Box {
    int y0 = 0, y1 = 0, x0 = 0, x1 = 0;  // [y0, y1) x [x0, x1)
    bool empty() const { return y1 <= y0 || x1 <= x0; }
};

Box intersect(const Box& a, const Box& b) {
    Box r;
    r.y0 = std::max(a.y0, b.y0);
    r.y1 = std::min(a.y1, b.y1);
    r.x0 = std::max(a.x0, b.x0);
    r.x1 = std::min(a.x1, b.x1);
    return r;
}

Box bbox_of(const uint8_t* m, int H, int W) {
    // 行ごとに「1つでも立っているか」を OR でまとめて判定し（ベクトル化される），
    // 立っている行だけ左右の端を探す
    Box b;
    b.y0 = H; b.y1 = 0; b.x0 = W; b.x1 = 0;
    for (int y = 0; y < H; ++y) {
        const uint8_t* row = m + (int64_t)y * W;
        uint8_t acc = 0;
        for (int x = 0; x < W; ++x) acc |= row[x];
        if (!acc) continue;
        b.y0 = std::min(b.y0, y);
        b.y1 = y + 1;
        int x0 = 0;
        while (!row[x0]) ++x0;
        int x1 = W - 1;
        while (!row[x1]) --x1;
        b.x0 = std::min(b.x0, x0);
        b.x1 = std::max(b.x1, x1 + 1);
    }
    return b;
}

// 正方形カーネル（一辺 2r+1）での膨張．cv2.dilate（既定の境界）と同じ：画像外は膨張に寄与しない．
// 結果は矩形 box（元の矩形を r 広げて画像内に切ったもの）の中だけを持つ．
struct Dilated {
    Box box;
    std::vector<uint8_t> data;  // box の大きさ
    uint8_t at(int y, int x) const {
        if (y < box.y0 || y >= box.y1 || x < box.x0 || x >= box.x1) return 0;
        return data[(int64_t)(y - box.y0) * (box.x1 - box.x0) + (x - box.x0)];
    }
};

Dilated dilate(const uint8_t* m, const Box& b, int H, int W, int r) {
    // 横・縦それぞれ，窓（2r+1）内の立っている画素の数を1画素ずつずらしながら数える（1画素あたり一定の手間）
    Dilated d;
    if (b.empty()) { d.box = b; return d; }
    d.box.y0 = std::max(0, b.y0 - r); d.box.y1 = std::min(H, b.y1 + r);
    d.box.x0 = std::max(0, b.x0 - r); d.box.x1 = std::min(W, b.x1 + r);
    const int h = d.box.y1 - d.box.y0, w = d.box.x1 - d.box.x0;
    std::vector<uint8_t> tmp((int64_t)h * w, 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = m + (int64_t)(y + d.box.y0) * W + d.box.x0;  // box の外の画素は元のマスクで 0
        int cnt = 0;
        for (int t = 0; t <= std::min(r, w - 1); ++t) cnt += row[t] ? 1 : 0;
        for (int x = 0; x < w; ++x) {
            tmp[(int64_t)y * w + x] = cnt > 0;
            const int add = x + r + 1, rem = x - r;
            if (add < w && row[add]) ++cnt;
            if (rem >= 0 && row[rem]) --cnt;
        }
    }
    d.data.assign((int64_t)h * w, 0);
    std::vector<int> cnt(w, 0);
    for (int t = 0; t <= std::min(r, h - 1); ++t)
        for (int x = 0; x < w; ++x) cnt[x] += tmp[(int64_t)t * w + x];
    for (int y = 0; y < h; ++y) {
        uint8_t* out = d.data.data() + (int64_t)y * w;
        for (int x = 0; x < w; ++x) out[x] = cnt[x] > 0;
        const int add = y + r + 1, rem = y - r;
        if (add < h) for (int x = 0; x < w; ++x) cnt[x] += tmp[(int64_t)add * w + x];
        if (rem >= 0) for (int x = 0; x < w; ++x) cnt[x] -= tmp[(int64_t)rem * w + x];
    }
    return d;
}

// numpy.median と同じ（偶数個なら中央2つの平均）
double median(std::vector<double>& v) {
    const size_t n = v.size();
    const size_t k = n / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    const double hi = v[k];
    if (n % 2 == 1) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + k);
    return (lo + hi) / 2.0;
}

}  // namespace

extern "C" {

// 点群を画素に戻す：P[v, u] = pts（(m, 3)），V = P[..., 2] > 0
void mr_point_image(const double* pts, const int64_t* v, const int64_t* u, int64_t m,
                    int H, int W, double* P, uint8_t* V) {
    std::fill(P, P + (int64_t)H * W * 3, 0.0);
    for (int64_t j = 0; j < m; ++j) {
        double* p = P + ((int64_t)v[j] * W + u[j]) * 3;
        p[0] = pts[j * 3]; p[1] = pts[j * 3 + 1]; p[2] = pts[j * 3 + 2];
    }
    for (int64_t i = 0; i < (int64_t)H * W; ++i) V[i] = P[i * 3 + 2] > 0 ? 1 : 0;
}

// convexity.organized_normals と同じ：左右・上下に h 画素離れた点の差分の外積をカメラ側へ向ける
void mr_organized_normals(const double* P, const uint8_t* V, int H, int W, int window,
                          double* N, uint8_t* ok) {
    const int h = std::max(window / 2, 1);
    std::fill(N, N + (int64_t)H * W * 3, 0.0);
    std::fill(ok, ok + (int64_t)H * W, 0);
    for (int y = h; y < H - h; ++y) {
        for (int x = h; x < W - h; ++x) {
            const int64_t c = (int64_t)y * W + x;
            const int64_t xr = c + h, xl = c - h, yd = c + (int64_t)h * W, yu = c - (int64_t)h * W;
            const double a0 = P[xr * 3] - P[xl * 3], a1 = P[xr * 3 + 1] - P[xl * 3 + 1],
                         a2 = P[xr * 3 + 2] - P[xl * 3 + 2];
            const double b0 = P[yd * 3] - P[yu * 3], b1 = P[yd * 3 + 1] - P[yu * 3 + 1],
                         b2 = P[yd * 3 + 2] - P[yu * 3 + 2];
            double n0 = a1 * b2 - a2 * b1, n1 = a2 * b0 - a0 * b2, n2 = a0 * b1 - a1 * b0;
            const double norm = std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);
            const bool valid = V[c] && V[xr] && V[xl] && V[yd] && V[yu] && norm > 0;
            const double s = norm > 0 ? norm : 1.0;
            n0 /= s; n1 /= s; n2 /= s;
            if (n0 * P[c * 3] + n1 * P[c * 3 + 1] + n2 * P[c * 3 + 2] > 0) { n0 = -n0; n1 = -n1; n2 = -n2; }
            if (valid) {
                N[c * 3] = n0; N[c * 3 + 1] = n1; N[c * 3 + 2] = n2;
                ok[c] = 1;
            }
        }
    }
}

// 各マスクに含まれる点の番号（index_img[y, x] = 点の番号，点のない画素は -1）を行優先の順に並べる．
// pts[valid_idx[m[v_v, v_u]]] と同じ順序．counts_only=1 なら offsets（n+1 個）だけ埋める．
void mr_gather_indices(const uint8_t* masks, int n, int H, int W, const int64_t* index_img,
                       int64_t* offsets, int64_t* out, int counts_only) {
    const int64_t HW = (int64_t)H * W;
    offsets[0] = 0;
    for (int k = 0; k < n; ++k) {
        const uint8_t* m = masks + k * HW;
        const Box b = bbox_of(m, H, W);
        int64_t cnt = 0;
        for (int y = b.y0; y < b.y1; ++y)
            for (int x = b.x0; x < b.x1; ++x) {
                const int64_t i = (int64_t)y * W + x;
                if (m[i] && index_img[i] >= 0) {
                    if (!counts_only) out[offsets[k] + cnt] = index_img[i];
                    ++cnt;
                }
            }
        offsets[k + 1] = offsets[k] + cnt;
    }
}

// 各マスクと各画像（0/1）の共通部分の画素数：out[k * n_img + j] = sum(mask_k & img_j)
void mr_mask_counts(const uint8_t* masks, int n, int H, int W, const uint8_t* imgs, int n_img,
                    int64_t* out) {
    const int64_t HW = (int64_t)H * W;
    for (int k = 0; k < n; ++k) {
        const uint8_t* m = masks + k * HW;
        const Box b = bbox_of(m, H, W);
        for (int j = 0; j < n_img; ++j) {
            const uint8_t* im = imgs + j * HW;
            int64_t s = 0;
            for (int y = b.y0; y < b.y1; ++y)
                for (int x = b.x0; x < b.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    s += (m[i] && im[i]) ? 1 : 0;
                }
            out[k * n_img + j] = s;
        }
    }
}

// 断片の組ごとの特徴（k < l の組だけ計算し，対称に埋める）
//   adjacent[k,l]  : any(dilate(m_k) & m_l)（一辺 ksize の正方形で膨張）
//   gap[k,l]       : |median(Z[m_k & dil_l], Z>0) − median(Z[m_l & dil_k], Z>0)|（隣接のみ．計算できなければ NaN）
//   nangle[k,l]    : 境界両側の平均法線の角度 [deg]（N が NULL でなく隣接のみ．各側 5 点未満なら NaN）
//   overlap[k,l]   : sum(m_k & m_l)
//   area[k]        : sum(m_k)
void mr_pair_features(const uint8_t* masks, int n, int H, int W, int ksize, const double* Z,
                      const double* N, uint8_t* adjacent, double* gap, double* nangle,
                      int64_t* overlap, int64_t* area) {
    const int64_t HW = (int64_t)H * W;
    const int r = ksize / 2;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Box> boxes(n);
    std::vector<Dilated> dil(n);
    for (int k = 0; k < n; ++k) {
        const uint8_t* m = masks + k * HW;
        boxes[k] = bbox_of(m, H, W);
        dil[k] = dilate(m, boxes[k], H, W, r);
        int64_t a = 0;
        for (int y = boxes[k].y0; y < boxes[k].y1; ++y)
            for (int x = boxes[k].x0; x < boxes[k].x1; ++x) a += m[(int64_t)y * W + x] ? 1 : 0;
        area[k] = a;
    }
    std::vector<double> zk, zl;
    for (int k = 0; k < n; ++k) {
        const uint8_t* mk = masks + k * HW;
        for (int l = k + 1; l < n; ++l) {
            const uint8_t* ml = masks + l * HW;
            const int64_t kl = (int64_t)k * n + l, lk = (int64_t)l * n + k;
            // 重なり
            int64_t ov = 0;
            const Box bo = intersect(boxes[k], boxes[l]);
            for (int y = bo.y0; y < bo.y1; ++y)
                for (int x = bo.x0; x < bo.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    ov += (mk[i] && ml[i]) ? 1 : 0;
                }
            overlap[kl] = overlap[lk] = ov;
            // 隣接：dil_k & m_l
            bool adj = false;
            const Box ba = intersect(dil[k].box, boxes[l]);
            for (int y = ba.y0; y < ba.y1 && !adj; ++y)
                for (int x = ba.x0; x < ba.x1; ++x)
                    if (ml[(int64_t)y * W + x] && dil[k].at(y, x)) { adj = true; break; }
            adjacent[kl] = adjacent[lk] = adj ? 1 : 0;
            gap[kl] = gap[lk] = nan;
            nangle[kl] = nangle[lk] = nan;
            if (!adj) continue;
            // 境界の帯：band_k = m_k & dil_l，band_l = m_l & dil_k
            zk.clear(); zl.clear();
            double sk[3] = {0, 0, 0}, sl[3] = {0, 0, 0};
            int64_t ck = 0, cl = 0;
            const Box bk = intersect(boxes[k], dil[l].box);
            for (int y = bk.y0; y < bk.y1; ++y)
                for (int x = bk.x0; x < bk.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    if (!mk[i] || !dil[l].at(y, x)) continue;
                    if (Z[i] > 0) zk.push_back(Z[i]);
                    if (N) {
                        const double* nv = N + i * 3;
                        if (std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]) > 0) {
                            sk[0] += nv[0]; sk[1] += nv[1]; sk[2] += nv[2]; ++ck;
                        }
                    }
                }
            const Box bl = intersect(boxes[l], dil[k].box);
            for (int y = bl.y0; y < bl.y1; ++y)
                for (int x = bl.x0; x < bl.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    if (!ml[i] || !dil[k].at(y, x)) continue;
                    if (Z[i] > 0) zl.push_back(Z[i]);
                    if (N) {
                        const double* nv = N + i * 3;
                        if (std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]) > 0) {
                            sl[0] += nv[0]; sl[1] += nv[1]; sl[2] += nv[2]; ++cl;
                        }
                    }
                }
            if (!zk.empty() && !zl.empty()) gap[kl] = gap[lk] = std::fabs(median(zk) - median(zl));
            if (N && ck >= 5 && cl >= 5) {
                double mk3[3], ml3[3];
                for (int t = 0; t < 3; ++t) { mk3[t] = sk[t] / ck; ml3[t] = sl[t] / cl; }
                const double nk = std::sqrt(mk3[0] * mk3[0] + mk3[1] * mk3[1] + mk3[2] * mk3[2]);
                const double nl = std::sqrt(ml3[0] * ml3[0] + ml3[1] * ml3[1] + ml3[2] * ml3[2]);
                double dot = 0;
                for (int t = 0; t < 3; ++t) dot += (mk3[t] / nk) * (ml3[t] / nl);
                dot = std::min(1.0, std::max(-1.0, dot));
                nangle[kl] = nangle[lk] = std::acos(dot) * 180.0 / M_PI;
            }
        }
    }
}

// numpy.percentile（既定の linear）と同じ値：idx = q/100·(n−1) の前後2点を補間する．v は並べ替えてよい作業領域
static double percentile_linear(std::vector<double>& v, double q) {
    const size_t n = v.size();
    const double idx = q / 100.0 * (double)(n - 1);
    const size_t lo = (size_t)std::floor(idx);
    const size_t hi = std::min(lo + 1, n - 1);
    const double t = idx - (double)lo;
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    const double a = v[lo];
    double b = a;
    if (hi != lo) b = *std::min_element(v.begin() + lo + 1, v.end());
    const double d = b - a;
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;     // numpy の _lerp と同じ
}

// merging.principal_extent と同じ：主成分の方向（分散の大きい順）ごとの [q, 100−q] パーセンタイル幅．
// 主成分は numpy の SVD の代わりに共分散の固有ベクトルで求める（軸は同じ．符号は幅に影響しない）
void mr_principal_extent(const double* p, int64_t n, double q, double* out) {
    double mean[3] = {0, 0, 0};
    for (int64_t i = 0; i < n; ++i) for (int t = 0; t < 3; ++t) mean[t] += p[i * 3 + t];
    for (int t = 0; t < 3; ++t) mean[t] /= (double)n;
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for (int64_t i = 0; i < n; ++i) {
        Eigen::Vector3d c(p[i * 3] - mean[0], p[i * 3 + 1] - mean[1], p[i * 3 + 2] - mean[2]);
        C += c * c.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);   // 固有値は昇順
    std::vector<double> proj(n);
    for (int a = 0; a < 3; ++a) {
        const Eigen::Vector3d v = es.eigenvectors().col(2 - a);   // 分散の大きい順
        for (int64_t i = 0; i < n; ++i)
            proj[i] = (p[i * 3] - mean[0]) * v(0) + (p[i * 3 + 1] - mean[1]) * v(1) + (p[i * 3 + 2] - mean[2]) * v(2);
        std::vector<double> w(proj);
        const double hi = percentile_linear(w, 100.0 - q);
        const double lo = percentile_linear(w, q);
        out[a] = hi - lo;
    }
}

// 断片ごとの点群の重心と第1主軸（merging.fragment_pca と同じ．主軸は共分散の最大固有ベクトル）．
// 点は全断片を連結した (sum, 3)，offsets は n+1 個
void mr_fragment_pca(const double* p, const int64_t* offsets, int n, double* centers, double* axes) {
    for (int k = 0; k < n; ++k) {
        const int64_t s = offsets[k], e = offsets[k + 1];
        double m[3] = {0, 0, 0};
        for (int64_t i = s; i < e; ++i) for (int t = 0; t < 3; ++t) m[t] += p[i * 3 + t];
        for (int t = 0; t < 3; ++t) m[t] /= (double)(e - s);
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        for (int64_t i = s; i < e; ++i) {
            Eigen::Vector3d c(p[i * 3] - m[0], p[i * 3 + 1] - m[1], p[i * 3 + 2] - m[2]);
            C += c * c.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
        const Eigen::Vector3d v = es.eigenvectors().col(2);
        for (int t = 0; t < 3; ++t) { centers[k * 3 + t] = m[t]; axes[k * 3 + t] = v(t); }
    }
}

// mr_pair_features と同じ特徴を，集合 A の各マスクと集合 B の各マスクの組（nA×nB）だけ計算する
void mr_pair_features_ab(const uint8_t* A, int nA, const uint8_t* B, int nB, int H, int W, int ksize,
                         const double* Z, const double* N, uint8_t* adjacent, double* gap, double* nangle) {
    const int64_t HW = (int64_t)H * W;
    const int r = ksize / 2;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Box> ba(nA), bb(nB);
    std::vector<Dilated> da(nA), db(nB);
    for (int k = 0; k < nA; ++k) { ba[k] = bbox_of(A + k * HW, H, W); da[k] = dilate(A + k * HW, ba[k], H, W, r); }
    for (int l = 0; l < nB; ++l) { bb[l] = bbox_of(B + l * HW, H, W); db[l] = dilate(B + l * HW, bb[l], H, W, r); }
    std::vector<double> zk, zl;
    for (int k = 0; k < nA; ++k) {
        const uint8_t* mk = A + k * HW;
        for (int l = 0; l < nB; ++l) {
            const uint8_t* ml = B + l * HW;
            const int64_t kl = (int64_t)k * nB + l;
            gap[kl] = nan; nangle[kl] = nan;
            bool adj = false;
            const Box bx = intersect(da[k].box, bb[l]);
            for (int y = bx.y0; y < bx.y1 && !adj; ++y)
                for (int x = bx.x0; x < bx.x1; ++x)
                    if (ml[(int64_t)y * W + x] && da[k].at(y, x)) { adj = true; break; }
            adjacent[kl] = adj ? 1 : 0;
            if (!adj) continue;
            zk.clear(); zl.clear();
            double sk[3] = {0, 0, 0}, sl[3] = {0, 0, 0};
            int64_t ck = 0, cl = 0;
            const Box bk = intersect(ba[k], db[l].box);
            for (int y = bk.y0; y < bk.y1; ++y)
                for (int x = bk.x0; x < bk.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    if (!mk[i] || !db[l].at(y, x)) continue;
                    if (Z[i] > 0) zk.push_back(Z[i]);
                    if (N) {
                        const double* nv = N + i * 3;
                        if (std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]) > 0) {
                            sk[0] += nv[0]; sk[1] += nv[1]; sk[2] += nv[2]; ++ck;
                        }
                    }
                }
            const Box bl2 = intersect(bb[l], da[k].box);
            for (int y = bl2.y0; y < bl2.y1; ++y)
                for (int x = bl2.x0; x < bl2.x1; ++x) {
                    const int64_t i = (int64_t)y * W + x;
                    if (!ml[i] || !da[k].at(y, x)) continue;
                    if (Z[i] > 0) zl.push_back(Z[i]);
                    if (N) {
                        const double* nv = N + i * 3;
                        if (std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]) > 0) {
                            sl[0] += nv[0]; sl[1] += nv[1]; sl[2] += nv[2]; ++cl;
                        }
                    }
                }
            if (!zk.empty() && !zl.empty()) gap[kl] = std::fabs(median(zk) - median(zl));
            if (N && ck >= 5 && cl >= 5) {
                double mk3[3], ml3[3];
                for (int t = 0; t < 3; ++t) { mk3[t] = sk[t] / ck; ml3[t] = sl[t] / cl; }
                const double nk = std::sqrt(mk3[0] * mk3[0] + mk3[1] * mk3[1] + mk3[2] * mk3[2]);
                const double nl = std::sqrt(ml3[0] * ml3[0] + ml3[1] * ml3[1] + ml3[2] * ml3[2]);
                double dot = 0;
                for (int t = 0; t < 3; ++t) dot += (mk3[t] / nk) * (ml3[t] / nl);
                dot = std::min(1.0, std::max(-1.0, dot));
                nangle[kl] = std::acos(dot) * 180.0 / M_PI;
            }
        }
    }
}

}  // extern "C"
