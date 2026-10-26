#!/usr/bin/env bash
# mask_reconstruction の C++ 基本処理（mr_kernels.cpp）を共有ライブラリにする．
#   bash mask_reconstruction/cpp/build.sh   → mask_reconstruction/cpp/build/libmrkernels.so
# -ffp-contract=off：積和の融合を止め，numpy と同じ浮動小数点の結果にする
set -e
cd "$(dirname "$0")"
mkdir -p build
g++ -O3 -march=native -ffp-contract=off -fPIC -shared -std=c++17 -I/usr/include/eigen3 mr_kernels.cpp -o build/libmrkernels.so
echo "built: $(pwd)/build/libmrkernels.so"
