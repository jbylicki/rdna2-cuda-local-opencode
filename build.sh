#!/usr/bin/env bash
set -e

git submodule update --init --recursive

cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
      cmake -S . -B build -G Ninja \
        -DGGML_HIP=ON \
        -DGPU_TARGETS=gfx1030 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP_ROCWMMA_FATTN=OFF
cmake --build build --config Release -j$(nproc)

./build/bin/llama-cli --version
