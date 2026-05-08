#!/usr/bin/env bash
set -e

# Usage: build.sh [--build=cpu|cuda|hip] [options]
#
# Backends:
#   cpu  - CPU-only build (default)
#   cuda - NVIDIA CUDA build
#   hip  - AMD ROCm/HIP build
#
# Options (per backend):
#   --cuda-archs=sm_89,sm_61    RTX 5060 Ti + GTX 1060 targets
#   --gpu-targets=gfx1030       AMD RDNA2 target (e.g., RX 6800 XT)
#
# Examples:
#   ./build.sh                          # CPU build (default)
#   ./build.sh --build=cuda             # CUDA with default architectures
#   ./build.sh --build=cuda --cuda-archs=sm_89,sm_61
#   ./build.sh --build=hip --gpu-targets=gfx1030

BUILD="cpu"
CUDA_ARCHS=""
GPU_TARGETS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build=*)
            BUILD="${1#--build=}"
            shift
            ;;
        --cuda-archs=*)
            CUDA_ARCHS="${1#--cuda-archs=}"
            shift
            ;;
        --gpu-targets=*)
            GPU_TARGETS="${1#--gpu-targets=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate backend
case "$BUILD" in
    cpu|cuda|hip) ;;
    *)
        echo "Invalid backend: $BUILD (must be cpu, cuda, or hip)"
        exit 1
        ;;
esac

echo "=== Building llama.cpp with backend: $BUILD ==="

# Initialize submodules
git submodule update --init --recursive

cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

# Common flags
CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
)

case "$BUILD" in
    cpu)
        echo "CPU-only build (no GPU backends)"
        ;;

    cuda)
        echo "CUDA build"

        if [[ -n "$CUDA_ARCHS" ]]; then
            # Convert comma-separated sm_NN to CMake-compatible semicolon-separated list
            # sm_89 -> 89-real, sm_61 -> 61-virtual (add appropriate suffixes)
            CMARK_ARCH_LIST=""
            IFS=',' read -ra ARCHS <<< "$CUDA_ARCHS"
            for arch in "${ARCHS[@]}"; do
                # Determine suffix based on architecture generation
                # sm_6x (Maxwell/Pascal) -> virtual (no PTX support historically)
                # sm_7x (Volta/Turing)   -> virtual
                # sm_8x (Ampere/Ada)     -> real
                # sm_9x (Blackwell)      -> real
                SM_NUM="${arch#sm_}"
                MAJOR="${SM_NUM:0:1}"

                if [[ "$MAJOR" -le 7 ]]; then
                    CMARK_ARCH_LIST+="$((${SM_NUM} ))-virtual;"
                else
                    CMARK_ARCH_LIST+="$((${SM_NUM} ))-real;"
                fi
            done
            # Remove trailing semicolon
            CMARK_ARCH_LIST="${CMARK_ARCH_LIST%;}"
            CMAKE_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CMARK_ARCH_LIST")
            echo "  CUDA architectures: $CMARK_ARCH_LIST"
        else
            echo "  Using default CUDA architectures (auto-detected)"
        fi

        CMAKE_FLAGS+=(-DGGML_CUDA=ON)
        ;;

    hip)
        echo "HIP/ROCm build"

        if [[ -z "$GPU_TARGETS" ]]; then
            # Default to RDNA2 which covers RX 6000 series and most common AMD GPUs
            GPU_TARGETS="gfx1030"
            echo "  No --gpu-targets specified, defaulting to $GPU_TARGETS (RDNA2)"
        fi

        CMAKE_FLAGS+=(-DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGETS")

        # RDNA2 (gfx1030) needs fma4 disabled for rocmWMMA attention
        if [[ "$GPU_TARGETS" == *"gfx1030"* || "$GPU_TARGETS" == *"gfx101"* || "$GPU_TARGETS" == *"gfx9"* ]]; then
            CMAKE_FLAGS+=(-DGGML_HIP_ROCWMMA_FATTN=OFF)
        fi

        echo "  GPU targets: $GPU_TARGETS"
        ;;
esac

echo ""

# Build
cmake -S . -B build -G Ninja "${CMAKE_FLAGS[@]}"
cmake --build build --config Release -j$(nproc)

echo ""
./build/bin/llama-cli --version
