# pkoscik's Local OpenCode setup

A local LLM inference setup using llama.cpp (TurboQuant fork) with ROCm on AMD hardware, serving as an OpenCode backend.

## Hardware

- **GPU:** AMD Radeon RX 6800 XT (16 GB, `gfx1030`, RDNA2)
- **CPU:** AMD Ryzen 7 7700X (`gfx1036` - hidden from llama.cpp)
- **RAM:** 64 GB system
- **OS:** Arch-based

## Quick Start

```bash
# 1. Build llama.cpp TurboQuant fork
./build.sh

# 2. Download models and start server (default: fast / 35B-A3B)
./run.sh

# 3. Point OpenCode to http://127.0.0.1:8080/v1
```

## Presets

`run.sh` has four built-in presets. Set `MODE` to select one:

| Preset | Model | Context | Thinking | CPU-MoE | Batch / UB | Best for |
|--------|-------|---------|----------|---------|------------|----------|
| `fast` | 35B-A3B MoE | 32k | off | 28 | 4096 / 2048 | Daily agent work |
| `smart` | 27B dense | 32k | on (2048 budget) | 0 | 4096 / 2048 | Hard one-shot questions |
| `bigctx` | 27B dense | 100k | off | 0 | 2048 / 512 | Reading large codebases |
| `custom` | (you set) | (you set) | (you set) | (you set) | 2048 / 512 | Experimenting |

```bash
./run.sh                           # default: fast
MODE=smart ./run.sh                # 27B with thinking
MODE=bigctx ./run.sh               # 27B with 100k context
MODE=fast CTX=65536 ./run.sh       # override context
MODE=fast THINKING=on ./run.sh     # force thinking on
MODE=fast N_CPU_MOE=32 ./run.sh    # tweak expert offload
MODE=bigctx UB=256 ./run.sh        # tighter compute buffer if OOM
```

## Setup

### 1. Install dependencies

```bash
sudo pacman -Syu

# ROCm SDK
sudo pacman -S rocm-hip-sdk rocm-hip-runtime rocm-opencl-runtime \
               hipblas rocblas rocsolver rocsparse rocwmma

# Build dependencies
sudo pacman -S base-devel cmake ninja git curl

# GPU permissions
sudo usermod -aG video,render $USER
```

Reboot or re-login for group changes to take effect.

### 2. Verify ROCm

```bash
rocminfo | grep -E 'gfx|Name'
#   Name: gfx1030        - your RX 6800 XT
#   Name: gfx1036        - Ryzen iGPU (must be hidden at runtime)
```

### 3. Build the TurboQuant fork

```bash
./build.sh
```

This clones `https://github.com/TheTom/llama-cpp-turboquant`, checks out the `feature/turboquant-kv-cache` branch, and builds with ROCm.

> **`GGML_HIP_ROCWMMA_FATTN=OFF` is required for RDNA2** (the WMMA fast-attention path only exists on RDNA3+)

### 4. Configure OpenCode

```bash
sudo pacman -S opencode

./init_opencode.sh
```

The model ID (`qwen36`) is just a label - `llama-server` serves whatever GGUF is loaded. No config change is needed when switching modes; just stop the server, switch mode, restart, and start a fresh session in OpenCode.

## Models

Two GGUFs are downloaded automatically by `run.sh`:

| Model | File | Size | Best for |
|-------|------|------|----------|
| Qwen3.6-27B Q3_K_XL | `Qwen3.6-27B-UD-Q3_K_XL.gguf` | 13.5 GB | Hard one-shot questions, reasoning |
| Qwen3.6-35B-A3B Q4_K_XL | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22 GB | Agent loops, daily driver |

The 27B is smarter but slower; the 35B-A3B is a MoE model that activates only 3B params per token - faster, with remaining experts offloaded to system RAM.

## Tuning Knobs

| Knob | What it does | Higher | Lower |
|------|-------------|--------|-------|
| `CTX` | Context window in tokens | Remembers more, slower, more VRAM | Snappier, less memory |
| `B` / `UB` | Batch / micro-batch for prompt eval | Faster ingest, more VRAM | Slower ingest, fits bigger contexts |
| `THINKING` | Internal reasoning before answering | Better one-shot quality, much slower | Faster, fine for agent loops |
| `THINK_BUDGET` | Max thinking tokens per turn | More deliberation | Avoids token spirals |
| `N_CPU_MOE` | MoE experts offloaded to RAM | Less VRAM, slightly slower | More VRAM, slightly faster |
| `--cache-type-k/v` | KV cache precision | turbo3 = 3-bit (fits more context) | f16/bf16 = full precision (safer) |
| `-ngl` | Layers on GPU (99 = all) | More on GPU = faster | More on CPU = more RAM |
| `-np` | Parallel conversation slots | Multiple clients | Single client gets full KV |

### Quantization

We use Unsloth's Dynamic 2.0 (UD-prefix) non-uniform quantization:

| Quant | Size | Quality vs BF16 | Note |
|-------|------|-----------------|---------|
| UD-Q2_K_XL | ~10 GB | ~92% | Only if really squeezed |
| UD-Q3_K_XL | ~13.5 GB | ~99% | Sweet spot for 27B |
| UD-Q4_K_XL | ~16.5 GB | ~99.5% | 35B-A3B default |
| UD-Q6_K | ~22 GB | ~99.9% | Too big without offload |

