#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# Presets (set MODE=… or just run with the default)
# -----------------------------------------------------------------------------
#   fast    -> 35B-A3B MoE,  no thinking, 32k ctx       [agent loops, daily driver]
#   smart   -> 27B dense,    short thinking, 32k ctx    [hard one-shot questions]
#   bigctx  -> 27B dense,    no thinking, 100k ctx      [reading huge files / many files]
#   custom  -> set MODEL yourself, ignore presets
# -----------------------------------------------------------------------------
MODE="${MODE:-fast}"

# -----------------------------------------------------------------------------
# Knobs - leave empty to use the preset's value, or set via env to override.
#   e.g.  MODE=fast CTX=65536 THINKING=on ./run.sh
# -----------------------------------------------------------------------------
MODEL="${MODEL:-}"                    # path to .gguf
CTX="${CTX:-}"                        # context size in tokens
THINKING="${THINKING:-}"              # on | off
THINK_BUDGET="${THINK_BUDGET:-2048}"  # max thinking tokens when THINKING=on
N_CPU_MOE="${N_CPU_MOE:-}"            # 0 for dense, ~28 for 35B-A3B on 16GB VRAM
B="${B:-}"                            # batch size (prompt eval throughput)
UB="${UB:-}"                          # micro-batch size (compute buffer = ~UB scaled)
PORT="${PORT:-8080}"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BIN="./llama-cpp-turboquant/build/bin/llama-server"
MODELS_DIR="./models"
MODEL_27B="$MODELS_DIR/Qwen3.6-27B-UD-Q3_K_XL.gguf"
MODEL_35B="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
URL_27B="https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-Q3_K_XL.gguf"
URL_35B="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

mkdir -p "$MODELS_DIR"

download_if_missing() {
  local path="$1" url="$2"
  if [[ -f "$path" ]]; then
    echo "✓ $path already present"
  else
    echo "↓ downloading $(basename "$path") ..."
    wget --continue --show-progress -O "$path" "$url"
  fi
}

# -----------------------------------------------------------------------------
# Apply preset (each preset fills in any knob the user didn't set)
# -----------------------------------------------------------------------------
case "$MODE" in
  fast)
    download_if_missing "$MODEL_35B" "$URL_35B"
    : "${MODEL:=$MODEL_35B}"
    : "${CTX:=32768}"
    : "${THINKING:=off}"
    : "${N_CPU_MOE:=28}"
    : "${B:=4096}"
    : "${UB:=2048}"
    ALIAS="qwen36-35b-a3b"
    ;;
  smart)
    download_if_missing "$MODEL_27B" "$URL_27B"
    : "${MODEL:=$MODEL_27B}"
    : "${CTX:=32768}"
    : "${THINKING:=on}"
    : "${N_CPU_MOE:=0}"
    : "${B:=4096}"
    : "${UB:=2048}"
    ALIAS="qwen36-27b"
    ;;
  bigctx)
    download_if_missing "$MODEL_27B" "$URL_27B"
    : "${MODEL:=$MODEL_27B}"
    : "${CTX:=102400}"
    : "${THINKING:=off}"
    : "${N_CPU_MOE:=0}"
    : "${B:=2048}"
    : "${UB:=512}"
    ALIAS="qwen36-27b-bigctx"
    ;;
  custom)
    if [[ -z "$MODEL" ]]; then
      echo "✗ MODE=custom but no MODEL set. Example:"
      echo "    MODE=custom MODEL=./models/foo.gguf CTX=16384 ./run.sh"
      exit 1
    fi
    : "${CTX:=32768}"
    : "${THINKING:=off}"
    : "${N_CPU_MOE:=0}"
    : "${B:=2048}"
    : "${UB:=512}"
    ALIAS="custom"
    ;;
  *)
    echo "✗ Unknown MODE: $MODE  (use: fast | smart | bigctx | custom)"
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------------
if [[ ! -x "$BIN" ]]; then
  echo "✗ llama-server binary not found at $BIN"
  echo "  Did you build the fork? See setup notes."
  exit 1
fi

if ss -tln 2>/dev/null | grep -q ":$PORT "; then
  echo "✗ port $PORT already in use"
  echo "  Run: lsof -i :$PORT  (or set PORT=8081)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Sampling params (Unsloth recommendations for Qwen3.6)
# -----------------------------------------------------------------------------
if [[ "$THINKING" == "on" ]]; then
  TEMP=0.6; TOP_P=0.95; PRESENCE=0.0
  REASONING_FLAGS=( --reasoning on --reasoning-budget "$THINK_BUDGET" )
else
  TEMP=0.7; TOP_P=0.8;  PRESENCE=1.5
  REASONING_FLAGS=( --reasoning off )
fi

# Only pass --n-cpu-moe when there's something to offload
MOE_FLAGS=()
if [[ "$N_CPU_MOE" -gt 0 ]]; then
  MOE_FLAGS=( --n-cpu-moe "$N_CPU_MOE" )
fi

# Hide Ryzen iGPU from ROCm
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Put GPUs in high perf state
# echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
# echo on | sudo tee /sys/class/drm/card*/device/power/control

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat <<EOF

----------------------------------------------
  MODE       : $MODE
  Model      : $MODEL
  Context    : $CTX
  Batch/UB   : $B / $UB
  Thinking   : $THINKING $( [[ $THINKING == on ]] && echo "(budget=$THINK_BUDGET)" )
  CPU-MoE    : $N_CPU_MOE $( [[ $N_CPU_MOE -gt 0 ]] && echo "(experts on RAM)" )
  Port       : $PORT
----------------------------------------------

EOF

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
exec "$BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" \
  -b "$B" -ub "$UB" \
  -ngl 99 \
  -fa 1 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --cache-ram 0 \
  --no-context-shift \
  --ctx-checkpoints 4 \
  --jinja \
  "${REASONING_FLAGS[@]}" \
  "${MOE_FLAGS[@]}" \
  --temp "$TEMP" --top-p "$TOP_P" --top-k 20 --min-p 0.0 \
  --presence-penalty "$PRESENCE" \
  -np 1
