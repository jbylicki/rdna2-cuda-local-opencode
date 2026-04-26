#!/usr/bin/env bash
set -e

mkdir -p ~/.config/opencode
CONFIG_FILE=~/.config/opencode/opencode.json

if [ -f "$CONFIG_FILE" ]; then
    echo "Warning: $CONFIG_FILE already exists."
    read -r -p "Do you want to override it? [y/N] " response
    case "$response" in
        [yY]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

cat > "$CONFIG_FILE" <<'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama.cpp",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "qwen36": {
          "name": "Qwen3.6 (local)"
        }
      }
    }
  },
  "model": "llamacpp/qwen36-27b",
  "small_model": "llamacpp/qwen36-27b"
}
EOF
