#!/bin/bash
# Start LLM server using llama-server from llama.cpp with Metal acceleration (Apple Silicon)
set -ex
cd "$(dirname "$0")/.."

# Configuration - adjust these as needed
# Default to a small, fast model. Override with UNMUTE_LLM_MODEL environment variable.
# Examples:
#   UNMUTE_LLM_MODEL="bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf"
#   UNMUTE_LLM_MODEL="bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF"  # Speech-optimized
MODEL="${UNMUTE_LLM_MODEL:-bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
CONTEXT_SIZE="${UNMUTE_LLM_CONTEXT_SIZE:-8192}"
PORT="${UNMUTE_LLM_PORT:-8091}"

# Check if llama-server is installed
if ! command -v llama-server &> /dev/null; then
    echo "llama-server not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is required to install llama.cpp"
        echo "Install Homebrew from https://brew.sh/ or install llama.cpp manually"
        exit 1
    fi
    brew install llama.cpp
fi

# Run llama-server with Metal acceleration
# -ngl -1 means offload all layers to GPU (Metal)
# -hf allows loading directly from Hugging Face
llama-server \
    -hf "$MODEL" \
    -c "$CONTEXT_SIZE" \
    -ngl -1 \
    --port "$PORT" \
    --host 0.0.0.0
