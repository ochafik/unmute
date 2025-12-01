#!/bin/bash
# Start TTS (Text-to-Speech) server with Metal acceleration (Apple Silicon)
set -ex
cd "$(dirname "$0")/.."

# On macOS with Metal, we don't need LD_LIBRARY_PATH as we do on Linux with CUDA.
# The Metal backend is linked statically.

# A fix for building Sentencepiece on newer compilers
export CXXFLAGS="-include cstdint"

# Fix for CMake 4.x compatibility with older CMakeLists.txt
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Use Rust 1.84 - newer versions (1.89+) have breaking const eval changes
CARGO="cargo"
if rustup run 1.84.0 cargo --version &>/dev/null; then
    CARGO="rustup run 1.84.0 cargo"
fi

# Enable MPS fallback to CPU for unsupported PyTorch operations
# This allows TTS to run on Apple Silicon, with some ops falling back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Find a suitable system Python (not miniforge/conda which lacks proper framework paths)
# Priority: homebrew Intel > homebrew ARM > system Python
SYSTEM_PYTHON=""
for p in /usr/local/bin/python3 /opt/homebrew/bin/python3; do
    if [ -x "$p" ]; then
        SYSTEM_PYTHON="$p"
        break
    fi
done

if [ -z "$SYSTEM_PYTHON" ]; then
    echo "Error: No suitable system Python found. Install Homebrew Python: brew install python@3.12"
    exit 1
fi

# Set PYO3_PYTHON for Rust compilation
export PYO3_PYTHON="$SYSTEM_PYTHON"

# Ensure the system Python's bin directory is first in PATH so moshi-server finds it
# This is critical: moshi-server spawns Python subprocesses for TTS
PYTHON_BIN_DIR=$(dirname "$SYSTEM_PYTHON")
export PATH="$PYTHON_BIN_DIR:$PATH"

# Install moshi-server with Metal features instead of CUDA
# Use --locked to ensure compatible dependency versions
# If you already have moshi-server installed and things are not working because of a
# prior CUDA build, you might have to force a rebuild with --force.
$CARGO install --features metal --locked moshi-server@0.6.4

# Install Python dependencies needed for TTS to the system Python
# The moshi package includes torch for Metal/MPS support
$SYSTEM_PYTHON -m pip install --quiet moshi huggingface_hub pydantic julius sentencepiece safetensors 2>/dev/null || true

~/.cargo/bin/moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089
