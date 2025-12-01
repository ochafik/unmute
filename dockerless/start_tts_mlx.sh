#!/bin/bash
# Start TTS (Text-to-Speech) server with MLX (Apple Silicon native)
set -ex
cd "$(dirname "$0")/.."

# Use Rust 1.84 - newer versions (1.89+) have breaking const eval changes
CARGO="cargo"
if rustup run 1.84.0 cargo --version &>/dev/null; then
    CARGO="rustup run 1.84.0 cargo"
fi

# Find a suitable system Python (not miniforge/conda which lacks proper framework paths)
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

# Ensure the system Python's bin directory is first in PATH
PYTHON_BIN_DIR=$(dirname "$SYSTEM_PYTHON")
export PATH="$PYTHON_BIN_DIR:$PATH"

# Install Python dependencies from pyproject.toml with MLX extras
echo "Installing MLX dependencies from pyproject.toml..."
$SYSTEM_PYTHON -m pip install --quiet "./dockerless[mlx]" 2>/dev/null || true

# Install moshi-server with Metal features
MOSHI_SERVER_FORK="$HOME/github/moshi/rust/moshi-server"
if [ -d "$MOSHI_SERVER_FORK" ]; then
    $CARGO install --features metal --locked --path "$MOSHI_SERVER_FORK"
else
    echo "Warning: moshi-server fork not found at $MOSHI_SERVER_FORK, using crates.io version"
    $CARGO install --features metal --locked moshi-server@0.6.4
fi

echo "Starting TTS server with MLX backend..."
~/.cargo/bin/moshi-server worker --config services/moshi-server/configs/tts_mlx.toml --port 8089
