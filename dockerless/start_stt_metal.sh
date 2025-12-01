#!/bin/bash
# Start STT (Speech-to-Text) server with Metal acceleration (Apple Silicon)
set -ex
cd "$(dirname "$0")/.."

# We need libpython because the TTS uses a Python component. STT and TTS have the same executable, so we need
# to have libpython even if we don't end up using it. For simplicity, we use the same code as for TTS, even though
# you don't need to install any of these Python packages if you're only using the STT.
[ -d .venv ] || uv venv
source .venv/bin/activate

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

# Set PYO3_PYTHON to use a Python with proper library paths
# Homebrew Python has proper framework paths, miniforge/conda does not
if [ -x "/opt/homebrew/bin/python3" ]; then
    export PYO3_PYTHON="/opt/homebrew/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    export PYO3_PYTHON="/usr/local/bin/python3"
fi

# Install moshi-server with Metal features instead of CUDA
# Use --locked to ensure compatible dependency versions
$CARGO install --features metal --locked moshi-server@0.6.4

moshi-server worker --config services/moshi-server/configs/stt.toml --port 8090
