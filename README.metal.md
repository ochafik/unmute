# Running Unmute on Apple Silicon (M1/M2/M3/M4)

This guide covers running Unmute locally on macOS with Apple Silicon, using Metal GPU acceleration for all ML workloads.

## Prerequisites

Install the following software in order:

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.12 (MUST be Homebrew Python, not conda/miniforge)
brew install python@3.12

# 3. Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 4. Install Rust 1.84.0 (required for moshi-server due to const eval changes in newer versions)
rustup install 1.84.0

# 5. Install llama.cpp (provides llama-server for LLM)
brew install llama.cpp

# 6. Install pnpm (for frontend)
curl -fsSL https://get.pnpm.io/install.sh | sh -

# 7. Install uv (fast Python package manager for backend)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify prerequisites:**
```bash
python3 --version          # Should be 3.12.x
rustup show                # Should show 1.84.0 available
llama-server --version     # Should work
pnpm --version             # Should work
uv --version               # Should work
```

## Clone Repositories

```bash
# Clone Unmute (use apple-silicon-support branch for MLX TTS)
git clone https://github.com/nichochar/unmute.git
cd unmute
git checkout apple-silicon-support

# Clone moshi fork with Apple Silicon fixes (required for TTS)
mkdir -p ~/github
git clone https://github.com/ochafik/moshi.git ~/github/moshi
cd ~/github/moshi
git checkout apple-silicon-tts
cd -  # Back to unmute directory
```

## Quick Start

```bash
# This starts all 5 services and opens the browser when ready
./dockerless/start_all_metal.sh
```

Press `Ctrl+C` to stop all services.

## Architecture

| Service | Port | Description |
|---------|------|-------------|
| LLM | 8091 | llama-server with Metal GPU acceleration |
| STT | 8090 | Speech-to-Text (moshi-server) |
| TTS | 8089 | Text-to-Speech (MLX with 8-bit quantization) |
| Backend | 8000 | FastAPI websocket server |
| Frontend | 3000 | Next.js web app |

## First Run

The first run takes longer because it:

1. Downloads LLM model (~1GB for default Llama-3.2-1B)
2. Compiles moshi-server with Metal support (~5-10 minutes)
3. Downloads TTS/STT models from HuggingFace
4. Installs all npm/Python dependencies

Subsequent runs are much faster.

## TTS Backend

The default uses **MLX with 8-bit quantization** for optimal performance on Apple Silicon.

**Performance (M2 Max):**
- MLX 8-bit: ~100-130ms frame times (default, smooth audio)
- MLX 4-bit: ~80-100ms frame times (faster, may affect quality)
- PyTorch MPS: ~200ms+ frame times (fallback, choppy audio)
- Target for real-time: 80ms per frame

*Performance may vary on other Apple Silicon chips.*

### Using PyTorch MPS Instead

If you need to use PyTorch MPS instead of MLX (not recommended):

```bash
# Start services individually with PyTorch MPS TTS
./dockerless/start_llm_metal.sh &
./dockerless/start_stt_metal.sh &
./dockerless/start_tts_metal.sh &    # PyTorch MPS instead of MLX
./dockerless/start_backend.sh &
./dockerless/start_frontend.sh &
```

## Logs

Monitor service logs in the `logs/` directory:

```bash
tail -f logs/llm.log      # LLM server
tail -f logs/stt.log      # Speech-to-Text
tail -f logs/tts.log      # Text-to-Speech
tail -f logs/backend.log  # Backend API
tail -f logs/frontend.log # Frontend
```

## Configuration

### LLM Model

Customize the LLM model via environment variables:

```bash
# Use a larger model for better quality
export UNMUTE_LLM_MODEL="bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Use a speech-optimized model
export UNMUTE_LLM_MODEL="bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF"

# Adjust context size
export UNMUTE_LLM_CONTEXT_SIZE=16384

./dockerless/start_all_metal.sh
```

### TTS Quantization

The default config uses 8-bit quantization. To adjust, edit `services/moshi-server/configs/tts_mlx.toml`:

```toml
[modules.tts_py.py]
quantize = 8   # Default: good balance of speed and quality
# quantize = 4   # Maximum speed, may affect quality
# quantize = 0   # No quantization (slowest, highest quality)
```

## Troubleshooting

### TTS not working

Make sure you have the moshi fork cloned at `~/github/moshi`:

```bash
ls ~/github/moshi/rust/moshi-server  # Should exist
```

If missing, clone it:
```bash
git clone https://github.com/ochafik/moshi.git ~/github/moshi
cd ~/github/moshi && git checkout apple-silicon-tts
```

### Rust compilation fails

Ensure you're using Rust 1.84.0:

```bash
rustup install 1.84.0
# The scripts auto-detect 1.84.0, but you can also set it as default:
rustup default 1.84.0
```

### Python issues

Ensure using Homebrew Python (not conda/miniforge):

```bash
which python3
# Should be /usr/local/bin/python3 (Intel Mac)
# or /opt/homebrew/bin/python3 (Apple Silicon)

# NOT: ~/miniforge3/bin/python3 or similar
```

If using conda, deactivate it:
```bash
conda deactivate
```

### Services fail to start

Check individual service logs:

```bash
# Check what's running
lsof -i :8089  # TTS
lsof -i :8090  # STT
lsof -i :8091  # LLM
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill stuck processes
pkill -f moshi-server
pkill -f llama-server
pkill -f uvicorn
pkill -f "next dev"
```

### moshi-server needs rebuild

If you previously built moshi-server for CUDA or a different configuration:

```bash
# Force rebuild
cargo install --features metal --locked --force moshi-server@0.6.4

# Or from the fork
cd ~/github/moshi/rust/moshi-server
cargo install --features metal --locked --force --path .
```

## Development

### Python Dependencies

The `dockerless/pyproject.toml` manages Python dependencies with optional extras:

```bash
# Install MPS (PyTorch Metal) dependencies
pip install "./dockerless[mps]"

# Install MLX dependencies
pip install "./dockerless[mlx]"
```

### Individual Service Scripts

| Script | Purpose |
|--------|---------|
| `start_all_metal.sh` | Start all services with process management |
| `start_llm_metal.sh` | LLM server only |
| `start_stt_metal.sh` | Speech-to-Text only |
| `start_tts_metal.sh` | Text-to-Speech (PyTorch MPS) |
| `start_tts_mlx.sh` | Text-to-Speech (MLX - recommended) |
| `start_backend.sh` | FastAPI backend only |
| `start_frontend.sh` | Next.js frontend only |

## Related Repositories

- [Unmute](https://github.com/nichochar/unmute) - Main project
- [Moshi Fork](https://github.com/ochafik/moshi/tree/apple-silicon-tts) - Apple Silicon TTS fixes
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference with Metal
