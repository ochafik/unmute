#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""
Metal Orchestrator - Start Unmute services with Metal acceleration (Apple Silicon)

This script orchestrates all Unmute services with native Metal acceleration by:
- Validating essential prerequisites (Rust, Python, llama-server, pnpm)
- Starting all services natively (LLM, STT, TTS, backend, frontend)
- No Docker required - everything runs with Metal acceleration

Usage:
    ./start_metal.py                                  # Start with Metal TTS (localhost)
    UNMUTE_HOST=192.168.1.100 ./start_metal.py        # Start on specific host for network access
    MLX_TTS=1 ./start_metal.py                        # Start with MLX TTS (experimental)


Prerequisites:
- uv (fast Python package manager for backend):
    curl -LsSf https://astral.sh/uv/install.sh | sh
- Homebrew:
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
- Python 3.12 (MUST be Homebrew Python, not conda/miniforge):
    brew install python@3.12
- Rust (via rustup):
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env
- Rust 1.84.0:
    rustup install 1.84.0
- llama.cpp (provides llama-server):
    brew install llama.cpp
- pnpm (for frontend):
    curl -fsSL https://get.pnpm.io/install.sh | sh -
"""

import os
import sys
import subprocess
import signal
import time
import shutil
from pathlib import Path
from typing import Optional, List

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Configuration
ROOT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
HOME_DIR = Path.home()

SERVICE_PORTS = {
    'stt': 8090,
    'tts': 8089,
    'llm': 8091,
    'backend': 8000,
    'frontend': 3000,
}

# Moshi configuration
MOSHI_GIT_URL = "https://github.com/ochafik/moshi"
MOSHI_BRANCH = "apple-silicon-tts"
MOSHI_LOCAL_PATH = HOME_DIR / "github" / "moshi" / "rust" / "moshi-server"

# Track child processes for cleanup
processes: List[subprocess.Popen] = []


def print_color(color: str, message: str) -> None:
    """Print colored message"""
    print(f"{color}{message}{NC}")


def check_command(cmd: str) -> bool:
    """Check if a command exists"""
    return shutil.which(cmd) is not None


def get_system_python() -> Optional[str]:
    """Get Homebrew Python path (not conda/miniforge)"""
    for p in ["/usr/local/bin/python3", "/opt/homebrew/bin/python3"]:
        if Path(p).exists() and "miniforge" not in p and "conda" not in p:
            return p
    return None


def check_prerequisites() -> bool:
    """Check essential prerequisites and return True if all pass"""
    print_color(YELLOW, "\nChecking prerequisites...\n")

    errors = 0
    warnings = 0

    # Python 3.12 (Homebrew) - REQUIRED for PYO3
    print("  Python 3.12 (Homebrew): ", end="")
    python_path = get_system_python()

    if not python_path:
        print_color(RED, "NOT FOUND")
        print_color(RED, "    → Required for moshi-server (PYO3 bindings)")
        print_color(RED, "    → Install with: brew install python@3.12")
        errors += 1
    else:
        try:
            version = subprocess.check_output([python_path, "--version"], text=True).strip()
            print_color(GREEN, f"OK ({python_path}, {version})")
        except:
            print_color(RED, "ERROR checking version")
            errors += 1

    # Rust (rustup) - REQUIRED
    print("  Rust (rustup): ", end="")
    if not check_command("rustup"):
        print_color(RED, "NOT FOUND")
        print_color(RED, "    → Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        errors += 1
    else:
        try:
            version = subprocess.check_output(["rustup", "--version"], text=True).strip().split('\n')[0]
            print_color(GREEN, f"OK ({version})")
        except:
            print_color(RED, "ERROR")
            errors += 1

    # Rust 1.84.0 - REQUIRED
    print("  Rust 1.84.0: ", end="")
    try:
        subprocess.run(
            ["rustup", "run", "1.84.0", "cargo", "--version"],
            capture_output=True,
            check=True
        )
        print_color(GREEN, "OK")
    except:
        print_color(RED, "NOT INSTALLED")
        print_color(RED, "    → Install with: rustup install 1.84.0")
        print_color(YELLOW, "      (Required due to const eval changes in newer Rust)")
        errors += 1

    # llama-server - REQUIRED
    print("  llama-server: ", end="")
    if not check_command("llama-server"):
        print_color(RED, "NOT FOUND")
        print_color(RED, "    → Install with: brew install llama.cpp")
        errors += 1
    else:
        print_color(GREEN, "OK")

    # pnpm - REQUIRED for frontend
    print("  pnpm: ", end="")
    if not check_command("pnpm"):
        print_color(RED, "NOT FOUND")
        print_color(RED, "    → Install with: curl -fsSL https://get.pnpm.io/install.sh | sh -")
        errors += 1
    else:
        try:
            version = subprocess.check_output(["pnpm", "--version"], text=True).strip()
            print_color(GREEN, f"OK (v{version})")
        except:
            print_color(GREEN, "OK")

    # uv - OPTIONAL (can fall back to python -m pip)
    print("  uv: ", end="")
    if not check_command("uv"):
        print_color(YELLOW, "NOT FOUND (will use python -m pip)")
        print_color(YELLOW, "    → Recommended: curl -LsSf https://astral.sh/uv/install.sh | sh")
        warnings += 1
    else:
        try:
            version = subprocess.check_output(["uv", "--version"], text=True).strip().split()[1]
            print_color(GREEN, f"OK (v{version})")
        except:
            print_color(GREEN, "OK")

    # Port availability - WARNING only
    print("\n  Port availability: ", end="")
    conflicts = []
    for port in SERVICE_PORTS.values():
        try:
            subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True,
                check=True
            )
            conflicts.append(port)
        except:
            pass

    if conflicts:
        print_color(YELLOW, f"PORTS IN USE: {', '.join(map(str, conflicts))}")
        print_color(YELLOW, "    → Stop services: pkill -f 'moshi-server|llama-server|uvicorn|next dev'")
        warnings += 1
    else:
        print_color(GREEN, f"OK ({', '.join(map(str, SERVICE_PORTS.values()))} available)")

    print()

    # Summary
    if errors > 0:
        print_color(RED, "══════════════════════════════════════════════════════════════")
        print_color(RED, f"  {errors} prerequisite(s) missing. Please install them first.")
        print_color(RED, "══════════════════════════════════════════════════════════════")
        return False
    elif warnings > 0:
        print_color(YELLOW, f"Prerequisites OK with {warnings} warning(s)")
    else:
        print_color(GREEN, "All prerequisites OK")

    print()
    return True




def start_llm_service(log_file: Path) -> subprocess.Popen:
    """Start LLM service (llama-server)"""
    print_color(BLUE, "Starting LLM (llama-server)...")

    model = os.getenv('UNMUTE_LLM_MODEL', 'bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf')
    context_size = os.getenv('UNMUTE_LLM_CONTEXT_SIZE', '8192')
    port = SERVICE_PORTS['llm']

    with open(log_file, 'w') as log:
        proc = subprocess.Popen(
            [
                'llama-server',
                '-hf', model,
                '-c', context_size,
                '-ngl', '-1',  # Offload all to GPU (Metal)
                '--port', str(port),
                '--host', '0.0.0.0'
            ],
            cwd=ROOT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    processes.append(proc)
    print(f"  PID: {proc.pid}, Log: {log_file}")
    return proc


def start_moshi_service(service_type: str, log_file: Path, mlx: bool = False) -> subprocess.Popen:
    """Start moshi-server for STT or TTS"""
    service_name = f"{'TTS (MLX)' if mlx else 'TTS (Metal)'}" if service_type == 'tts' else "STT"
    print_color(BLUE, f"Starting {service_name} (moshi-server)...")

    # Get system Python for PYO3
    system_python = get_system_python()
    if not system_python:
        print_color(RED, "ERROR: Homebrew Python not found (needed for PYO3)")
        sys.exit(1)

    # Prepare environment
    env = os.environ.copy()
    env['PYO3_PYTHON'] = system_python
    env['PATH'] = f"{Path(system_python).parent}:{env['PATH']}"
    env['CXXFLAGS'] = "-include cstdint"
    env['CMAKE_POLICY_VERSION_MINIMUM'] = "3.5"
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

    if service_type == 'tts':
        env['NO_TORCH_COMPILE'] = "1"

    # Determine cargo command
    cargo_cmd = "cargo"
    try:
        subprocess.run(
            ["rustup", "run", "1.84.0", "cargo", "--version"],
            capture_output=True,
            check=True
        )
        cargo_cmd = "rustup run 1.84.0 cargo"
    except:
        pass

    # Install moshi-server (prefer local fork, fallback to git)
    print(f"  Installing moshi-server...")
    install_cmd = cargo_cmd.split() + ["install", "--features", "metal", "--locked"]

    if False and MOSHI_LOCAL_PATH.exists():
        print(f"    Using local fork: {MOSHI_LOCAL_PATH}")
        install_cmd += ["--path", str(MOSHI_LOCAL_PATH)]
    else:
        print(f"    Using git: {MOSHI_GIT_URL} (branch: {MOSHI_BRANCH})")
        install_cmd += [
            "--git", MOSHI_GIT_URL,
            "--branch", MOSHI_BRANCH,
            "moshi-server"
        ]

    # Run cargo install (this may take time on first run)
    try:
        subprocess.run(
            install_cmd,
            env=env,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print_color(RED, f"  ERROR installing moshi-server: {e.stderr.decode()}")
        sys.exit(1)

    # Install Python dependencies for TTS (moshi-server embeds Python)
    if service_type == 'tts':
        print("  Installing Python dependencies...")

        # Use Homebrew Python's pip (moshi-server needs these in the embedded interpreter)
        pip_cmd = f"{system_python} -m pip"

        # Install TTS dependencies (PyTorch, HuggingFace Hub, etc.)
        # These are needed by tts.py which runs inside moshi-server via PyO3
        core_deps = "huggingface_hub pydantic safetensors torch numpy"
        subprocess.run(
            f'{pip_cmd} install --quiet {core_deps}',
            shell=True,
            env=env,
            capture_output=True,
        )

        # Additional MLX dependencies
        if mlx:
            subprocess.run(
                f'{pip_cmd} install --quiet "mlx>=0.4.0" sentencepiece julius',
                shell=True,
                env=env,
                capture_output=True,
            )

    # Start moshi-server
    config_file = f"services/moshi-server/configs/{'tts_mlx' if mlx else service_type}.toml"
    port = SERVICE_PORTS[service_type]

    with open(log_file, 'w') as log:
        proc = subprocess.Popen(
            [
                str(HOME_DIR / ".cargo" / "bin" / "moshi-server"),
                "worker",
                "--config", config_file,
                "--port", str(port)
            ],
            cwd=ROOT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    processes.append(proc)
    print(f"  PID: {proc.pid}, Log: {log_file}")
    return proc


def start_backend_service(log_file: Path) -> subprocess.Popen:
    """Start backend service (FastAPI via uvicorn)"""
    print_color(BLUE, "Starting Backend (FastAPI)...")

    # Get host from environment (defaults to localhost)
    host = os.getenv('UNMUTE_HOST', 'localhost')

    # Set service URLs
    env = os.environ.copy()
    env['KYUTAI_STT_URL'] = f"ws://{host}:{SERVICE_PORTS['stt']}"
    env['KYUTAI_TTS_URL'] = f"ws://{host}:{SERVICE_PORTS['tts']}"
    env['KYUTAI_LLM_URL'] = f"http://{host}:{SERVICE_PORTS['llm']}"

    cmd = ["uv", "run"] if check_command("uv") else [sys.executable, "-m"]
    cmd.extend([
        "uvicorn", "unmute.main_websocket:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", str(SERVICE_PORTS['backend']),
        "--ws-per-message-deflate=false"
    ])

    with open(log_file, 'w') as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    processes.append(proc)
    print(f"  PID: {proc.pid}, Log: {log_file}")
    return proc


def start_frontend_service(log_file: Path) -> subprocess.Popen:
    """Start frontend service (Next.js)"""
    print_color(BLUE, "Starting Frontend (Next.js)...")

    frontend_dir = ROOT_DIR / "frontend"

    # Run pnpm install first
    print("  Running pnpm install...")
    subprocess.run(
        ["pnpm", "install"],
        cwd=frontend_dir,
        capture_output=True,
    )

    with open(log_file, 'w') as log:
        proc = subprocess.Popen(
            ["pnpm", "dev"],
            cwd=frontend_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    processes.append(proc)
    print(f"  PID: {proc.pid}, Log: {log_file}")
    return proc


def wait_for_service(name: str, url: str, max_attempts: int = 60) -> bool:
    """Wait for a service to be ready"""
    import urllib.request
    import urllib.error

    attempt = 0
    while attempt < max_attempts:
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print_color(GREEN, f"✓ {name} is ready")
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError):
            pass

        attempt += 1
        time.sleep(2)

    print_color(RED, f"✗ {name} failed to start (check logs)")
    return False


def start_all_services(mlx_tts: bool = False) -> bool:
    """Start all Metal-accelerated services"""
    LOGS_DIR.mkdir(exist_ok=True)

    # Get host for health checks (defaults to localhost)
    host = os.getenv('UNMUTE_HOST', 'localhost')

    print_color(YELLOW, f"\nStarting services (logs in {LOGS_DIR}/)...\n")

    # Start services
    start_llm_service(LOGS_DIR / "llm.log")
    start_moshi_service('stt', LOGS_DIR / "stt.log")
    start_moshi_service('tts', LOGS_DIR / "tts.log", mlx=mlx_tts)
    start_backend_service(LOGS_DIR / "backend.log")
    start_frontend_service(LOGS_DIR / "frontend.log")

    print()
    print_color(YELLOW, "Waiting for services to be ready...")
    print_color(YELLOW, "(This may take a few minutes on first run for moshi compilation)")
    print()

    # Wait for health checks
    services = [
        ('LLM', f'http://{host}:8091/health', 180),
        ('STT', f'http://{host}:8090/api/build_info', 300),
        ('TTS', f'http://{host}:8089/api/build_info', 300),
        ('Backend', f'http://{host}:8000/health', 120),
        ('Frontend', f'http://{host}:3000', 120),
    ]

    all_ready = all(wait_for_service(name, url, max_attempts) for name, url, max_attempts in services)
    return all_ready




def cleanup(signum: object = None, frame: object = None) -> None:
    """Cleanup all processes on exit"""
    print()
    print_color(YELLOW, "Shutting down all services...")

    for proc in processes:
        try:
            if proc.poll() is None:
                print(f"Stopping process {proc.pid}...")
                proc.terminate()
        except:
            pass

    time.sleep(2)

    for proc in processes:
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass

    # Kill specific patterns
    for pattern in ['moshi-server worker', 'llama-server.*8091', 'uvicorn.*unmute',
                    'next dev', 'node.*next', 'cargo install']:
        try:
            subprocess.run(['pkill', '-f', pattern], capture_output=True)
        except:
            pass

    print_color(GREEN, "All services stopped.")
    sys.exit(0)


def open_browser() -> None:
    """Open browser to the frontend"""
    host = os.getenv('UNMUTE_HOST', 'localhost')
    url = f"http://{host}:3000"

    print_color(BLUE, f"Opening browser to {url}...")
    try:
        if sys.platform == 'darwin':
            subprocess.run(['open', url], check=False)
        elif sys.platform.startswith('linux'):
            subprocess.run(['xdg-open', url], check=False)
        else:
            print_color(YELLOW, f"Please open {url} in your browser")
    except:
        print_color(YELLOW, f"Please open {url} in your browser")


def main() -> None:
    """Main entry point"""
    print_color(BLUE, "======================================")
    print_color(BLUE, "Starting Unmute with Metal acceleration")
    print_color(BLUE, "======================================")
    print()

    mlx_tts = os.getenv('MLX_TTS') == '1'

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Start all services
    all_ready = start_all_services(mlx_tts)

    print()
    if all_ready:
        print_color(GREEN, "======================================")
        print_color(GREEN, "Unmute is ready!")
        print_color(GREEN, "======================================")
    else:
        print_color(YELLOW, "======================================")
        print_color(YELLOW, "Some services may not be ready yet.")
        print_color(YELLOW, f"Check the logs in {LOGS_DIR}/")
        print_color(YELLOW, "======================================")
    print()

    open_browser()

    print()
    print(f"Press {GREEN}Ctrl+C{NC} to stop all services")
    print()
    print("Monitor logs with:")
    print(f"  tail -f {LOGS_DIR}/llm.log")
    print(f"  tail -f {LOGS_DIR}/stt.log")
    print(f"  tail -f {LOGS_DIR}/tts.log")
    print(f"  tail -f {LOGS_DIR}/backend.log")
    print(f"  tail -f {LOGS_DIR}/frontend.log")
    print()

    # Monitor
    try:
        while True:
            time.sleep(10)
            dead_count = sum(1 for p in processes if p.poll() is not None)
            if dead_count > 2:
                print_color(RED, "Multiple services have stopped. Check logs for errors.")
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
