#!/bin/bash
# Start all Unmute services with Metal acceleration (Apple Silicon) and open the browser
# Uses process groups for clean shutdown
set -e
cd "$(dirname "$0")/.."

# Enable MPS fallback to CPU for unsupported PyTorch operations on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
LOGS_DIR="./logs"
mkdir -p "$LOGS_DIR"

# File to track child PIDs
PID_FILE="$LOGS_DIR/.pids"
echo "" > "$PID_FILE"

# Function to kill a process tree
kill_tree() {
    local pid=$1
    local sig=${2:-TERM}

    # Get all child processes
    local children=$(pgrep -P $pid 2>/dev/null || true)

    # Kill children first
    for child in $children; do
        kill_tree $child $sig
    done

    # Kill the process itself
    if kill -0 $pid 2>/dev/null; then
        kill -$sig $pid 2>/dev/null || true
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down all services...${NC}"

    # Read PIDs and kill them
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo "Stopping process tree for PID $pid..."
                kill_tree $pid TERM
            fi
        done < "$PID_FILE"
    fi

    # Wait a moment for graceful shutdown
    sleep 2

    # Force kill any remaining processes
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo "Force killing PID $pid..."
                kill_tree $pid KILL
            fi
        done < "$PID_FILE"
    fi

    # Clean up specific known processes that might have been spawned
    pkill -f "moshi-server worker" 2>/dev/null || true
    pkill -f "llama-server.*8091" 2>/dev/null || true
    pkill -f "uvicorn.*unmute" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "node.*next" 2>/dev/null || true
    pkill -f "cargo install.*moshi" 2>/dev/null || true

    rm -f "$PID_FILE"
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Function to start a service and track its PID
start_service() {
    local name="$1"
    local script="$2"
    local log="$3"

    echo -e "${BLUE}Starting $name...${NC}"

    # Start in background (macOS compatible - no setsid)
    bash "$script" > "$log" 2>&1 &
    local pid=$!
    echo $pid >> "$PID_FILE"
    echo -e "  PID: $pid, Log: $log"
}

# Function to wait for a service to be ready
wait_for_service() {
    local name="$1"
    local url="$2"
    local max_attempts="${3:-60}"
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo -e "${RED}✗ $name failed to start (check logs)${NC}"
    return 1
}

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Starting Unmute with Metal acceleration${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v llama-server &> /dev/null; then
    echo -e "${RED}Error: llama-server not found. Install with: brew install llama.cpp${NC}"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo not found. Install Rust from https://rustup.rs${NC}"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    echo -e "${RED}Error: pnpm not found. Install with: curl -fsSL https://get.pnpm.io/install.sh | sh -${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites OK${NC}"
echo ""

# Start all services
echo -e "${YELLOW}Starting services (logs in $LOGS_DIR/)...${NC}"
echo ""

start_service "LLM (llama-server)" \
    "./dockerless/start_llm_metal.sh" \
    "$LOGS_DIR/llm.log"

start_service "STT (moshi-server)" \
    "./dockerless/start_stt_metal.sh" \
    "$LOGS_DIR/stt.log"

start_service "TTS (moshi-server MLX)" \
    "./dockerless/start_tts_mlx.sh" \
    "$LOGS_DIR/tts.log"

start_service "Backend (FastAPI)" \
    "./dockerless/start_backend.sh" \
    "$LOGS_DIR/backend.log"

start_service "Frontend (Next.js)" \
    "./dockerless/start_frontend.sh" \
    "$LOGS_DIR/frontend.log"

echo ""
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
echo -e "${YELLOW}(This may take a few minutes on first run due to compilation/downloads)${NC}"
echo ""

# Wait for services (with longer timeout for first-time compilation)
READY=true

wait_for_service "LLM" "http://localhost:8091/health" 180 || READY=false
wait_for_service "Backend" "http://localhost:8000/health" 120 || READY=false
wait_for_service "Frontend" "http://localhost:3000" 120 || READY=false
wait_for_service "STT" "http://localhost:8090/api/build_info" 300 || READY=false
wait_for_service "TTS" "http://localhost:8089/api/build_info" 300 || READY=false

echo ""
if [ "$READY" = true ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Unmute is ready!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    # Open browser
    echo -e "${BLUE}Opening browser...${NC}"
    if command -v open &> /dev/null; then
        open "http://localhost:3000"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:3000"
    else
        echo -e "${YELLOW}Please open http://localhost:3000 in your browser${NC}"
    fi
else
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Some services may not be ready yet.${NC}"
    echo -e "${YELLOW}Check the logs in $LOGS_DIR/${NC}"
    echo -e "${YELLOW}======================================${NC}"
fi

echo ""
echo -e "Press ${GREEN}Ctrl+C${NC} to stop all services"
echo ""
echo -e "Monitor logs with:"
echo -e "  tail -f $LOGS_DIR/llm.log"
echo -e "  tail -f $LOGS_DIR/stt.log"
echo -e "  tail -f $LOGS_DIR/tts.log"
echo -e "  tail -f $LOGS_DIR/backend.log"
echo -e "  tail -f $LOGS_DIR/frontend.log"
echo ""

# Keep script running
while true; do
    sleep 10

    # Check if any critical process died
    if [ -f "$PID_FILE" ]; then
        dead_count=0
        while read -r pid; do
            if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
                dead_count=$((dead_count + 1))
            fi
        done < "$PID_FILE"

        if [ $dead_count -gt 2 ]; then
            echo -e "${RED}Multiple services have stopped. Check logs for errors.${NC}"
        fi
    fi
done
