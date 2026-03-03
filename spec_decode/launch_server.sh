#!/usr/bin/env bash
# Launch vLLM OpenAI API server in the foreground for speculative decoding benchmark.
#
# Usage:
#   ./launch_server.sh                    # Start with baseline (no speculative decoding). Ctrl+C to stop.
#   CONFIG=ngram_k5 ./launch_server.sh   # Start with configs/ngram_k5 preset
#   ./launch_server.sh stop              # Kill any process using the port (e.g. stray server)
#
# Presets: configs/*.json (e.g. suffix_decoding, ngram_k5, ngram_k3). CONFIG=baseline = no preset.
#
# Environment (defaults):
#   CONFIG                 Preset name (default: baseline). Uses configs/<CONFIG>.json
#   CUDA_VISIBLE_DEVICES   GPU ids (default: 0,1,2,3)
#   PORT                   Server port (default: 8000)
#   MODEL                  Model name (default: Qwen/Qwen3-30B-A3B)
#   TP_SIZE                Tensor parallel size (default: 2)
#   MAX_MODEL_LEN          Max model length (default: 32768)
#   GPU_MEM_UTIL           GPU memory utilization (default: 0.80)
#   SEED                   Random seed (default: 42)
#   ENABLE_EP              Enable expert parallel (default: 1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
PID_FILE="${SCRIPT_DIR}/.server.pid"

stop_server() {
  local port="${1:-8000}"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping server (PID $pid)..."
      kill "$pid" 2>/dev/null || true
      sleep 2
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$PID_FILE"
    echo "Server stopped."
  fi
  if command -v fuser &>/dev/null; then
    if fuser -k "${port}/tcp" 2>/dev/null; then
      echo "Freed port $port."
    fi
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  PORT="${PORT:-8000}"
  stop_server "$PORT"
  exit 0
fi
CUDA_VISIBLE_DEVICES=0,1
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
SEED="${SEED:-42}"
ENABLE_EP="${ENABLE_EP:-1}"

# Kill any stale server on this port
if command -v fuser &>/dev/null; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 2
fi

CMD=(
  vllm serve "$MODEL"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --seed "$SEED"
  --port "$PORT"
  --disable-log-requests
)

if [[ "$ENABLE_EP" == "1" ]]; then
  CMD+=(--enable-expert-parallel)
fi

CONFIG="${CONFIG:-baseline}"
if [[ "$CONFIG" != "baseline" ]]; then
  CONFIG_FILE="${CONFIGS_DIR}/${CONFIG}.json"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE (available: baseline, $(cd "$CONFIGS_DIR" 2>/dev/null && ls -1 *.json 2>/dev/null | sed 's/\.json$//' | tr '\n' ',' | sed 's/,$//')"
    exit 1
  fi
  SPEC_CONFIG_JSON=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$CONFIG_FILE")
  CMD+=(--speculative-config "$SPEC_CONFIG_JSON")
  echo "Using preset: $CONFIG ($CONFIG_FILE)"
fi

echo "Starting vLLM server on port $PORT. Press Ctrl+C to stop."
echo "In another terminal run the benchmark after you see 'Application startup complete'."
exec "${CMD[@]}"
