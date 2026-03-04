#!/usr/bin/env bash
# Run speculative decoding sweep: for each CONFIG (baseline, suffix_decoding, ngram),
# start server → run guidellm benchmark → stop server, then repeat with next config.
#
# Usage:
#   ./sweep.sh <data.jsonl>
#   ./sweep.sh /path/to/e22_sessions_guidellm.jsonl
#   MAX_SECONDS=30 ./sweep.sh data/my_bench.jsonl
#
# Arguments:
#   data.jsonl           Path to JSONL data file for guidellm benchmark (required)
#
# Environment (optional):
#   MAX_SECONDS          Max seconds per rate (default: 60)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Data path from user (required)
DATA="${1:?Usage: $0 <data.jsonl>}"

# Fixed sweep settings (not overridable)
CONFIGS="baseline,suffix_decoding,ngram_k4"
PORT="8000"
RATES="1,2,3,4,5,6,7,8"
WARMUP="0.1"
COOLDOWN="0.1"

MAX_SECONDS="${MAX_SECONDS:-60}"
# Results under vllm-benchmark/results/<timestamp>
OUT_DIR="${BENCH_ROOT}/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
echo "Results will be saved to: $OUT_DIR"

wait_for_server() {
  local port="$1"
  local url="http://localhost:${port}/health"
  local max_wait="${2:-300}"
  local elapsed=0
  echo "Waiting for server on port $port (up to ${max_wait}s)..."
  while ! curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q 200; do
    sleep 5
    elapsed=$((elapsed + 5))
    if [[ $elapsed -ge $max_wait ]]; then
      echo "Timeout waiting for server at $url"
      return 1
    fi
    # echo "  ${elapsed}s ..."
  done
  echo "Server ready (${elapsed}s)."
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 5
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Server stopped."
  fi
  SERVER_PID=""
  if command -v fuser &>/dev/null; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
  fi
}

run_one() {
  local config="$1"
  local out_name="$2"
  local result_path="${OUT_DIR}/${out_name}"

  echo "========== CONFIG: $config =========="
  stop_server

  CONFIG="$config" PORT="$PORT" "$SCRIPT_DIR/launch_server.sh" &
  SERVER_PID=$!
  trap 'stop_server; exit 130' INT TERM

  if ! wait_for_server "$PORT"; then
    stop_server
    return 1
  fi

  guidellm benchmark \
    --target "http://localhost:${PORT}" \
    --profile constant \
    --rate "$RATES" \
    --max-seconds "$MAX_SECONDS" \
    --data "$DATA" \
    --warmup "$WARMUP" \
    --cooldown "$COOLDOWN" \
    --output-path "$result_path"

  echo "Saved: $result_path"
  stop_server
  trap - INT TERM
  echo ""
}

if [[ ! -f "$DATA" ]]; then
  echo "Data file not found: $DATA"
  exit 1
fi

IFS=',' read -ra CONFIG_LIST <<< "$CONFIGS"
for config in "${CONFIG_LIST[@]}"; do
  config="${config// /}"
  # result filename: baseline -> result-baseline.json, suffix_decoding -> result-suffix_decoding.json
  out_name="result-${config}.json"
  run_one "$config" "$out_name"
done

echo "Sweep done. Results in $OUT_DIR:"
for config in "${CONFIG_LIST[@]}"; do
  config="${config// /}"
  echo "  result-${config}.json"
done
