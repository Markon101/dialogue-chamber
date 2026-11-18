#!/usr/bin/env bash
# Chamber-owned equivalent of serve_best.sh. Never kills an existing server.
set -euo pipefail
SD_DIR="${SNAPDRAGON_DIR:-$HOME/projects/llama-snapdragon}"
source "$SD_DIR/env.sh"
MODE="${1:-gemma4-htp}"
PORT="${2:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-4096}"
THREADS="${THREADS:-6}"
MIN_MB="${MIN_MB:-600}"
SERVER_BIN="$SD_DIR/llama-b12034/bin/llama-server"
GUARD_SH="$SD_DIR/guard.sh"
case "$MODE" in
  gemma4-htp)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/gemma-4-E4B-it-Q4_0.gguf" --device HTP0 -ngl 99 \
      -ot "per_layer_token_embd.weight=CPU,token_embd.weight=CPU,output.weight=CPU" \
      --no-repack --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 -t "$THREADS" -tb "$THREADS" \
      --reasoning on --reasoning-format deepseek --reasoning-budget 200
    ;;
  gemma4-cpu-mtp)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/gemma-4-E4B-it-Q4_0.gguf" -md "$HOME/models/mtp-gemma-4-E4B-it-Q4_0.gguf" \
      --device none -ngl 0 --draft-max 4 --draft-min 1 --draft-p-min 0.75 \
      --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 -t "$THREADS" -tb "$THREADS" \
      --reasoning on --reasoning-format deepseek --reasoning-budget 200
    ;;
  llama1b-htp)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/Llama-3.2-1B-Instruct-Q4_0.gguf" --device HTP0 -ngl 99 \
      --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 -t "$THREADS" -tb "$THREADS"
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
