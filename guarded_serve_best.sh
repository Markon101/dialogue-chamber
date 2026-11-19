#!/data/data/com.termux/files/usr/bin/bash
# Chamber-owned equivalent of serve_best.sh. Never kills an existing server.
set -euo pipefail
SD_DIR="${SNAPDRAGON_DIR:-$HOME/projects/llama-snapdragon}"
source "$SD_DIR/env.sh"
MODE="${1:-gemma4-htp}"
PORT="${2:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-4096}"
THREADS="${THREADS:-6}"
MTP_MIN_MB="${MIN_MB:-1200}"
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
  gemma4-htp-q8)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/gemma-4-E4B-it-Q4_0.gguf" --device HTP0 -ngl 99 \
      -ot "per_layer_token_embd.weight=CPU,token_embd.weight=CPU,output.weight=CPU" \
      --no-repack --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 --ctx-checkpoints 0 \
      -ctk q8_0 -ctv q8_0 --flash-attn on \
      --batch-size 512 --ubatch-size 128 -t "$THREADS" -tb "$THREADS" \
      --reasoning on --reasoning-format deepseek --reasoning-budget 200
    ;;
  gemma4-htp-mtp)
    exec "$GUARD_SH" -m "$MTP_MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/gemma-4-E4B-it-Q4_0.gguf" -md "$HOME/models/mtp-gemma-4-E4B-it-Q4_0.gguf" \
      --device HTP0 -ngl 99 \
      -ot "per_layer_token_embd.weight=CPU,token_embd.weight=CPU,output.weight=CPU" \
      --no-repack --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
      --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 --ctx-checkpoints 0 \
      -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --flash-attn on \
      --batch-size 512 --ubatch-size 128 -t "$THREADS" -tb "$THREADS" \
      --reasoning on --reasoning-format deepseek --reasoning-budget 200 --verbosity 4
    ;;
  gemma4-cpu-mtp)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/gemma-4-E4B-it-Q4_0.gguf" -md "$HOME/models/mtp-gemma-4-E4B-it-Q4_0.gguf" \
      --device none -ngl 0 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
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
  llama1b-htp-q8)
    exec "$GUARD_SH" -m "$MIN_MB" -t 86400 -- "$SERVER_BIN" \
      -m "$HOME/models/Llama-3.2-1B-Instruct-Q4_0.gguf" --device HTP0 -ngl 99 \
      --fit off --host "$HOST" --port "$PORT" --parallel 1 \
      --ctx-size "$CTX_SIZE" --cache-ram 0 --ctx-checkpoints 0 \
      -ctk q8_0 -ctv q8_0 --flash-attn on \
      -t "$THREADS" -tb "$THREADS"
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
