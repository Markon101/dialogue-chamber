#!/usr/bin/env bash
# ==============================================================================
# Dialogue Chamber — Shared Launcher Library
# Safe server management, process ownership tracking, locking, and wake-locks.
# ==============================================================================

# Ensure directory paths
CHAMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPDRAGON_DIR="${SNAPDRAGON_DIR:-$HOME/projects/llama-snapdragon}"
SERVE_SCRIPT="${SERVE_SCRIPT:-$SNAPDRAGON_DIR/serve_best.sh}"
LOCK_DIR="${LOCK_DIR:-$CHAMBER_DIR/.locks}"
RUN_LOCK_FILE="$LOCK_DIR/chamber_run.lock"
SERVER_STATE_DIR="$LOCK_DIR/servers"
WAKELOCK_HOLDERS_DIR="$LOCK_DIR/wakelock_holders"

mkdir -p "$LOCK_DIR" "$SERVER_STATE_DIR" "$WAKELOCK_HOLDERS_DIR"

# ------------------------------------------------------------------------------
# Concurrency & Experiment Locking
# ------------------------------------------------------------------------------
CHAMBER_LOCK_FD=200

chamber_acquire_lock() {
  local wait_sec="${1:-0}"
  mkdir -p "$LOCK_DIR"
  exec 200>"$RUN_LOCK_FILE"
  if [ "$wait_sec" -gt 0 ]; then
    if ! flock -w "$wait_sec" 200; then
      local holder
      holder=$(cat "$RUN_LOCK_FILE.pid" 2>/dev/null || echo "unknown")
      echo "Error: Could not acquire experiment lock after ${wait_sec}s. Held by PID: $holder" >&2
      return 1
    fi
  else
    if ! flock -n 200; then
      local holder
      holder=$(cat "$RUN_LOCK_FILE.pid" 2>/dev/null || echo "unknown")
      echo "Error: Another Dialogue Chamber experiment is currently running (PID: $holder)." >&2
      echo "Pass --force-unlock if a previous run crashed, or wait for it to finish." >&2
      return 1
    fi
  fi
  echo "$$" > "$RUN_LOCK_FILE.pid"
  return 0
}

chamber_release_lock() {
  if [ -f "$RUN_LOCK_FILE.pid" ]; then
    local holder
    holder=$(cat "$RUN_LOCK_FILE.pid" 2>/dev/null || true)
    if [ "$holder" = "$$" ]; then
      rm -f "$RUN_LOCK_FILE.pid" 2>/dev/null || true
    fi
  fi
  flock -u 200 2>/dev/null || true
  exec 200>&- 2>/dev/null || true
}

chamber_force_unlock() {
  rm -f "$RUN_LOCK_FILE" "$RUN_LOCK_FILE.pid" 2>/dev/null || true
  echo "[Chamber Launcher] Experiment lock force-cleared."
}

# ------------------------------------------------------------------------------
# Termux Wake-lock Management (Multi-process reference counted)
# ------------------------------------------------------------------------------
chamber_acquire_wakelock() {
  mkdir -p "$WAKELOCK_HOLDERS_DIR"
  touch "$WAKELOCK_HOLDERS_DIR/$$"
  if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock 2>/dev/null || true
  fi
}

chamber_release_wakelock() {
  rm -f "$WAKELOCK_HOLDERS_DIR/$$" 2>/dev/null || true
  # Clean up any stale holder files whose PIDs are no longer active
  local active_holders=0
  for holder_file in "$WAKELOCK_HOLDERS_DIR"/*; do
    [ -e "$holder_file" ] || continue
    local pid
    pid=$(basename "$holder_file")
    if kill -0 "$pid" 2>/dev/null; then
      active_holders=$((active_holders + 1))
    else
      rm -f "$holder_file" 2>/dev/null || true
    fi
  done

  # Only unlock if no other active chamber processes need the wake-lock
  if [ "$active_holders" -eq 0 ]; then
    if command -v termux-wake-unlock >/dev/null 2>&1; then
      termux-wake-unlock 2>/dev/null || true
    fi
  fi
}

# ------------------------------------------------------------------------------
# Model Mode / Name Normalization
# ------------------------------------------------------------------------------
# Maps friendly aliases to (model_mode, model_file, default_ctx)
chamber_resolve_model() {
  local input="${1:-gemma4-htp}"
  case "$input" in
    gemma|gemma4|gemma4-htp)
      echo "gemma4-htp gemma-4-E4B-it-Q4_0.gguf 32768"
      ;;
    gemma-mtp|gemma4-mtp|gemma4-cpu-mtp)
      echo "gemma4-cpu-mtp gemma-4-E4B-it-Q4_0.gguf 32768"
      ;;
    llama1b|llama-1b|llama1b-htp)
      echo "llama1b-htp Llama-3.2-1B-Instruct-Q4_0.gguf 131072"
      ;;
    *)
      # Custom mode or filename
      local fname
      fname=$(basename "$input")
      echo "$input $fname 4096"
      ;;
  esac
}

# ------------------------------------------------------------------------------
# Server Inspection & Health
# ------------------------------------------------------------------------------
chamber_check_health() {
  local host="${1:-127.0.0.1}"
  local port="${2:-8080}"
  local timeout="${3:-2}"
  local code
  code=$(curl -s -m "$timeout" -o /dev/null -w "%{http_code}" "http://$host:$port/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    return 0
  fi
  # Fallback: check /v1/models if /health returns 404 or differs
  code=$(curl -s -m "$timeout" -o /dev/null -w "%{http_code}" "http://$host:$port/v1/models" 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    return 0
  fi
  return 1
}

# Queries running server for loaded model ID
chamber_get_loaded_model() {
  local host="${1:-127.0.0.1}"
  local port="${2:-8080}"
  local timeout="${3:-2}"
  local resp
  resp=$(curl -s -m "$timeout" "http://$host:$port/v1/models" 2>/dev/null || true)
  if [ -n "$resp" ]; then
    # Parse first model id using jq or python
    local mid
    mid=$(echo "$resp" | jq -r '.data[0].id // empty' 2>/dev/null || true)
    if [ -n "$mid" ]; then
      basename "$mid"
      return 0
    fi
  fi
  echo ""
  return 1
}

# ------------------------------------------------------------------------------
# Process Ownership Tracking
# ------------------------------------------------------------------------------
chamber_server_state_file() {
  local port="${1:-8080}"
  echo "$SERVER_STATE_DIR/server_${port}.json"
}

# Checks if a server on port is running AND owned by this harness
chamber_is_server_owned() {
  local port="${1:-8080}"
  local state_file
  state_file=$(chamber_server_state_file "$port")
  if [ ! -f "$state_file" ]; then
    return 1
  fi
  local s_pid
  s_pid=$(jq -r '.pid // empty' "$state_file" 2>/dev/null || true)
  if [ -z "$s_pid" ]; then
    return 1
  fi
  if kill -0 "$s_pid" 2>/dev/null; then
    # Verify it looks like our server / guard process
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/$s_pid/cmdline" 2>/dev/null || true)
    if [[ "$cmd" =~ llama|guard|serve_best ]]; then
      return 0
    fi
  fi
  # Stale state file
  rm -f "$state_file" 2>/dev/null || true
  return 1
}

chamber_get_owned_pid() {
  local port="${1:-8080}"
  local state_file
  state_file=$(chamber_server_state_file "$port")
  if [ -f "$state_file" ]; then
    jq -r '.pid // empty' "$state_file" 2>/dev/null || true
  fi
}

# ------------------------------------------------------------------------------
# Safe Server Start / Verification / Stop
# ------------------------------------------------------------------------------
chamber_ensure_server() {
  local host="${1:-127.0.0.1}"
  local port="${2:-8080}"
  local requested_mode="${3:-gemma4-htp}"
  local requested_model="${4:-gemma-4-E4B-it-Q4_0.gguf}"
  local log_file="${5:-$SNAPDRAGON_DIR/logs/gemma4_server.log}"
  local allow_switch="${6:-false}"

  mkdir -p "$(dirname "$log_file")"

  if chamber_check_health "$host" "$port" 2; then
    local loaded
    loaded=$(chamber_get_loaded_model "$host" "$port" 2 || true)
    local req_base
    req_base=$(basename "$requested_model")

    if [ -n "$loaded" ] && [ "$loaded" != "$req_base" ]; then
      echo "[Chamber Launcher] Model mismatch detected on http://$host:$port!" >&2
      echo "  Currently running: $loaded" >&2
      echo "  Requested model:   $req_base ($requested_mode)" >&2

      if [ "$allow_switch" = "true" ]; then
        if chamber_is_server_owned "$port"; then
          echo "[Chamber Launcher] --allow-server-switch specified. Safely stopping owned server on port $port..."
          chamber_stop_owned_server "$port"
          sleep 2
        else
          echo "Error: Cannot switch server on port $port because it is an UNOWNED/EXTERNAL server." >&2
          echo "We will not kill an external server. Please run on an alternate port or terminate it manually." >&2
          return 1
        fi
      else
        echo "Error: Server on port $port is running model '$loaded', not requested '$req_base'." >&2
        echo "To safely replace an owned server, pass --allow-server-switch." >&2
        return 1
      fi
    else
      echo "[Chamber Launcher] llama-server is ALREADY running and healthy on http://$host:$port (model: ${loaded:-$req_base})"
      return 0
    fi
  fi

  # Server not running or stopped for switch; launch it
  if [ ! -x "$SERVE_SCRIPT" ]; then
    echo "Error: Server launch script not found or not executable: $SERVE_SCRIPT" >&2
    return 1
  fi

  echo "[Chamber Launcher] Starting llama-server ($requested_mode) on http://$host:$port ..."
  local state_file
  state_file=$(chamber_server_state_file "$port")

  # Run server detached in new process group
  setsid "$SERVE_SCRIPT" "$requested_mode" "$port" > "$log_file" 2>&1 < /dev/null &
  local s_pid=$!

  # Record ownership state immediately
  cat << EOF > "$state_file"
{
  "pid": $s_pid,
  "port": $port,
  "host": "$host",
  "mode": "$requested_mode",
  "model": "$requested_model",
  "started_at": $(date +%s),
  "log_file": "$log_file"
}
EOF

  echo -n "[Chamber Launcher] Waiting for server to initialize"
  local server_ready="false"
  for i in $(seq 1 90); do
    if chamber_check_health "$host" "$port" 1; then
      server_ready="true"
      echo " — READY (${i}s)!"
      break
    fi
    # Check if process died during startup
    if ! kill -0 "$s_pid" 2>/dev/null; then
      echo ""
      echo "Error: llama-server process $s_pid exited unexpectedly during startup. Log excerpt:" >&2
      tail -n 25 "$log_file" >&2
      rm -f "$state_file" 2>/dev/null || true
      return 1
    fi
    echo -n "."
    sleep 1
  done

  if [ "$server_ready" != "true" ]; then
    echo ""
    echo "Error: llama-server failed to become healthy within 90s. Log excerpt:" >&2
    tail -n 25 "$log_file" >&2
    chamber_stop_owned_server "$port" || true
    return 1
  fi

  return 0
}

chamber_stop_owned_server() {
  local port="${1:-8080}"
  local state_file
  state_file=$(chamber_server_state_file "$port")

  if [ ! -f "$state_file" ]; then
    return 0
  fi

  local s_pid
  s_pid=$(jq -r '.pid // empty' "$state_file" 2>/dev/null || true)

  if [ -n "$s_pid" ] && kill -0 "$s_pid" 2>/dev/null; then
    echo "[Chamber Launcher] Stopping owned server process group (PID $s_pid) on port $port..."
    # Send SIGTERM to process group and process
    kill -TERM -- "-$s_pid" 2>/dev/null || kill -TERM "$s_pid" 2>/dev/null || true

    # Wait up to 10 seconds for clean shutdown
    for _ in $(seq 1 10); do
      if ! kill -0 "$s_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    # Force kill if still lingering
    if kill -0 "$s_pid" 2>/dev/null; then
      echo "[Chamber Launcher] Forcing termination of PID $s_pid..."
      kill -9 -- "-$s_pid" 2>/dev/null || kill -9 "$s_pid" 2>/dev/null || true
    fi
  fi

  rm -f "$state_file" 2>/dev/null || true
  return 0
}
