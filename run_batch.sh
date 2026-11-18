#!/usr/bin/env bash
# ==============================================================================
# Dialogue Chamber — Batch Runner (Compatibility Wrapper)
# Delegates to Python-first batch runner with manifests, tracking, and recovery.
# ==============================================================================

set -euo pipefail
CHAMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHAMBER_DIR"

exec python3 chamber.py --batch "$@"
