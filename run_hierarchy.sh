#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Dialogue Chamber — Weird Hierarchy Experiment (Compatibility Wrapper)
# Runs the 5-agent local weird hierarchy experiment on Gemma 4.
# ==============================================================================

set -euo pipefail
CHAMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHAMBER_DIR"

exec python3 chamber.py --preset weird_hierarchy "$@"
