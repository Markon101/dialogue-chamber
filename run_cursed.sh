#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Dialogue Chamber — Cursed Seance Experiment (Compatibility Wrapper)
# Runs the cursed_seance preset on Gemma 4 + Hermes.
# ==============================================================================

set -euo pipefail
CHAMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHAMBER_DIR"

exec python3 chamber.py --preset cursed_seance "$@"
