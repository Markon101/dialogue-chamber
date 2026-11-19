#!/usr/bin/env bash
# ==============================================================================
# Dialogue Chamber — Unified Launcher Interface (Python-First Orchestration)
# ==============================================================================
# Usage:
#   ./run_prompt.sh "Your custom prompt here"
#   ./run_prompt.sh -p hermes_vs_local_debate -t 8 "Debate this topic..."
#   ./run_prompt.sh -m llama1b-htp "Fast prompt on Llama 1B..."
#   ./run_prompt.sh --context-info
#   ./run_prompt.sh --doctor
# ==============================================================================

set -euo pipefail

CHAMBER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHAMBER_DIR"

source "$CHAMBER_DIR/launcher_lib.sh"

# If no arguments provided at all, enter interactive prompt helper
if [ $# -eq 0 ]; then
  echo "=========================================================="
  echo "           THE DIALOGUE CHAMBER — CUSTOM RUNNER           "
  echo "=========================================================="
  echo ""
  echo "Select a preset:"
  echo "  1) two_agent_baseline      (Gemma 4 + Hermes, clean dialogue)"
  echo "  2) hermes_vs_local_debate  (Gemma 4 + Hermes, claims & challenges on board)"
  echo "  3) weird_hierarchy         (5 local Gemma agents, non-transitive council)"
  echo "  4) cursed_seance           (2 Gemmas + Hermes, haunting investigation)"
  echo "  5) shared_world            (2 Gemmas + Hermes, collaborative markdown world)"
  echo "  6) creative_collision      (Surreal inventions & strange mathematics)"
  echo ""
  read -rp "Enter choice [1-6, default 1]: " PRESET_CHOICE
  case "$PRESET_CHOICE" in
    2) PRESET="hermes_vs_local_debate";;
    3) PRESET="weird_hierarchy";;
    4) PRESET="cursed_seance";;
    5) PRESET="shared_world";;
    6) PRESET="creative_collision";;
    *) PRESET="two_agent_baseline";;
  esac

  read -rp "Number of turns [default: preset-defined]: " INPUT_TURNS

  echo ""
  echo "Enter your prompt / topic below (press Enter, then Ctrl+D when finished, or type single line):"
  PROMPT="$(cat)"

  CMD=(python3 chamber.py --preset "$PRESET")
  if [ -n "$INPUT_TURNS" ]; then
    CMD+=(--turns "$INPUT_TURNS")
  fi
  if [ -n "$PROMPT" ]; then
    CMD+=(--topic "$PROMPT")
  fi
  exec "${CMD[@]}"
fi

# Delegate directly to Python orchestrator
exec python3 chamber.py "$@"
