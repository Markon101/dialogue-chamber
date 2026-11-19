#!/data/data/com.termux/files/usr/bin/sh
set -eu
SERVER_BIN="$HOME/gitapps/dory-server-build/bin/llama-server"
DORY_MODEL="${DORY_MODEL:-$HOME/llama.cpp-dory/dory-q5_k_m.gguf}"
export OCL_ICD_ASSUME_ICD_EXTENSION=1
export LD_LIBRARY_PATH="$HOME/gitapps/dory-server-build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SERVER_BIN" \
  -m "$DORY_MODEL" --alias dory --override-kv dory.num_loops=int:3 \
  -c 4096 -ctk q8_0 -ctv q8_0 -fa on -b 128 -ub 128 -t 4 \
  -ngl 999 --fit off --parallel 1 --no-context-shift \
  --cache-ram 0 --ctx-checkpoints 0 \
  --jinja --reasoning off --temp 1 --top-p 0.95 --top-k 0 --min-p 0 \
  --repeat-penalty 1 -n 256 --host 127.0.0.1 --port 8080 --verbosity 4
