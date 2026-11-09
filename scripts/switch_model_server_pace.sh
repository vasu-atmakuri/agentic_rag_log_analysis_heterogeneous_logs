#!/bin/zsh
# Restarts the local model server in one of two modes.
#   day   : background priority (efficiency cores only); the Mac stays responsive, runs are slower
#   night : normal priority (all cores); fastest
# Memory caps (one model loaded, 8K context, quantized cache) apply in both modes.
set -e
MODE=${1:-day}
export OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=2m \
       OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
osascript -e 'quit app "Ollama"' 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true
sleep 2
if [[ "$MODE" == "night" ]]; then
  nohup ollama serve > ~/ollama_server.log 2>&1 &
else
  nohup taskpolicy -b ollama serve > ~/ollama_server.log 2>&1 &
fi
sleep 4
if pgrep -f "ollama serve" >/dev/null; then echo "model server running in $MODE mode"; else echo "model server failed to start; see ~/ollama_server.log"; exit 1; fi
