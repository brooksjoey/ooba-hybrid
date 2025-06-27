#!/bin/bash
set -e

ROOT="/root/ooba-hybrid"
LOG_FILE="$ROOT/logs/orchestrator.log"

echo "[+] Starting orchestrator..."
cd "$ROOT/webui"
source venv/bin/activate

export PYTHONPATH="$ROOT"

echo "[+] Orchestrator launched at $(date)" | tee -a "$LOG_FILE"
python3 -m orchestrator.orchestrator 2>&1 | tee -a "$LOG_FILE"