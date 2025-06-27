set -e

ROOT="/root/ooba-hybrid"
LOG_FILE="$ROOT/logs/init_stack.log"
VENV_DIR="$ROOT/webui/venv"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) [+] Setting up full stack at $ROOT" | tee -a "$LOG_FILE"

# Create directory layout
mkdir -p "$ROOT"/{scripts/bootstrap,orchestrator/prompt_profiles,memory/{db,ingest},config,logs,webui}

# Create files
touch "$ROOT"/{README.md}
touch "$ROOT"/memory/schema.json
touch "$ROOT"/logs/{brain.log,orchestrator.log}
touch "$ROOT"/config/{apis.env,routing.yaml,settings.json}
touch "$ROOT"/orchestrator/{brain.py,agent_orchestrator.py,dynamic_router.py,memory_chain.py,orchestrator.py,agents.yaml}
touch "$ROOT"/orchestrator/prompt_profiles/{planner_prompt.txt,summarizer_prompt.txt}
touch "$ROOT"/scripts/{run_orchestrator.sh,ingest_context.py,sync_tasks.py,update_brain.sh}
chmod +x "$ROOT/scripts/run_orchestrator.sh"

# Bootstrap scripts
for f in install-deps.sh install_memory_stack.sh ooba-setup.sh; do
  cp "$ROOT/scripts/bootstrap/$f" "$ROOT/scripts/$f"
  chmod +x "$ROOT/scripts/$f"
done

# Ensure venv + python deps
echo "$(timestamp) [+] Checking python3.10-venv..." | tee -a "$LOG_FILE"
if ! dpkg -s python3.10-venv >/dev/null 2>&1; then
    echo "$(timestamp) [-] Missing python3.10-venv — installing..." | tee -a "$LOG_FILE"
    apt update && apt install -y python3.10-venv
fi

# Venv init
echo "$(timestamp) [+] Initializing Python venv..." | tee -a "$LOG_FILE"
cd "$ROOT/webui"
python3 -m venv venv --system-site-packages || {
    echo "$(timestamp) [!] venv creation failed." | tee -a "$LOG_FILE"
    exit 1
}

# Activate and install requirements
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

REQ="$ROOT/scripts/bootstrap/requirements.txt"
if [[ -f "$REQ" ]]; then
  echo "$(timestamp) [+] Installing requirements from $REQ" | tee -a "$LOG_FILE"
  pip install -r "$REQ"
fi

# Finalize environment
echo "source $VENV_DIR/bin/activate" >> ~/.bashrc

echo "$(timestamp) [✓] Stack fully initialized." | tee -a "$LOG_FILE"
