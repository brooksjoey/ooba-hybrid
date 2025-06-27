#!/bin/bash

# Define the expected directory structure and files
expected_paths=(
    "scripts/Bootstrap/init_stack.sh"
    "scripts/Bootstrap/install_memory_stack.sh"
    "scripts/Bootstrap/install-deps.sh"

    "scripts/Startup-Reboot/start-on-reboot.sh"
    "scripts/Startup-Reboot/run-ooba.sh"
    "scripts/Startup-Reboot/run_orchestrator.sh"

    "scripts/Live-Ops/ingest_context.py"

    "orchestrator/orchestrator.py"
    "orchestrator/agent_orchestrator.py"
    "orchestrator/dynamic_router.py"
    "orchestrator/memory_chain.py"
    "orchestrator/agents.yaml"

    "memory/"
    "logs/"
    "config/"
    "oobabooga/"
    ".env"
    "requirements.txt"
)

# Base directory
base_dir="$HOME/ooba-hybrid"

# Check if base directory exists
if [ ! -d "$base_dir" ]; then
    echo "❌ Error: $base_dir not found."
    exit 1
fi

cd "$base_dir" || exit 1

# Check each expected path
missing=0
for path in "${expected_paths[@]}"; do
    if [ -e "$path" ]; then
        echo "✅ Found: $path"
    else
        echo "❌ Missing: $path"
        ((missing++))
    fi
done

# Final summary
if [ $missing -eq 0 ]; then
    echo -e "\n🎉 All expected files and folders are present."
else
    echo -e "\n⚠️  $missing missing file(s)/directory(ies)."
fi
