#!/bin/bash
set -e

echo "[Memory Stack] Installing core packages for memory + orchestration..."

# Activate virtual environment
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

# Install core memory stack components
pip install --no-cache-dir \
    chromadb==0.4.24 \
    langchain==0.1.20 \
    langchain-core==0.1.52 \
    langchain-community==0.0.32 \
    langchainhub==0.1.15 \
    tiktoken==0.6.0 \
    openai==1.30.1 \
    pydantic==1.10.14

echo "[Memory Stack] Installation complete. ✅"