#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Ollama Memory Proxy - Installer"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+."
    exit 1
fi

PYTHON=python3
PIP=pip3

# Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "[WARNING] Ollama not found. Install from https://ollama.com"
    echo "          The proxy requires Ollama running on port 11434."
    echo ""
fi

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    $PYTHON -m venv venv
    source venv/bin/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt

# Pre-download embedding model
echo ""
echo "[3/3] Pre-downloading embedding model (~80MB, one-time)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "  Start the proxy:  python run.py"
echo "  (activate venv first: source venv/bin/activate)"
echo "  Then point clients to http://localhost:11435"
echo "============================================================"
