#!/usr/bin/env bash
# ==============================================================================
# TarjimaAI: English-to-Uzbek Neural Machine Translation Local Launcher
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PORT=${PORT:-8000}
HOST=${HOST:-"127.0.0.1"}
URL="http://${HOST}:${PORT}"

echo "======================================================================"
echo "  🌟 TARJIMA-AI: LOCAL NEURAL MACHINE TRANSLATION ENGINE"
echo "  Bidirectional GRU + Bahdanau Attention (Apple Silicon / CPU Accelerated)"
echo "======================================================================"
echo ""

# Check if port is already active
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "⚡ TarjimaAI server is already active on ${URL}!"
    echo "Opening web interface in your default browser..."
    open "${URL}" 2>/dev/null || xdg-open "${URL}" 2>/dev/null || true
    echo ""
    echo "To stop the running server, run: kill \$(lsof -t -i:${PORT})"
    exit 0
fi

# Auto-open browser after 2 seconds in background
(
    sleep 2
    echo "Opening ${URL} in your browser..."
    open "${URL}" 2>/dev/null || xdg-open "${URL}" 2>/dev/null || true
) &

echo "Starting local FastAPI server with real PyTorch model weights..."
echo "Target: ${URL}"
echo "Press Ctrl+C to stop the server at any time."
echo ""

python3 -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload
