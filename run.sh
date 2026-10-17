#!/usr/bin/env bash
# TarjimaAI: English-to-Uzbek Neural Machine Translation Launcher

set -e

PORT=${PORT:-8000}
HOST=${HOST:-"127.0.0.1"}

echo "=========================================================="
echo "  🌟 TarjimaAI | English-Uzbek Neural Machine Translator"
echo "  Featuring Bahdanau Attention & Real-Time Alignment Matrix"
echo "=========================================================="
echo "Starting FastAPI server at http://${HOST}:${PORT} ..."

python3 -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload
