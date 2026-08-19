#!/usr/bin/env bash
# Start the SmartDoc API. Reads API_PORT from .env, defaults to 8006.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
PORT="${API_PORT:-8006}"
echo "Starting API on http://127.0.0.1:${PORT}  (Ctrl+C to stop)"
exec ./venv/bin/uvicorn app.main:app --reload --port "${PORT}"
