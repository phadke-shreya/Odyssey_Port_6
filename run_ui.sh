#!/usr/bin/env bash
# Start the SmartDoc UI. Points itself at the API using API_URL from .env.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
export API_URL="${API_URL:-http://127.0.0.1:${API_PORT:-8006}}"
echo "Starting UI, talking to API at ${API_URL}"
exec ./venv/bin/streamlit run streamlit_app.py
