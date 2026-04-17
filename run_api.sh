#!/usr/bin/env bash
# Run the a3d-predictor FastAPI server.
#
# Usage:
#   ./run_api.sh                      # default: 0.0.0.0:8000, 1 worker
#   ./run_api.sh --port 9000
#   ./run_api.sh --host 127.0.0.1 --port 8080 --workers 2
#
# The script must be run from the project root (~/a3d-predictor) with the
# 'a3d' conda environment active, or with conda run:
#   conda run -n a3d ./run_api.sh

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Allow flag overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)    HOST="$2";    shift 2 ;;
        --port)    PORT="$2";    shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "Starting a3d-predictor API on http://${HOST}:${PORT}  (workers=${WORKERS})"
echo "Interactive docs: http://${HOST}:${PORT}/docs"
echo ""

exec uvicorn src.serving.api:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}"
