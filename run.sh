#!/usr/bin/env bash
# run.sh — reproduce NeuroScan figures from stored results
# Usage: bash run.sh [--figures | --web]
set -euo pipefail

MODE="${1:---figures}"

case "$MODE" in
  --figures)
    echo "Generating publication figures..."
    python generate_figures.py
    echo "Figures saved to figures/"
    ;;
  --web)
    echo "Starting web app (frontend + backend)..."
    echo "Backend: see src/web/neuroscan-web/backend/main.py"
    echo "Frontend: cd src/web/neuroscan-web && npm run dev"
    ;;
  *)
    echo "Usage: bash run.sh [--figures | --web]"
    exit 1
    ;;
esac
