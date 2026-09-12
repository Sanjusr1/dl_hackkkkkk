#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  python3 serve_dashboard.py "$@"
else
  python serve_dashboard.py "$@"
fi
