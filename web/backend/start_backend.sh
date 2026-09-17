#!/usr/bin/env bash
set -euo pipefail
# Run this from the Hoola project root (the directory containing hoola/, agents/, rl/).
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
