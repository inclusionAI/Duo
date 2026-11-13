#!/usr/bin/env bash
set -euo pipefail
runtime=$(cd -- "$(dirname -- "$0")" && pwd)
source "$runtime/environment.sh"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
cd "$DUO_RUNTIME_DIR"
exec "$DUO_WEB_PYTHON" web/gateway.py
