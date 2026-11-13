#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$RELEASE_ROOT/third_party/sglang-omni:$RELEASE_ROOT/third_party/duo${PYTHONPATH:+:$PYTHONPATH}"
export TORCHTITAN_LIVE=1
PYTHON=${PYTHON:-$RELEASE_ROOT/.venv-duplex/bin/python}
COMMAND=${1:-}
shift || true
case "$COMMAND" in
  checkpoint)
    exec "$PYTHON" "$RELEASE_ROOT/tools/prepare_models.py" "$@"
    ;;
  intent)
    exec "$PYTHON" "$RELEASE_ROOT/third_party/sglang-omni/scripts/merge_duo_intent_adapter.py" "$@"
    ;;
  *)
    echo 'Usage: bash prepare.sh {checkpoint|intent} [options]' >&2
    exit 2
    ;;
esac
