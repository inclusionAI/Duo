#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${DUPLEX_ENV:-$RELEASE_ROOT/deploy.env}"
export PYTHONPATH="$RELEASE_ROOT/third_party/sglang-omni:${SGLANG_ROOT:?}/python:$RELEASE_ROOT/third_party/duo"
export TORCHTITAN_LIVE=1
cd "$RELEASE_ROOT/third_party/sglang-omni"
PYTHON=${PYTHON:-$RELEASE_ROOT/.venv-duplex/bin/python}
ASR_PORT=${ASR_PORT:-18084}
BACKEND_PORT=${BACKEND_PORT:-18082}
WEB_PORT=${WEB_PORT:-18081}

case "${1:-}" in
  asr)
    export CUDA_VISIBLE_DEVICES=${ASR_GPU:-3}
    exec "$PYTHON" -m sglang_omni.cli serve \
      --model-path "${ASR_MODEL:?}" --model-name duo-asr \
      --asr.engine.max_running_requests 1 --asr.engine.enable_torch_compile false \
      --asr.engine.mem_fraction_static 0.1 --asr.factory.pre_lm_cache_max_entries 1 \
      --asr.factory.prefill_coalesce_wait_ms 0.01 --port "$ASR_PORT"
    ;;
  backend)
    # ASR occupies one rank of the shared Intent device set.
    export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    OPTIONS=()
    if [[ -n ${REALIZATION_MODELOPT_CHECKPOINT:-} ]]; then
      OPTIONS+=(--realization-modelopt-checkpoint "$REALIZATION_MODELOPT_CHECKPOINT" --realization-fp8-direct-kernels)
    fi
    if [[ -n ${FP8_PROFILE_PATTERN:-} ]]; then
      OPTIONS+=(--realization-fp8-attention-calibration "$FP8_PROFILE_PATTERN")
    fi
    if [[ -n ${FP8_ATTENTION_LIBRARY:-} ]]; then
      OPTIONS+=(--realization-fp8-attention-library "$FP8_ATTENTION_LIBRARY")
    fi
    exec "$PYTHON" scripts/serve_duo_realtime.py \
      --codec-model-path "${CODEC_MODEL:?}" \
      --intent-model-path "${INTENT_MODEL:?}" --intent-adapter-path "${INTENT_ADAPTER:?}" \
      --intent-backend native --intent-tp-size 4 --intent-device cuda:0 \
      --intent-worker-devices 0 1 2 3 \
      --realization-checkpoint "${REALIZATION_CHECKPOINT:?}" --realization-backend native \
      --realization-stage dmd --steps 4 --flow-shift 5 --realization-tp-size 8 --realization-sp-size 1 \
      --realization-worker-devices 0 1 2 3 4 5 6 7 --realization-device cuda:0 \
      --realization-optimize-copy-sync --overlap-intent-codec \
      --realization-kv-window-tokens "${REALIZATION_KV_WINDOW_TOKENS:-65536}" \
      --realization-reference-window-chunks "${REALIZATION_REFERENCE_WINDOW_CHUNKS:-14}" \
      --realization-kv-workspace-capacity-tokens "${REALIZATION_KV_WORKSPACE_CAPACITY_TOKENS:-131072}" \
      --codec-device cuda:4 --codec-worker-devices 4 5 6 7 \
      --codec-decode-backend solarx --codec-attention-backend torch_cudnn_sdpa \
      --asr-device "cuda:${ASR_GPU:-3}" \
      --asr-base-url "http://127.0.0.1:$ASR_PORT/v1" --asr-model-name duo-asr \
      --height 768 --width 576 --max-context-tokens "${INTENT_KV_WINDOW_TOKENS:-32768}" \
      --intent-kv-window-tokens "${INTENT_KV_WINDOW_TOKENS:-32768}" --max-new-tokens 128 \
      --warmup-sample-directory "${SAMPLE_ROOT:?}" \
      --warmup-reference-image "${PORTRAIT_ROOT:?}/seaside/portrait.jpg" \
      --warmup-reference-image "$PORTRAIT_ROOT/dorm/portrait.jpg" \
      --warmup-blocks "${WARMUP_BLOCKS:-40}" --port "$BACKEND_PORT" "${OPTIONS[@]}"
    ;;
  web)
    exec "$PYTHON" playground/duo/server.py \
      --backend-url "http://127.0.0.1:$BACKEND_PORT" \
      --sample-directory "${SAMPLE_ROOT:?}" --portrait-directory "${PORTRAIT_ROOT:?}" \
      --port "$WEB_PORT"
    ;;
  check)
    exec "$PYTHON" "$RELEASE_ROOT/tools/check_runtime.py" "$RELEASE_ROOT"
    ;;
  *)
    echo 'Usage: bash serve.sh {asr|backend|web|check}' >&2
    exit 2
    ;;
esac
