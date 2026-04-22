#!/usr/bin/env bash
set -euo pipefail

echo "=== vLLM-XPU container starting ==="
echo "Args: $*"

if command -v sycl-ls >/dev/null 2>&1; then
    echo "--- sycl-ls ---"
    sycl-ls || true
    echo "---------------"
fi

exec python3 -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --device xpu \
    "$@"
