# syntax=docker/dockerfile:1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# vLLM on Intel Arc GPUs (A-series and B-series) with an OpenAI-compatible API.
# Uses Intel's pre-built vLLM XPU image which ships the correct PyTorch/IPEX
# versions and pre-compiled SYCL kernels, avoiding pip version-conflict issues.

ARG VLLM_TAG=0.14.1-xpu
FROM intel/vllm:${VLLM_TAG}

ENV DEBIAN_FRONTEND=noninteractive

# Intel-documented runtime env vars for Arc GPUs.
# SYCL_CACHE_PERSISTENT: avoids per-start JIT kernel recompilation (can take minutes on first run).
# USE_XETLA=OFF: required for Arc A-series and Flex stability; harmless no-op on Xe2 (B-series).
# SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS: perf win via legacy L0 adapter (A-series, oneAPI <2025.3).
# UR_L0_USE_IMMEDIATE_COMMANDLISTS: equivalent for L0 V2 adapter (B-series / Xe2, oneAPI 2025.3+).
# ZES_ENABLE_SYSMAN: enables accurate VRAM reporting across multiple GPUs.
# UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS: required for single allocations >4 GiB (7B+ models in bf16/fp16).
# VLLM_WORKER_MULTIPROC_METHOD=spawn: SYCL contexts are not fork-safe; prevents deadlocks on multi-GPU.
ENV SYCL_CACHE_PERSISTENT=1
ENV USE_XETLA=OFF
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
ENV ZES_ENABLE_SYSMAN=1
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
ENV HF_HOME=/root/.cache/huggingface

VOLUME ["/root/.cache/huggingface"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health

COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["--model", "Qwen/Qwen2.5-7B-Instruct", "--dtype", "bfloat16", "--max-model-len", "8192"]
