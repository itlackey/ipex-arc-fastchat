# syntax=docker/dockerfile:1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# vLLM on Intel Arc GPUs (A-series and B-series) with an OpenAI-compatible API.
# Base image ships PyTorch 2.8 + XPU, the Intel GPU driver stack, and oneAPI runtimes.

FROM intel/intel-extension-for-pytorch:2.8.10-xpu

ENV DEBIAN_FRONTEND=noninteractive

# Intel-documented runtime env vars for Arc GPUs.
# SYCL_CACHE_PERSISTENT avoids per-start kernel recompilation (minutes of startup latency).
# USE_XETLA=OFF is recommended for Arc A-Series and Flex.
# SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS is a significant perf win on kernel 6.2+.
# ZES_ENABLE_SYSMAN enables correct multi-GPU memory reporting.
# UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS permits single allocations >4 GiB (needed for 7B+ in bf16).
ENV SYCL_CACHE_PERSISTENT=1
ENV USE_XETLA=OFF
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV ZES_ENABLE_SYSMAN=1
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

ARG VLLM_VERSION=0.14.0
RUN pip install --no-cache-dir \
        "vllm[xpu]==${VLLM_VERSION}" \
        "transformers>=4.51.3"

ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
ENV HF_HOME=/root/.cache/huggingface

VOLUME ["/root/.cache/huggingface"]
EXPOSE 8000

COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["--model", "Qwen/Qwen2.5-7B-Instruct", "--dtype", "bfloat16"]
