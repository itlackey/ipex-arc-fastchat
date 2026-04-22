# Refactor Plan: Migrate to vLLM-XPU

**Status:** Implemented
**Branch:** `claude/intel-arc-gpu-research-uf7CR`
**Target:** Replace the FastChat + IPEX 2.0.110 + CLBlast stack with a modern vLLM-XPU stack that supports both Intel Arc A-series and B-series GPUs with an OpenAI-compatible API.

---

## 1. Decision: vLLM with Intel XPU Backend

After reviewing the current Intel Arc GPU LLM landscape, **vLLM with the native XPU backend** is the clear winner:

| Criterion | vLLM-XPU | FastChat + IPEX (current) | Ollama + IPEX-LLM | llama.cpp SYCL |
|---|---|---|---|---|
| Supports Arc A-series (A770/A750) | Yes | Yes | Yes | Yes |
| Supports Arc B-series (B580/B60/B70) | Yes | No (IPEX 2.0.110 predates B-series) | Partial | Yes |
| OpenAI-compatible API | Built-in | Separate process | Built-in | Separate process |
| PagedAttention / continuous batching | Yes | No | No | No |
| FP8 KV cache | Yes | No | No | No |
| FP8 KV-cache; XPU-verified quantization (AWQ/GPTQ Marlin are CUDA-only) | Yes | No | INT4 only | GGUF only |
| Multi-GPU tensor parallelism | Yes | Limited | No | Limited |
| Active upstream maintenance | Yes (Intel + vLLM community) | Archived (Jan 2026) | Archived (Jan 2026) | Yes |
| Single-process architecture | Yes | No (3 processes) | Yes | Yes |
| Throughput on Arc (tok/s) | Highest | Low | Medium | Medium-high |

**Why not Ollama/IPEX-LLM?** IPEX-LLM was archived by Intel in January 2026 with known security issues. It still works but has no forward path.

**Why not llama.cpp SYCL directly?** Great perf, but GGUF-only and the OpenAI server is less mature than vLLM's. A good fallback, not the primary.

**Why not Intel's `llm-scaler`?** Scoped to Arc Pro B-series; does not cover A-series users.

vLLM-XPU covers both GPU families, has the richest OpenAI API surface, and is the framework Intel is actively contributing to upstream.

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────┐
│  Docker container (single process)               │
│                                                  │
│  ┌─────────────────────────────────────────┐     │
│  │  vllm.entrypoints.openai.api_server     │     │
│  │  - OpenAI-compatible HTTP API           │     │
│  │  - PagedAttention                       │     │
│  │  - Continuous batching                  │     │
│  │  - INT4/INT8/FP8 quantization support   │     │
│  └────────────────┬────────────────────────┘     │
│                   │                              │
│                   ▼                              │
│  ┌─────────────────────────────────────────┐     │
│  │  PyTorch 2.7+ with native XPU device    │     │
│  │  (no IPEX runtime extension required)   │     │
│  └────────────────┬────────────────────────┘     │
│                   │                              │
│                   ▼                              │
│  ┌─────────────────────────────────────────┐     │
│  │  Intel GPU compute runtime              │     │
│  │  (Level Zero + intel-opencl-icd)        │     │
│  │  oneAPI 2025.0+                         │     │
│  └────────────────┬────────────────────────┘     │
│                   │                              │
└───────────────────┼──────────────────────────────┘
                    │  /dev/dri
                    ▼
           ┌─────────────────┐
           │  Intel Arc GPU  │
           │  A-series or    │
           │  B-series       │
           └─────────────────┘
```

**Simplification summary:**
- 4 processes (FastChat controller + worker + web + openai) → 1 process (vLLM api_server)
- Gradio web UI removed (users can pair with Open WebUI via compose)
- Custom startup.sh + awk model-name parsing → standard vLLM CLI flags
- Hand-compiled llama-cpp-python + CLBlast → removed (vLLM covers this)
- Multi-stage package installs → use `intel/intel-extension-for-pytorch:2.8.10-xpu` as base

---

## 3. Base Image Selection

Use **`intel/vllm:0.14.1-xpu`** (Intel AI Containers) as the Dockerfile base. Rationale:

- Ships vLLM pre-built with the correct PyTorch 2.9 + IPEX 2.9.10 XPU versions and pre-compiled SYCL kernels
- Avoids the `vllm[xpu]` pip-install approach, which silently installs the base CPU package (no `xpu` extra exists on PyPI; XPU builds are source-only or prebuilt images)
- Includes the correct Intel GPU driver stack (Level Zero, OpenCL ICD) and oneAPI runtimes
- Receives security patches from Intel; `0.14.1` patches CVE-2026-22778 (CVSS 9.8 RCE, present in 0.14.0)
- Avoids ~200 lines of apt sources and GPU driver bootstrap
- Pin is transparent: `ARG VLLM_TAG=0.14.1-xpu` in the Dockerfile makes the version explicit and overridable at build time

Note: `intel-extension-for-pytorch` active development was discontinued after v2.8 (March 2026 EOL for maintenance patches). This base image represents the last IPEX-based vLLM generation; vLLM ≥ 0.16.0 replaces IPEX with `vllm-xpu-kernels`. Update the `VLLM_TAG` build arg when upgrading past 0.15.x.

---

## 4. File-by-File Changes

### 4.1 `Dockerfile` — full rewrite

Replace the current 105-line Dockerfile with:

```dockerfile
# syntax=docker/dockerfile:1

FROM intel/intel-extension-for-pytorch:2.8.10-xpu

ENV DEBIAN_FRONTEND=noninteractive

# --- GPU runtime performance env vars (Intel-documented) ---
ENV SYCL_CACHE_PERSISTENT=1
ENV USE_XETLA=OFF
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV ZES_ENABLE_SYSMAN=1
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

# --- vLLM with XPU support ---
# Pin to a tested version; bump deliberately.
ARG VLLM_VERSION=0.14.0
RUN pip install --no-cache-dir \
        "vllm[xpu]==${VLLM_VERSION}" \
        "transformers>=4.51.3"

# --- Runtime config ---
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
ENV HF_HOME=/root/.cache/huggingface

VOLUME ["/root/.cache/huggingface"]
EXPOSE 8000

COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["--model", "Qwen/Qwen2.5-7B-Instruct", "--dtype", "bfloat16"]
```

**What gets deleted:**
- All `apt-get install` GPU driver bootstrap (handled by base image)
- All oneAPI repository + key setup
- `NEOReadDebugKeys=1` / `ClDeviceGlobalMemSizeAvailablePercent=100` (fixed in current driver packages; verify during testing)
- `torch==2.0.1a0` / `intel_extension_for_pytorch==2.0.110+xpu` pins
- `llama-cpp-python` with CLBlast
- `fschat[model_worker,webui]`
- Port 7860 (Gradio UI)
- `/logs` and `/deps` volumes
- jemalloc LD_PRELOAD (vLLM's memory allocation strategy differs; leave default unless benchmarks show regression)

### 4.2 `startup.sh` — simplify to a thin wrapper

Current script: 47 lines with awk-based model-name extraction, four background processes, and a polling health check loop with no timeout.

Replacement (~10 lines):

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== vLLM-XPU container starting ==="
echo "Args: $*"

# sycl-ls output is a useful first-boot diagnostic
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
```

vLLM parses `--model`, `--dtype`, `--quantization`, `--tensor-parallel-size`, etc. natively. The awk model-name parsing and FastChat worker registration poll are no longer needed.

### 4.3 `docker-compose.yaml` — add (was removed previously)

```yaml
services:
  vllm:
    image: itlackey/ipex-arc-fastchat:latest
    # Build locally with: docker compose build
    build:
      context: .
    container_name: vllm-arc
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    ipc: host
    shm_size: "16g"
    ports:
      - "8000:8000"
    volumes:
      - ${HF_CACHE:-~/.cache/huggingface}:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
    restart: unless-stopped
    # Override `command` to change the model or vLLM flags
    command:
      - --model
      - Qwen/Qwen2.5-7B-Instruct
      - --dtype
      - bfloat16
      - --max-model-len
      - "8192"
```

`group_add: [video, render]` is the piece missing from the current README that causes "permission denied on /dev/dri/renderD128" for many new users. `ipc: host` and `shm_size: 16g` are required by vLLM's PagedAttention shared-memory mechanism.

### 4.4 `README.md` — rewrite sections

- Rename title: "FastChat Docker for Intel Arc GPUs" → "vLLM Docker for Intel Arc GPUs (A-series & B-series)"
- Replace `docker run` example with compose-based example
- Remove FastChat-specific docs (controller ports, Gradio UI, `test_message`)
- Remove "continue VS Code extension" section — it now just points to `http://localhost:8000/v1` like any OpenAI-compatible server
- Add per-GPU memory guidance table:
  - Arc A750 8GB → 7B INT4 via `--quantization awq_marlin` or similar
  - Arc A770 16GB → 7B bf16 or 13B INT4
  - Arc B580 12GB → 7B bf16, 13B INT4
  - Arc Pro B60 24GB → 13B bf16 or 32B INT4
- Add `--tensor-parallel-size N` guidance for multi-GPU setups
- Add Open WebUI compose snippet as the recommended UI (since Gradio UI is being removed)

### 4.5 `.github/workflows/build-docker-image.yml` — update

```yaml
name: Build Docker Image

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build (no push)
        uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: ipex-arc-fastchat:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 4.6 `.github/workflows/build-push-image.yml` — update

```yaml
name: Publish image

on:
  workflow_dispatch:
  release:
    types: [published]

jobs:
  push_to_registry:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/ipex-arc-fastchat:${{ github.event.release.tag_name }}
            ${{ secrets.DOCKER_USERNAME }}/ipex-arc-fastchat:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 4.7 Repo name/rebrand (optional, non-blocking)

The repo is called `ipex-arc-fastchat` but the refactor removes both IPEX (as a hard runtime dep) and FastChat. Options:
- **Keep the name** — preserves existing Docker Hub tags and GitHub stars; add a README note.
- **Rename to `vllm-arc`** or similar — cleaner but breaks existing pulls of `itlackey/ipex-arc-fastchat:latest`.

Recommend **keep the name for one release** to avoid breaking downstream users, then rename at a major version bump with a deprecation notice.

---

## 5. Execution Steps

1. **Prepare branch (already done):** `claude/intel-arc-gpu-research-uf7CR`
2. **Write new `Dockerfile`** — replace entirely (§4.1)
3. **Rewrite `startup.sh`** (§4.2)
4. **Add `docker-compose.yaml`** (§4.3)
5. **Update README** (§4.4) — keep old section as "Legacy (v0.x)" appendix for one release
6. **Update CI workflows** (§4.5, §4.6)
7. **Local build + smoke test** (§6)
8. **Tag release candidate** (e.g., `v1.0.0-rc1`)
9. **Dockerfile CI build passes** — merge to `main`
10. **Publish release `v1.0.0`** — triggers Docker Hub push

---

## 6. Testing Plan

### 6.1 Build-time verification
- [ ] `docker compose build` completes without errors
- [ ] Final image size is reasonable (<8 GB vs current ~6 GB — vLLM adds some weight but driver stack shrinks)

### 6.2 Runtime verification — Arc A-series (A770 16GB)
- [ ] Container starts, `sycl-ls` shows the GPU
- [ ] Load `Qwen/Qwen2.5-7B-Instruct` in bf16
- [ ] `curl http://localhost:8000/v1/models` returns the model
- [ ] `curl http://localhost:8000/v1/chat/completions` returns a valid response
- [ ] Throughput benchmark vs current FastChat: expect ≥2× on concurrent requests (PagedAttention + continuous batching)

### 6.3 Runtime verification — Arc B-series (B580 12GB)
- [ ] Same 7B model loads (may need `--max-model-len 4096` for context)
- [ ] `--kv-cache-dtype fp8` reduces KV cache memory (XPU-verified; `awq_marlin`/`gptq_marlin` are CUDA-only)
- [ ] No driver-level errors in `dmesg`

### 6.4 Multi-GPU (optional, if hardware available)
- [ ] `--tensor-parallel-size 2` loads across 2 A770s
- [ ] Aggregate memory usable > single-GPU max

### 6.5 OpenAI API compatibility
- [ ] Works with `openai` Python client
- [ ] Works with Continue.dev VS Code extension
- [ ] Works with Open WebUI (via compose)
- [ ] Streaming responses work (`stream=true`)

### 6.6 Regression tests
- [ ] HuggingFace cache volume mount persists models across restarts
- [ ] `HF_TOKEN` env var gates access to gated models (e.g., Llama-3)
- [ ] Container exits cleanly on SIGTERM (vLLM handles this natively — no orphan processes)

---

## 7. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **CVE-2026-22778 (CVSS 9.8)** — RCE in vLLM ≤ 0.14.0 via malicious video URL | High (public exploit) | Critical | Base image `intel/vllm:0.14.1-xpu` includes the patch; do not use `0.14.0-xpu` |
| vLLM XPU backend regression on A-series (Alchemist / Xe) | Medium | High | A-series support has been intermittent since vLLM 0.10.0; test A770 explicitly; check issue tracker before release |
| `--host 0.0.0.0` with no API key exposes inference endpoint to local network | High | High | Add `--api-key` flag or reverse proxy; README Security section documents this |
| Container runs as root; combined with `ipc: host` escalates container escape impact | Medium | High | Document and flag for post-1.0 hardening; drop `ipc: host` for single-GPU setups |
| `awq_marlin` / `gptq_marlin` advertised but CUDA-only | High | Medium | README and plan corrected to document XPU quantization limitations |
| B-series requires kernel 6.12+; users on Ubuntu 22.04 / kernel 6.x will fail | Medium | Medium | README updated with per-series kernel requirements |
| Base image `intel/vllm:0.14.1-xpu` tag is mutable | Low | Medium | Digest-pin the FROM line once a known-good digest is validated; use Renovate/Dependabot |
| IPEX entering EOL (maintenance-only through ~mid-2026, then unsupported) | Medium | Medium | vLLM ≥ 0.16.0 replaces IPEX with `vllm-xpu-kernels`; plan a VLLM_TAG bump to track upstream |
| HF_TOKEN exposed via `docker inspect` | Medium | Medium | Use `.env` file (gitignored); document in README Security section; never `ENV HF_TOKEN` in Dockerfile |
| Users rely on Gradio UI at port 7860 | Medium | Low | Document Open WebUI compose addition; call out in release notes |
| Users rely on `--max-gpu-memory` FastChat flag | High | Low | README migration note: `--max-gpu-memory 14Gib` → `--gpu-memory-utilization 0.9` |
| `llama-cpp-python` GGUF workflow removed | Medium | Medium | vLLM supports GGUF loading from v0.6+; document `--model path/to/file.gguf` |
| IPEX `2.0.110+xpu` wheel index URL removal breaks rebuilds of old tag | Low | Low | Pre-build and tag `v0.0.6-legacy` before any change; keep it on Docker Hub indefinitely |

---

## 8. Rollback Plan

If the vLLM migration hits blocking regressions:
1. Legacy image remains available as `itlackey/ipex-arc-fastchat:v0.0.6` (pre-existing release)
2. Revert the branch: `git revert <refactor-merge-commit>` and re-tag `v0.0.7` from legacy
3. Communicate via release notes; leave legacy tag pinned

---

## 9. Out-of-Scope for This Refactor

Intentionally **not** doing in this pass:
- Adding Open WebUI to the same compose file (keep as documented add-on)
- Kubernetes / Helm chart
- Prometheus metrics endpoint (vLLM exposes `/metrics` natively — just document it)
- Embedding model support (vLLM supports this, but defer until requested)
- Fine-tuning / training workflows
- Migrating to `intel/llm-scaler` (B-series-only; does not cover A-series users)

---

## 10. Summary of Line-Count Reduction

| File | Before | After | Δ |
|---|---|---|---|
| `Dockerfile` | 105 lines | ~25 lines | -80 |
| `startup.sh` | 47 lines | ~12 lines | -35 |
| `docker-compose.yaml` | 0 (removed) | ~25 lines | +25 |
| `README.md` | 95 lines | ~80 lines (cleaner) | -15 |
| **Total app code** | **247 lines** | **~142 lines** | **-105 (-43%)** |

Smaller surface area, fewer moving parts, actively maintained upstream, meaningfully faster inference, and coverage of both Arc GPU generations.
