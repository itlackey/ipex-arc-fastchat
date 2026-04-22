# vLLM Docker for Intel Arc GPUs (A-series & B-series)

An OpenAI-compatible LLM inference server for Intel Arc GPUs, powered by [vLLM](https://github.com/vllm-project/vllm) with the native Intel XPU backend. Supports both the A-series (A770, A750) and the B-series (B580, Arc Pro B60/B70).

> **Upcoming rename (next release):** This image will be republished as `itlackey/vllm-arc` to reflect the move away from FastChat and IPEX-LLM. The `itlackey/ipex-arc-fastchat` tag will continue to be published for one additional release and then stop receiving updates. Pin to a specific version tag if you need stability across the rename.

## What changed in this release

- **Serving framework:** FastChat → vLLM (PagedAttention, continuous batching, INT4/FP8 quantization)
- **GPU stack:** PyTorch 2.0.1a0 + IPEX 2.0.110 (2023, EOL) → PyTorch 2.8 + native XPU device
- **Base image:** Custom Ubuntu + oneAPI build → `intel/intel-extension-for-pytorch:2.8.10-xpu`
- **GPU series:** A-series only → A-series **and** B-series
- **Processes per container:** 4 (FastChat controller + worker + gradio + openai) → 1 (vLLM)
- **Quantization:** none → INT4 / INT8 / FP8 / AWQ / GPTQ via vLLM flags
- **Gradio web UI:** removed. Pair with [Open WebUI](https://github.com/open-webui/open-webui) for a browser interface.

## Requirements

- Intel Arc GPU (A-series or B-series) with a current Linux kernel (6.2+) and the Intel GPU userspace drivers installed on the host
- Docker with access to `/dev/dri`
- The host user (or the container runtime) must be in the `video` and `render` groups

## Quick start with Docker Compose

```sh
docker compose up -d
```

This loads `Qwen/Qwen2.5-7B-Instruct` in bfloat16 with an 8K context on port 8000. Override the model by editing `docker-compose.yaml` or by running:

```sh
MODEL=meta-llama/Llama-3.1-8B-Instruct docker compose run --rm --service-ports vllm \
    --model $MODEL --dtype bfloat16 --max-model-len 8192
```

Gated models (e.g. Llama-3) need a HuggingFace token:

```sh
HF_TOKEN=hf_xxx docker compose up -d
```

## Quick start with `docker run`

```sh
docker run -d \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=16g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest \
    --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

`--ipc=host` and `--shm-size=16g` are required by vLLM's PagedAttention shared-memory mechanism. `--group-add video --group-add render` is required on most Linux distributions for the container to use `/dev/dri/renderD128`.

## Using the API

vLLM exposes an OpenAI-compatible API at `http://localhost:8000/v1`. It works with any OpenAI client unchanged — just point `base_url` at the container and use `EMPTY` (or anything) as the API key.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Other useful endpoints:
- `GET /v1/models` — list the loaded model
- `POST /v1/completions` — legacy completions
- `POST /v1/embeddings` — if the loaded model supports embeddings
- `GET /metrics` — Prometheus metrics (tokens/sec, queue depth, KV cache usage, etc.)

## GPU memory sizing

| GPU | VRAM | Recommended model sizes |
|---|---|---|
| Arc A750 | 8 GB | 7B with `--quantization awq_marlin` (INT4) |
| Arc A770 | 16 GB | 7B bf16, or 13B INT4 |
| Arc B580 | 12 GB | 7B bf16 (shorter context), 13B INT4 |
| Arc Pro B60 | 24 GB | 13B bf16, 32B INT4 |
| Arc Pro B70 | 32 GB | 32B bf16, 70B INT4 |

Key vLLM flags for tuning:
- `--gpu-memory-utilization 0.9` — fraction of VRAM vLLM may use (default 0.9)
- `--max-model-len N` — maximum context length; reduce to fit larger batch sizes
- `--quantization awq_marlin` (or `gptq_marlin`, `fp8`) — load quantized weights
- `--tensor-parallel-size N` — shard across multiple GPUs
- `--dtype bfloat16` (preferred) or `float16`

Run `python3 -m vllm.entrypoints.openai.api_server --help` inside the container for the full flag list.

## Multi-GPU

To shard across two Arc GPUs:

```sh
docker run -d \
    --device /dev/dri \
    --group-add video --group-add render \
    --ipc=host --shm-size=16g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2
```

## Adding a web UI

The container no longer ships a Gradio UI. The recommended pairing is [Open WebUI](https://github.com/open-webui/open-webui):

```yaml
# add to docker-compose.yaml
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://vllm:8000/v1
      - OPENAI_API_KEY=EMPTY
    depends_on:
      - vllm
```

## Using with editor integrations

Any tool that accepts an OpenAI-compatible endpoint will work. Point it at `http://localhost:8000/v1` with any placeholder API key. Examples: Continue.dev, Cursor (custom endpoint), Zed, aider.

## Development

Build locally:

```sh
docker compose build
# or
docker build -t itlackey/ipex-arc-fastchat:dev .
```

Override the pinned vLLM version at build time:

```sh
docker build --build-arg VLLM_VERSION=0.14.0 -t itlackey/ipex-arc-fastchat:dev .
```

## Acknowledgments

This project originally built on work by [Nuullll](https://github.com/Nuullll) and their [ipex-sd-docker-for-arc-gpu](https://github.com/Nuullll/ipex-sd-docker-for-arc-gpu) project for getting Arc GPUs working inside a container. The current release replaces the FastChat + IPEX-LLM stack (both archived by Intel in early 2026) with vLLM on top of native PyTorch XPU.
