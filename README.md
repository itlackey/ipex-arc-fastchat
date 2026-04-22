# vLLM Docker for Intel Arc GPUs (A-series & B-series)

An OpenAI-compatible LLM inference server for Intel Arc GPUs, powered by [vLLM](https://github.com/vllm-project/vllm) with the native Intel XPU backend. Supports both the A-series (A770, A750) and the B-series (B580, Arc Pro B60/B70).

> **Upcoming rename (next release):** This image will be republished as `itlackey/vllm-arc` to reflect the move away from FastChat and IPEX-LLM. The `itlackey/ipex-arc-fastchat` tag will continue to be published for one additional release and then stop receiving updates. Pin to a specific version tag if you need stability across the rename.

## What changed in this release

- **Serving framework:** FastChat → vLLM (PagedAttention, continuous batching, INT4/FP8 quantization)
- **GPU stack:** PyTorch 2.0.1a0 + IPEX 2.0.110 (2023, EOL) → PyTorch 2.9 + native XPU device
- **Base image:** Custom Ubuntu + oneAPI build → `intel/vllm:0.14.1-xpu` (Intel's pre-built XPU image)
- **GPU series:** A-series only → A-series **and** B-series
- **Processes per container:** 4 (FastChat controller + worker + gradio + openai) → 1 (vLLM)
- **Quantization:** none → INT4 / INT8 / FP8 / AWQ / GPTQ via vLLM flags
- **Gradio web UI:** removed. Pair with [Open WebUI](https://github.com/open-webui/open-webui) for a browser interface.

## Requirements

- Intel Arc GPU with the Intel GPU userspace drivers installed on the host:
  - A-series (A770, A750): Linux kernel **6.2 or newer**
  - B-series (B580, Arc Pro B60/B70): Linux kernel **6.12 or newer** — Ubuntu 24.04 LTS with the HWE kernel stack is recommended
- Docker with access to `/dev/dri`
- The host user (or the container runtime) must be in the `video` and `render` groups

**Host driver setup** — see [dgpu-docs.intel.com](https://dgpu-docs.intel.com/driver/client/overview.html) for full instructions. Minimum packages on Ubuntu:

```sh
sudo apt-get install -y intel-opencl-icd libze-intel-gpu1 intel-level-zero-gpu
sudo usermod -aG render,video $USER   # log out and back in after this
```

> **A-series (A770/A750) compatibility note:** Some vLLM versions ≥ 0.10.0 have reported issues with Alchemist (Xe) GPU support due to changes in the attention backend. If inference fails on A-series hardware, check the [vLLM GitHub issue tracker](https://github.com/vllm-project/vllm/issues) for current A-series XPU status before filing a new issue.

## Quick start with Docker Compose

```sh
docker compose up -d
```

This loads `Qwen/Qwen3-4B` in bfloat16 (float16 on A-series) with an 8K context on port 8000. Qwen3-4B fits comfortably on A770 16 GB (~8 GB weights, ~6 GB free for KV cache) and matches Qwen2.5-7B-Instruct quality. Override the model by editing `docker-compose.yaml` or by running:

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
    --shm-size=16g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=${HF_TOKEN:-} \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest \
    --model Qwen/Qwen3-4B --dtype bfloat16 --max-model-len 8192
```

`--shm-size=16g` is required by vLLM's shared-memory IPC mechanism for multi-process workers. `--group-add video --group-add render` is required on most Linux distributions for the container to use `/dev/dri/renderD128`. For multi-GPU tensor-parallel setups, add `--ipc=host` to share the full host IPC namespace across GPU workers.

## Using the API

vLLM exposes an OpenAI-compatible API at `http://localhost:8000/v1`. It works with any OpenAI client unchanged — just point `base_url` at the container and use `EMPTY` (or anything) as the API key.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
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
| Arc A750 | 8 GB | 7B with INT4 quantization (see Quantization note) |
| Arc A770 *(16 GB SKU)* | 16 GB | 7B fp16, or 13B INT4 |
| Arc A770 *(8 GB SKU)* | 8 GB | 7B INT4 only |
| Arc B580 | 12 GB | 7B bf16 (shorter context), 13B INT4 |
| Arc Pro B60 | 24 GB | 13B bf16, 32B INT4 |
| Arc Pro B70 | 32 GB | 32B bf16, 70B INT4 |

> **A-series note:** `--dtype bfloat16` silently falls back to `float16` on Alchemist (A-series) GPUs — this is expected vLLM behavior. Both precisions use the same VRAM; only the log output differs. Use `--dtype float16` explicitly on A-series to avoid the warning.
>
> **Quantization note:** `awq_marlin` and `gptq_marlin` rely on CUDA-specific Marlin kernels and are **not supported on Intel XPU**. For KV-cache compression, use `--kv-cache-dtype fp8` (verified XPU support in 0.14.x). Check the [vLLM XPU quantization docs](https://docs.vllm.ai/en/stable/features/quantization/) for currently supported weight quantization formats on XPU.

Key vLLM flags for tuning:
- `--gpu-memory-utilization 0.9` — fraction of VRAM vLLM may use (default 0.9)
- `--max-model-len N` — maximum context length; reduce to fit larger batch sizes
- `--kv-cache-dtype fp8` — FP8 KV-cache compression (XPU-verified)
- `--tensor-parallel-size N` — shard across multiple GPUs
- `--dtype bfloat16` (B-series) or `--dtype float16` (A-series)

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
    --model Qwen/Qwen3-32B \
    --dtype bfloat16 \
    --tensor-parallel-size 2
```

## Adding a web UI

The container no longer ships a Gradio UI. The recommended pairing is [Open WebUI](https://github.com/open-webui/open-webui):

```yaml
# add to docker-compose.yaml
  open-webui:
    image: ghcr.io/open-webui/open-webui:latest
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

Override the vLLM XPU base image tag at build time:

```sh
docker build --build-arg VLLM_TAG=0.14.1-xpu -t itlackey/ipex-arc-fastchat:dev .
```

## Security

vLLM's OpenAI-compatible server runs with **no authentication by default**. Anyone who can reach port 8000 can query the API, enumerate loaded models, and access the `/metrics` endpoint. On a shared or networked machine:

- Add `--api-key <random-token>` to your run command and pass the same token as `api_key=` in your client.
- Or restrict port 8000 to localhost with a reverse proxy (nginx, Caddy) and handle auth there.
- The `/metrics` Prometheus endpoint is also unauthenticated — restrict it to scraper IPs at the network layer.
- Never set `HF_TOKEN` via `ENV` in a derived Dockerfile — it bakes the token permanently into image layers. Use a `.env` file (see `.env.example`) or Docker secrets instead.

## Acknowledgments

This project originally built on work by [Nuullll](https://github.com/Nuullll) and their [ipex-sd-docker-for-arc-gpu](https://github.com/Nuullll/ipex-sd-docker-for-arc-gpu) project for getting Arc GPUs working inside a container. The current release replaces the FastChat + IPEX-LLM stack (both archived by Intel in early 2026) with vLLM on top of native PyTorch XPU.
