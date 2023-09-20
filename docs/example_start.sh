#!/bin/bash

#python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device xpu
python3 -m fastchat.serve.controller --host 0.0.0.0 
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu --host 0.0.0.0 --max-gpu-memory '14Gib'
python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --share
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
# #python3 -m fastchat.serve.huggingface_api  --model-path lmsys/vicuna-7b-v1.3 --device xpu --max-gpu-memory '14Gib'
