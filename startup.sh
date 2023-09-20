#!/bin/bash

# Get the model name
MODEL_NAME=$1

# Start the controller
python3 -m fastchat.serve.controller --host 0.0.0.0 &
sleep 2

# Start the model worker
python3 -m fastchat.serve.model_worker --model-path $MODEL_NAME --device xpu --host 0.0.0.0 --max-gpu-memory '14Gib' &
sleep 5
# Health check for controller using a test message
while true; do
    if python3 -m fastchat.serve.test_message --model-name $MODEL_NAME; then
        break
    fi
    sleep 5  # wait for 5 seconds before the next attempt
done

# Start the web server
# Replace this with the actual command to start your web server
python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --model-list-mode 'reload' &

# Start the OpenAI API
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000



