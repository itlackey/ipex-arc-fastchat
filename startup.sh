
#!/bin/bash

# Find the value for --model-path using sed
model_path=$(echo "$@" | awk -F'--model-path ' '{print $2}' | awk '{print $1}')

# Extract the model name after the "/" character
short_model_name=${model_path#*/}

# Print args
echo "Model path: $model_path"
echo "Model name: $short_model_name"
echo "Worker args: $@"
echo "Enable web: $FS_ENABLE_WEB"
echo "Enable OpenAI API: $FS_ENABLE_OPENAI_API"

# Start the controller
python3 -m fastchat.serve.controller --host 0.0.0.0 &

# Start the model worker
python3 -m fastchat.serve.model_worker --device xpu --host 0.0.0.0 $@ &

# Health check for controller using a test message
while true; do
  response=$(python3 -m fastchat.serve.test_message --model-name $short_model_name)
  if echo "$response" | grep -q "worker_addr: http://localhost:21002"; then
    echo "Model registered spinning up services..."
    break
  else
    echo "Waiting for model..."
  fi
  sleep 3  # wait before the next attempt
done


# Check to see if the web server should be enabled
if [[ "${FS_ENABLE_WEB}" == "true" ]]; then
  # Start the web server
  echo "Enabling web server..."
  python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --model-list-mode 'reload' &
fi

if [[ "${FS_ENABLE_OPENAI_API}" == "true" ]]; then
    # Start the OpenAI API
    echo "Enabling OpenAI API server..."
    python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
fi