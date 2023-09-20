#!/bin/bash

# Start the controller
python3 -m fastchat.serve.controller &

# Health check for controller using a test message
while true; do
    if python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.3; then
        break
    fi
    sleep 5  # wait for 5 seconds before the next attempt
done

# Start the model worker
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 &

# Health check for model worker using curl
while true; do
    # replace 'localhost:port' with the actual URL of the model worker
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}\n" 'localhost:port')
    if [ "$HTTP_CODE" -eq 200 ]; then
        break
    fi
    sleep 5  # wait for 5 seconds before the next attempt
done

# Start the OpenAI API
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

# Start the web server
# Replace this with the actual command to start your web server
python3 -m fastchat.serve.gradio_web_server &

# #!/bin/bash

# # # source /opt/intel/oneapi/setvars.sh
# # #git clone https://huggingface.co/lmsys/vicuna-7b-v1.3 /models/lmsys/vicuna-7b-v1.3
# # #python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device xpu
# # #git clone https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g /models/vicuna-7B-1.1-GPTQ-4bit-128g
# # #git clone https://huggingface.co/$FC_MODEL_PATH /models/$FC_MODEL_PATH

# #python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device xpu
# python3 -m fastchat.serve.controller --host 0.0.0.0 
# python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu --host 0.0.0.0 --max-gpu-memory '14Gib'
# python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --share
# python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000

# # #python3 -m fastchat.serve.huggingface_api  --model-path lmsys/vicuna-7b-v1.3 --device xpu --max-gpu-memory '14Gib'

