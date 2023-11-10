# FastChat Docker for Intel Arc GPUs
 
This project provides a Docker container that can be used to host a [FastChat](https://github.com/lm-sys/FastChat) web server and OpenAI API. This project is based heavily on the work done by [Nuullll](https://github.com/Nuullll) and their [ipex-sd-docker-for-arc-gpu](https://github.com/Nuullll/ipex-sd-docker-for-arc-gpu) project. Thank you to them for doing the heavy lifting of getting the Arc GPU working in a docker container.

## Running the container

Spinning up the web server is as simple as using the `docker run` command but it requires a few cli arguments to be specified. 

- `--device /dev/dri`: is needed to enable access to the GPU hardware
- `-p 7860:7860`: Expose the port for the web UI
- `-p 8000:8000`: Expose the port for the OpenAI API
- `-v ~/local/path:/remote/path`: Multiple can be mounted to a specified folder.
    - **Recommended** `-v ~/ai/huggingface:/root/.cache/huggingface`: Mount a volume to a local folder that contains the models downloaded from hugging face. *It is highly recommended to set this volume to a local folder you want to store the large models.* This also prevents the need to re-download the model for different containers and between restarts.
    -  **Optional** `-v ~/ai/fastchat/logs:/logs`: Provided to map to a local folder to store fast chat logs. By default the container will write logs to the `/logs` folder. We recommend mapping this to a local folder that is easy to access. This will help in troubleshooting issues with the FastChat services.   
- `itlackey/ipex-arc-fastchat:latest`: The latest published version of the docker image 

```sh
## Run in the background with default FastChat worker settings 
docker run -d \
    --device /dev/dri \
    -v ~/ai/models/huggingface:/root/.cache/huggingface \
    -p 7860:7860 \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest
```

This will start a container on port 7860, and you can access the FastChat web server by visiting http://localhost:7860 in your web browser. To access the OpenAI API, you can use the http://localhost:8000/v1 url and whatever client you need.

**Please note** that the default settings for the container are to run the `lmsys/vicuna-7b-v1.3` with 14Gib of VRAM allocated. You can change these settings by providing CLI arguments to the FastChat worker.

### Worker configuration

Additional parameters can be provided after the docker image name that will be passed to the call to `fastchat.serve.model_worker`. This allows you to specify the model to load when starting the container. It also allows you to customize the settings for the FastChat worker. 

Here is an example of running CodeLlama-7b-Instruct-hf model on a single A770 with 14Gib.

```sh
## Run in the background
docker run -d \
    --device /dev/dri \
    -v ~/ai/models/huggingface:/root/.cache/huggingface \
    -v ~/ai/fastchat/logs:/logs \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest \
    --model-path codellama/CodeLlama-7b-Instruct-hf --max-gpu-memory 14Gib
```
There are several arguments that can be passed to the model worker. Check the FastChat [documentation](https://github.com/lm-sys/FastChat#single-gpu) or run `python3 -m fastchat.serve.model_worker --help` on the container to see a list of options.

The most notable options are to adjust the max gpu memory (for A750 `--max-gpu-memory 7Gib`) and the number of GPUs (for multiple GPUs `--num-gpus 2`). Check the FastChat documentation to find more detailed information. 

## Using the model

There are several ways to utilize the model that is now running in the docker container. Below is a list of ways to interact with the model once the container is up and running.

### Using the web browser

Once the container is up and running, you should be able to navigate to http://localhost:7860 in your browser and see a basic chat interface. Here you can have a conversation with the model, ask it to generate code, answer questions, etc. This works similar to ChatGPT and other online LLM chat services.

### Using the continue VSCode extension

To use the container as a coding assistant like GitHub Co-Pilot, you can install the [continue extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue) for VS Code. Once the extension is installed, you will need to update your configuration to point to the container instead of the OpenAI API endpoint.  More information can be found in [continue's docs](https://continue.dev/docs/walkthroughs/codellama#fastchat-api), but here is a quick overview.

Open your `~/.continue/config.py` file and make these changes.

Include this line of code at the top of the file.

```python
from continuedev.src.continuedev.libs.llm.openai import OpenAI
```

Then switch the models parameter to specify the model you are running and the FastChat API endpoint. 

```python

    models=Models(default=OpenAI(
        model="CodeLlama-7b-Instruct-hf",
        api_base='http://localhost:8000/v1',
        api_key="EMPTY")),

```
**NOTE:** The model should be set to whatever you specified when starting the container. This example assumes the container was started with the `--model-path codellama/CodeLlama-7b-Instruct-hf` argument specified. If you started the container with the default settings, the model should be set to `vicuna-7b-v1.3`.



## Development

To get started, you will need to clone this repository.

You can build your modified image by using the following command:

`docker build -f Dockerfile -t [your-username]/ipex-arc-fastchat:latest .`



