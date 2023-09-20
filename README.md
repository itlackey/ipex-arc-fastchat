# FastChat Docker for Intel Arc GPUs
 
This project provides a Docker container that can be used to host a [FastChat](https://github.com/lm-sys/FastChat) web server and OpenAI API. This project is based heavily on the work done by [Nuullll](https://github.com/Nuullll) and their [ipex-sd-docker-for-arc-gpu](https://github.com/Nuullll/ipex-sd-docker-for-arc-gpu) project. Thank you to them for doing the heavy lifting of getting the Arc GPU working in a docker container.

## Running the container

Running the container requires a few cli arguments to be specified. 

- `--device /dev/dri`: is needed to enable access to the GPU hardware
- `-v ~/ai/huggingface:/root/.cache/huggingface`: Multiple volumes need to be mounted
    - apps: a local folder to store applications and fast chat logging
    - deps: a place to store the python dependencies
    - huggingface: contains the models downloaded from hugging face
- `-p 7860:7860`: Several ports need to be exposed from docker to allow access to the web server and API
- `itlackey/ipex-arc-fastchat:latest`: The docker image name
- `codellama/CodeLlama-7b-Instruct-hf`: The hugging face model id. This can be changed to whatever compatible model available on hugging face.

Here is an example of running CodeLlama-7b-Instruct-hf model.

```sh
docker run -d \
    --device /dev/dri \
    -v ~/ai/apps:/apps \
    -v ~/ai/deps:/deps \
    -v ~/ai/huggingface:/root/.cache/huggingface \
    -p 7860:7860 \
    -p 21001:21001 \
    -p 21002:21002 \
    -p 8000:8000 \
    itlackey/ipex-arc-fastchat:latest codellama/CodeLlama-7b-Instruct-hf

```

This will start a container on port 7860, and you can access the FastChat web server by visiting http://localhost:7860 in your web browser. To access the OpenAI API, you can use the http://localhost:8000/v1 url and whatever client you need.


### Using the continue VSCode extension

Include this at the top of your config.py:

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

## Development

To get started, you will need to clone this repository and build a Docker image. You can do this using the following command:

`docker build -f Dockerfile -t itlackey ipex-arc-fastchat:latest .`

Once the image has been built, you can run a container using the following command:

```sh
docker run -it \
    --device /dev/dri \
    -v ~/ai/apps:/apps \
    -v ~/ai/deps:/deps \
    -v ~/ai/huggingface:/root/.cache/huggingface \
    -p 7860:7860 \
    -p 21001:21001 \
    -p 21002:21002 \
    -p 8000:8000 \
    ipex-arc-fastchat:latest codellama/CodeLlama-7b-Instruct-hf
```
