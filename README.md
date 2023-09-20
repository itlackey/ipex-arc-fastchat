# FastChat Docker for Intel Arc GPUs
 
This project provides a Docker container that can be used to host a FastChat web server and OpenAI API. The FastChat web server is a rapidly growing platform with a growing number of users. The OpenAI API is a powerful and flexible API that provides access to a variety of AI technologies and tools.

## Running the container

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
    itlackey/ipex-arc-fastchat:latest

```

## Development

To get started, you will need to clone this repository and build a Docker image. You can do this using the following command:

docker build -t ipex-arc-fastchat .
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
    ipex-arc-fastchat:latest
```


This will start a container on port 8080, and you can access the FastChat web server by visiting http://localhost:8080 in your web browser. To access the OpenAI API, you can use the API documentation provided by OpenAI.

For more information, please refer to the README.md file in the root of the repository. Thank you for using the FastChat Docker Container!