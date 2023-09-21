#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION} AS oneapi-lib-installer

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    gnupg2 \
    gpg-agent \
    unzip \
    wget

# oneAPI packages
RUN no_proxy=$no_proxy wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
   | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
   | tee /etc/apt/sources.list.d/oneAPI.list

ARG DPCPP_VER=2023.2.1-16
ARG MKL_VER=2023.2.0-49495
# intel-oneapi-compiler-shared-common provides `sycl-ls`
ARG CMPLR_COMMON_VER=2023.2.1
# Install runtime libs to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-oneapi-runtime-dpcpp-cpp=${DPCPP_VER} \
    intel-oneapi-runtime-mkl=${MKL_VER} \
    intel-oneapi-compiler-shared-common-${CMPLR_COMMON_VER}=${DPCPP_VER}

# Prepare Intel Graphics driver index
ARG DEVICE=flex
RUN no_proxy=$no_proxy wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy %s\n' "$DEVICE" | \
    tee /etc/apt/sources.list.d/intel.gpu.jammy.list

ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION}

RUN mkdir /oneapi-lib
COPY --from=oneapi-lib-installer /opt/intel/oneapi/lib /oneapi-lib/
ARG CMPLR_COMMON_VER=2023.2.1
COPY --from=oneapi-lib-installer /opt/intel/oneapi/compiler/${CMPLR_COMMON_VER}/linux/bin/sycl-ls /bin/
COPY --from=oneapi-lib-installer /usr/share/keyrings/intel-graphics.gpg /usr/share/keyrings/intel-graphics.gpg
COPY --from=oneapi-lib-installer /etc/apt/sources.list.d/intel.gpu.jammy.list /etc/apt/sources.list.d/intel.gpu.jammy.list

# Set oneAPI lib env
ENV LD_LIBRARY_PATH=/oneapi-lib:/oneapi-lib/intel64:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

ARG PYTHON=python3.10
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} lib${PYTHON} python3-pip && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

ARG ICD_VER=23.17.26241.33-647~22.04
ARG LEVEL_ZERO_GPU_VER=1.3.26241.33-647~22.04
ARG LEVEL_ZERO_VER=1.11.0-647~22.04
ARG LEVEL_ZERO_DEV_VER=1.11.0-647~22.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-opencl-icd=${ICD_VER} \
    intel-level-zero-gpu=${LEVEL_ZERO_GPU_VER} \
    level-zero=${LEVEL_ZERO_VER} \
    level-zero-dev=${LEVEL_ZERO_DEV_VER} && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

# Stable Diffusion Web UI dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjemalloc-dev \
    python3-venv \
    git \
    numactl && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

ENV venv_dir=/deps/venv
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

# Force 100% available VRAM size for compute-runtime
# See https://github.com/intel/compute-runtime/issues/586
ENV NEOReadDebugKeys=1
ENV ClDeviceGlobalMemSizeAvailablePercent=100

# llama.cpp dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    git-lfs \
    curl \
    opencl-headers \
    clblast-utils

# Install Torch
RUN python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Install llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python  --force-reinstall --upgrade --no-cache-dir

# Install fschat
RUN pip3 install "fschat[model_worker,webui]"


COPY startup.sh /bin/start_fastchat.sh
RUN chmod 755 /bin/start_fastchat.sh

VOLUME [ "/deps" ]
VOLUME [ "/apps" ]
VOLUME [ "/root/.cache/huggingface" ]
WORKDIR /apps/fastchat/logs

EXPOSE 7860
EXPOSE 8000
ENV FS_ENABLE_WEB=true
ENV FS_ENABLE_OPENAI_API=true
ENV FS_ENABLE_HF_API=false

ENTRYPOINT [ "/bin/bash", "/bin/start_fastchat.sh" ]
CMD ["--model-path lmsys/vicuna-7b-v1.3 --max-gpu-memory 14Gib"]
