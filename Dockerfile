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

FROM ubuntu:${UBUNTU_VERSION} AS fastchat-xpu


ARG PYTHON=python3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    gnupg2 \
    gpg-agent \
    unzip \
    wget \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjemalloc-dev \
    git \
    git-lfs \
    curl \
    opencl-headers \
    clblast-utils \
    numactl \
    python3 libpython3.11 python3-pip python3-venv
    

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip setuptools

# Force 100% available VRAM size for compute-runtime
# See https://github.com/intel/compute-runtime/issues/586
ENV NEOReadDebugKeys=1
ENV ClDeviceGlobalMemSizeAvailablePercent=100


# oneAPI packages
RUN no_proxy=$no_proxy wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
   | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
   | tee /etc/apt/sources.list.d/oneAPI.list

# Intel driver index
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
    | tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Install Intel packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    level-zero-dev intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl intel-oneapi-compiler-shared-common-2023.2.1 \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo  && \
    apt-get clean && \    
    rm -rf  /var/lib/apt/lists/*


ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

# Install Torch
RUN pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Install llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python  --force-reinstall --upgrade --no-cache-dir

# Install fschat
RUN pip install "fschat[model_worker,webui]"

VOLUME [ "/deps" ]
VOLUME [ "/logs" ]
VOLUME [ "/root/.cache/huggingface" ]
RUN mkdir /logs
WORKDIR /logs

EXPOSE 7860
EXPOSE 8000
ENV FS_ENABLE_WEB=true
ENV FS_ENABLE_OPENAI_API=true
ENV LOGDIR=/logs

COPY startup.sh /bin/start_fastchat.sh
RUN chmod 755 /bin/start_fastchat.sh
ENTRYPOINT [ "/bin/bash", "/bin/start_fastchat.sh" ]
CMD ["--model-path lmsys/vicuna-7b-v1.3 --max-gpu-memory 14Gib"]
