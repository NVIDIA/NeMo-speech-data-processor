FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG SOURCE
COPY ${SOURCE:-.} /workspace/NeMo-speech-data-processor
RUN rm -rf /workspace/NeMo-speech-data-processor/docker/

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Install basics
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python-is-python3 \
    build-essential \
    curl \
    ffmpeg \
    git \
    sox \
    libsox-fmt-mp3 \
    unzip \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

#need to install numpy before reqs, even thougth it present in reqs (cause it requred to install [python-sox], otherwise we face an error)
WORKDIR /workspace/NeMo-speech-data-processor
RUN pwd && ls -al && ls -R
RUN pip install numpy 
RUN find requirements/ -name "*.txt" -exec pip install -r {} \;
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


# Update NeMo to allow timestamps
RUN git clone https://github.com/NVIDIA/NeMo.git /src/NeMo/
WORKDIR /src/NeMo
RUN git reset --hard 0547550ad803fce1e4a019f92e9c59f4c902e7e0
RUN ./reinstall.sh
RUN pip install python-swiftclient ffmpeg-python

# Set working directory back to NeMo-speech-data-processor
WORKDIR /src/NeMo-speech-data-processor

# Set up entrypoint
CMD ["bash"]