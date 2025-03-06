FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=America/Los_Angeles

# Install basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    g++ \
    git \
    htop \
    imagemagick \
    libegl1 \
    libegl1-mesa-dev \
    libgl1 \
    libgl1-mesa-dev \
    libgles2 \
    libgles2-mesa-dev \
    libglvnd-dev \
    libglvnd0 \
    libglx0 \
    libnss3-dev \
    libopenexr-dev \
    libx264-dev \
    ninja-build \
    sox \
    tmux \
    unzip \
    vim \
    wget

# Update pip
RUN pip install --upgrade pip

# Install torchaudio
# WORKDIR /src
# RUN git clone https://github.com/pytorch/audio && cd audio && python setup.py develop

# Install Flash-Attention
# RUN git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention && MAX_JOBS=4 python setup.py install

# Install NeMo SDP
WORKDIR /src
RUN git clone https://github.com/NVIDIA/NeMo-speech-data-processor.git && cd NeMo-speech-data-processor && pip install -r requirements/main.txt
RUN pip install transformers accelerate ndjson torchaudio pyannote-audio jupyter notebook
RUN pip install flash-attn --no-build-isolation
RUN pip install https://github.com/LahiLuk/YouTokenToMe/archive/master.zip
RUN pip install git+https://github.com/m-bain/whisperx.git --upgrade
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN pip install megatron-core transformer_engine[pytorch]

## Update NeMo to allow timestamps
RUN git clone https://github.com/NVIDIA/NeMo.git /src/NeMo/
WORKDIR /src/NeMo
RUN git reset --hard 0547550ad803fce1e4a019f92e9c59f4c902e7e0
RUN ./reinstall.sh
RUN pip install python-swiftclient ffmpeg-python

WORKDIR /src/NeMo-speech-data-processor
