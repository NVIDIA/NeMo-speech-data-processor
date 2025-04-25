FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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

# Install NeMo SDP
WORKDIR /src
RUN git clone https://github.com/NVIDIA/NeMo-speech-data-processor.git && cd NeMo-speech-data-processor && git checkout 29b9b1ec0ceaf3ffa441c1d01297371b3f8e11d2 && pip install -r requirements/main.txt
RUN pip install transformers accelerate ndjson torchaudio pyannote-audio jupyter notebook
RUN pip install flash-attn --no-build-isolation
RUN pip install https://github.com/LahiLuk/YouTokenToMe/archive/master.zip
RUN pip install whisperx
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN pip install megatron-core transformer_engine[pytorch] pyarabic

RUN pip install nemo_toolkit['all']==2.1.0 
RUN pip install python-swiftclient ffmpeg-python

# Set working directory back to NeMo-speech-data-processor
WORKDIR /src/NeMo-speech-data-processor

# Set up entrypoint
CMD ["bash"]