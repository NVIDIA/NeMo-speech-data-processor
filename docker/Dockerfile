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

#install typing-ext manually
RUN pip install typing-extensions

# Clone the NeMo SDP repository
COPY . /src/NeMo-speech-data-processor
RUN rm -rf /src/NeMo-speech-data-processor/.git


WORKDIR /src/NeMo-speech-data-processor
#need to install numpy before reqs, even thougth it present in reqs (cause it requred to install [python-sox], otherwise we face an error)
RUN pip install numpy 
RUN find requirements/ -name "*.txt" -exec pip install -r {} \;
# Set working directory back to NeMo-speech-data-processor
WORKDIR /src/NeMo-speech-data-processor

# Set up entrypoint
CMD ["bash"]
