# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app

# System libs for pygame/opencv headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first so we can swap CPU/GPU builds via build-args
ARG TORCH_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements-cloud.txt ./requirements-cloud.txt
RUN python3 -m pip install --no-cache-dir -r requirements-cloud.txt \
    && python3 -m pip install --no-cache-dir tensorboard==2.16.2

COPY . .

# Default to fast headless training; override in docker run if you want rendering
CMD ["python3", "train_ppo_good.py", "--no_render"]
