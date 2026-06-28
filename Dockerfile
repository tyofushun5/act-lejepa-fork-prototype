ARG CUDA_IMAGE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_DATASETS_CACHE=/cache/huggingface/datasets \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=offline \
    WANDB_DIR=/workspace/act-jepa/wandb \
    WANDB_CACHE_DIR=/workspace/act-jepa/.cache/wandb \
    WANDB_SILENT=true \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /workspace/act-jepa

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libglfw3 \
    libosmesa6 \
    libsm6 \
    libvulkan1 \
    libxext6 \
    libxrender1 \
    mesa-vulkan-drivers \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --index-url "${PYTORCH_INDEX_URL}" \
    torch==2.12.1 \
    torchvision==0.27.1 \
    torchcodec==0.14.0

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY . .

RUN mkdir -p /cache/huggingface logs wandb .cache/wandb

# Keep bare `docker run act-jepa:latest` from starting a training job implicitly.
CMD ["bash"]
