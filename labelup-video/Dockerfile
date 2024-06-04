# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

#TODO switch to cuda 
#FROM python:${PYTHON_VERSION}-slim AS python-base 
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    DEBIAN_FRONTEND=noninteractive 

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get autoremove -y

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt update; \
    apt install -y wget

#opencv dependencies
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt


RUN python -m pip install --upgrade pip
RUN chmod 777 /app
# TODO install cotracker 
RUN git clone https://github.com/facebookresearch/co-tracker

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    cd /app/co-tracker/ && \
    pip install -e .

#install labelup core

RUN git clone https://github.com/MetroStar/labelup-core 

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    cd /app/labelup-core/ && \
    pip install -e .


RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install opencv-python einops timm matplotlib moviepy flow_vis

RUN mkdir /app/co-tracker/checkpoints
RUN cd /app/co-tracker/checkpoints/ && \
    wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

# TODO install SAM 
RUN mkdir /app/sam_weights
RUN cd /app/sam_weights/ && \
    wget https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth


# sam_weights/sam_vit_h_4b8939.pth
# install seggpt
RUN mkdir /app/seggpt_weights
RUN cd /app/seggpt_weights/ && \
    wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth
# TODO make sure code references local cache

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . .

EXPOSE 9090

#CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app # --preload causes cuda errors
CMD gunicorn --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app