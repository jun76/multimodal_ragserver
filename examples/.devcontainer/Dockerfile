FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
# FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    zsh \
    unzip \
    locales \
    net-tools \
    lsof \
    tree \
    python3 \
    python3-pip \
    sqlite3 \
    postgresql-16 \
    postgresql-client-16 \
    postgresql-server-dev-16 \
    postgresql-16-pgvector \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

CMD ["sleep", "infinity"]
