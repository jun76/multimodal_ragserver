FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
# FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 基本ツール & GUI関連
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    zsh \
    unzip \
    locales \
    net-tools \
    lsof \
    python3 \
    python3-pip \
    sqlite3 \
    postgresql \
    postgresql-contrib \
    postgresql-16 \
    postgresql-client-16 \
    postgresql-server-dev-16 \
    postgresql-16-pgvector \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US:en

# Pythonパッケージ
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

CMD ["sleep", "infinity"]
