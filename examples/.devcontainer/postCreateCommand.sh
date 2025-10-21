#!/usr/bin/env bash
set -e

# 依存パッケージインストール
uv export --no-hashes | uv pip install --no-cache-dir --break-system-packages --system -r -

# CUDA 動作確認
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
