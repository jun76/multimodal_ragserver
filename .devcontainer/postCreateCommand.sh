#!/usr/bin/env bash
set -e

# CUDA 動作確認
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
