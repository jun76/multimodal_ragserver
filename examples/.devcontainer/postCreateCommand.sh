#!/usr/bin/env bash
set -e

# CUDA availability check
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
