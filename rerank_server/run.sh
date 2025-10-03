#!/bin/sh

set -e

HF_RERANK_HOST=0.0.0.0
HF_RERANK_PORT=8002

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

uvicorn hf_rerank_server:app --host "$HF_RERANK_HOST" --port "$HF_RERANK_PORT" &
