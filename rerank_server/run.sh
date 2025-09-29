#!/bin/sh

set -e

LOCAL_RERANK_HOST=0.0.0.0
LOCAL_RERANK_PORT=8002

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

uvicorn local_rerank_server:app --host "$LOCAL_RERANK_HOST" --port "$LOCAL_RERANK_PORT" &
