#!/bin/sh

set -e

LOCAL_EMBED_HOST=0.0.0.0
LOCAL_EMBED_PORT=8001

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

uvicorn local_embed_server:app --host "$LOCAL_EMBED_HOST" --port "$LOCAL_EMBED_PORT" &
