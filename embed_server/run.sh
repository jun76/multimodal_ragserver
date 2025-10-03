#!/bin/sh

set -e

HFCLIP_EMBED_HOST=0.0.0.0
HFCLIP_EMBED_PORT=8001

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

uvicorn hfclip_embed_server:app --host "$HFCLIP_EMBED_HOST" --port "$HFCLIP_EMBED_PORT" &
