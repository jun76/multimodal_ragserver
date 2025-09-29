# !/bin/sh

set -e

CHROMA_PERSIST_DIR=chroma_db
CHROMA_HOST=localhost
CHROMA_PORT=8003

chroma run --path $CHROMA_PERSIST_DIR --host $CHROMA_HOST --port $CHROMA_PORT &
