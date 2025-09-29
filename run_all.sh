#!/bin/sh

set -e

# echo "start chroma server..."
# chroma_server/run.sh
# sleep 5

echo "start embed server..."
embed_server/run.sh
sleep 5

echo "start rerank server..."
rerank_server/run.sh
sleep 5

echo "start ragserver..."
ragserver/run.sh
