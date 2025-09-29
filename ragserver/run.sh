#!/bin/sh

set -e

RAGSERVER_HOST=0.0.0.0
RAGSERVER_PORT=8000

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

uvicorn ragserver.main:app --host "$RAGSERVER_HOST" --port "$RAGSERVER_PORT" &
