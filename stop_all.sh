#!/bin/sh

set -e

terminate_service() {
    name="$1"
    port="$2"

    if [ -z "$name" ] || [ -z "$port" ]; then
        return
    fi

    echo "stopping $name (port: $port)..."
    pids=$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)

    if [ -z "$pids" ]; then
        echo "  no process found"
        return
    fi

    for pid in $pids; do
        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        echo "  sending SIGINT to pid $pid"
        kill -INT "$pid" 2>/dev/null || true
    done

    sleep 2

    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  force stopping pid $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

terminate_service "ragserver" 8000
terminate_service "local rerank server" 8002
terminate_service "local embed server" 8001
terminate_service "chroma server" 8003

wait || true

terminate_service "ragserver" 8000
terminate_service "local rerank server" 8002
terminate_service "local embed server" 8001
terminate_service "chroma server" 8003

echo "all services stopped."
