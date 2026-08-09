#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="qdrant"
STORAGE_DIR="${QDRANT_STORAGE_DIR:-$HOME/qdrant_storage}"
IMAGE="docker.io/qdrant/qdrant:latest"

mkdir -p "$STORAGE_DIR"

if podman container exists "$CONTAINER_NAME"; then
    echo "Container '$CONTAINER_NAME' already exists — starting it."
    podman start "$CONTAINER_NAME"
else
    echo "Creating and starting container '$CONTAINER_NAME'."
    podman run -d \
        --name "$CONTAINER_NAME" \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "$STORAGE_DIR:/qdrant/storage:Z" \
        "$IMAGE"
fi

podman ps --filter "name=$CONTAINER_NAME"