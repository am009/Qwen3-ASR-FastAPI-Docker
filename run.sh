#!/bin/bash
set -e

# Load .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Defaults
API_PORT="${API_PORT:-18123}"
HOST_MODELS_PATH="${HOST_MODELS_PATH:-./models}"

IMAGE_NAME="qwen3-asr"
CONTAINER_NAME="qwen3_asr_api"

# Build image if not exists
docker build -t "$IMAGE_NAME" .

# Remove old container if exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run
docker run -d \
  --name "$CONTAINER_NAME" \
  --privileged \
  --gpus all \
  --env-file .env \
  -p "${API_PORT}:8000" \
  -v "${HOST_MODELS_PATH}:/app/models" \
  -v ./src:/app/src \
  -v ./qwen_asr:/app/qwen_asr \
  "$IMAGE_NAME"

echo "Container '$CONTAINER_NAME' started on port $API_PORT"
