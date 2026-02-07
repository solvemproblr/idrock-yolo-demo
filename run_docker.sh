#!/bin/bash

# Script to run the YOLO phone detection demo in custom Docker image
# with GPU, camera, and display support

# Build the Docker image if it doesn't exist
IMAGE_NAME="yolo-phone-demo"
if [[ "$(sudo docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Building Docker image (one-time setup)..."
  sudo docker build -t $IMAGE_NAME .
fi

xhost +local:docker  # Allow Docker to access X server

sudo docker run --rm --gpus all \
  -v /home/newuu_1/idrock-yolo-demo:/workspace \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device=/dev/video0:/dev/video0 \
  --ipc=host \
  $IMAGE_NAME

xhost -local:docker  # Revoke access after running
