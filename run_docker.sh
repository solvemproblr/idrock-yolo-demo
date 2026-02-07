#!/bin/bash

# Script to run the YOLO phone detection demo in NVIDIA PyTorch container
# with GPU, camera, and display support

xhost +local:docker  # Allow Docker to access X server

sudo docker run --rm --gpus all \
  -v /home/newuu_1/idrock-yolo-demo:/workspace \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device=/dev/video0:/dev/video0 \
  --ipc=host \
  nvcr.io/nvidia/pytorch:26.01-py3 bash -c "\
    apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
                       libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
                       libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
                       libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 \
                       libxkbcommon-x11-0 libdbus-1-3 && \
    pip install opencv-python ultralytics && \
    python main.py"

xhost -local:docker  # Revoke access after running
