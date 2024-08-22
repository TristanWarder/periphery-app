# Use Nvidia L4T Jetpack image as a base
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# Update packages
RUN apt update && apt upgrade -y

# Setup python and cpp build tools
RUN apt install cmake python-is-python3 python3-pip ninja-build -y

# Add container user
RUN useradd -m -d /home/periphery periphery
RUN chown -R periphery:periphery /home/periphery

WORKDIR /home/periphery

USER periphery

# Install yolov8 python dependencies
RUN pip install numpy onnx onnxsim opencv-python torch torchvision ultralytics
