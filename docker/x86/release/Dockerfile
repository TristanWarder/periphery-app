# Use Nvidia L4T Jetpack image as a base
FROM nvcr.io/nvidia/tensorrt:25.01-py3
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update packages
RUN apt update && apt upgrade -y

# Setup python and pip
RUN apt install cmake python-is-python3 python3-pip protobuf-compiler sudo git -y

# Add container user
RUN useradd -m -d /home/periphery periphery
RUN chown -R periphery:periphery /home/periphery
RUN adduser periphery video
RUN adduser periphery sudo
RUN echo "periphery:periphery" | chpasswd

WORKDIR /home/periphery

RUN DEBIAN_FRONTEND="noninteractive" apt install libopencv-dev -y

# Run pip as user
USER periphery
# Install yolov8 python dependencies
RUN pip install onnxsim --use-pep517
RUN pip install numpy onnx opencv-python torch torchvision ultralytics

RUN echo "export PATH=$PATH:/home/periphery/.local/bin" >> ~/.bashrc
RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc

# Install packages as root
USER root
# Install other packages
RUN apt install curl ninja-build udev nano vim gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good -y

# Switch to user
USER periphery

# Install NodeJS
ENV NODE_VERSION=22.7.0
ENV NVM_DIR=/home/periphery/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN echo "PATH=${PATH}" >> ~/.bashrc
RUN node --version
RUN npm --version
RUN npm install cmake-js -g

ADD https://github.com/TristanWarder/periphery-app.git ~/

WORKDIR /home/periphery/periphery-app
RUN npm install
