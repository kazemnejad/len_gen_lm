"Dockerfile" 71L, 2614B                                                                                                                                                                                                                                                  39,1          All
FROM nvcr.io/nvidia/pytorch:23.09-py3

#FROM nvcr.io/nvidia/pytorch:22.12-py3
LABEL maintainer="Kazemnejad"

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH='2.1.0'
ARG CUDA='cu122'

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir transformers accelerate

# Uninstall `transformer-engine` shipped with the base image
RUN python3 -m pip uninstall -y transformer-engine

# Uninstall `torch-tensorrt` shipped with the base image
RUN python3 -m pip uninstall -y torch-tensorrt


RUN python3 -m pip uninstall -y pydantic
RUN python3 -m pip install --upgrade --no-cache-dir "pydantic<2.0.0"

RUN git clone https://github.com/microsoft/DeepSpeed/
RUN cd DeepSpeed && rm -rf build && TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_AIO=1 pip install . \
    --global-option="build_ext" --global-option="-j8" --no-cache -v \
    --disable-pip-version-check 2>&1

# The base image ships with `pydantic==1.8.2` which is not working - i.e. the next command fails
RUN python3 -c "from deepspeed.launcher.runner import main"

# Install the rest of the python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade \
    wandb \
    datasets \
    evaluate \
    sentencepiece \
    scikit-learn \
    protobuf

# Add ngrok's GPG key and debian repository
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | gpg --dearmor -o /usr/share/keyrings/ngrok-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ngrok-archive-keyring.gpg] https://ngrok-agent.s3.amazonaws.com buster main" > /etc/apt/sources.list.d/ngrok.list

# Install ngrok
RUN apt-get update && \
    apt-get install -y ngrok && \
    apt-get clean

RUN ngrok authtoken