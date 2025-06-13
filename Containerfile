# Usage:
# $ podman build . -t gpu_play
# $ podman run -v $PWD:/src -it gpu_play

FROM    docker.io/library/ubuntu:latest

RUN     apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake gdb g++ git python3 wget

# Install CUDA toolkit for ARM
RUN     mkdir /scratch && cd scratch && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        apt-get -y install cuda-toolkit-12-9

ENV PATH="/usr/local/cuda-12.9/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

