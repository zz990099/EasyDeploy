
# Base image starts with CUDA
ARG BASE_IMG=nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel
FROM ${BASE_IMG} as base
ENV BASE_IMG=nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel

ENV TENSORRT_VERSION=8.5.2.2

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update


# Install basic dependencies
RUN apt install -y \
    build-essential \
    manpages-dev \
    wget \
    zlib1g \
    software-properties-common \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    ca-certificates \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    mecab-ipadic-utf8 \
    libopencv-dev \
    cmake 

# install cmake-3.22.6
RUN cd /tmp && \
    wget https://gh.api.99988866.xyz/https://github.com/Kitware/CMake/releases/download/v3.22.6/cmake-3.22.6.tar.gz && \
    tar -xzvf cmake-3.22.6.tar.gz && \
    rm cmake-3.22.6.tar.gz
RUN cd /tmp/cmake-3.22.6/ && \
    ./configure && \
    make -j4 && \
    make install

# install glog
RUN cd /tmp && \
    wget https://gh.api.99988866.xyz/https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz && \
    tar -xzvf v0.5.0.tar.gz && \
    rm v0.5.0.tar.gz
RUN cd /tmp/glog-0.5.0 && \
    mkdir build && cd build && \
    cmake .. && make -j4 && \
    make install

# install gtest
RUN cd /tmp && \
    wget https://gh.api.99988866.xyz/https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz && \
    tar -xzvf release-1.11.0.tar.gz && \
    rm release-1.11.0.tar.gz
RUN cd /tmp/googletest-release-1.11.0 && \
    mkdir build && cd build && \
    cmake .. && make -j4 && \
    make install