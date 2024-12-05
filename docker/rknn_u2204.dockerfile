FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list && \
  echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  apt-get update

RUN apt-get install -y \
  build-essential \
  manpages-dev \
  wget \
  software-properties-common \
  git \
  libssl-dev \
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
  mecab-ipadic-utf8 \
  sudo 

RUN apt-get install -y \
  cmake \
  libopencv-dev \
  libeigen3-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libassimp-dev \
  assimp-utils

# use github release proxy speedup
# see [https://github.com/hunshcn/gh-proxy]
RUN cd /tmp && \
  wget https://gh.api.99988866.xyz/https://github.com/airockchip/rknn-toolkit2/archive/refs/tags/v2.3.0.tar.gz

RUN cd /tmp && \
  tar -xzvf v2.3.0.tar.gz && \
  rm v2.3.0.tar.gz

RUN cd /tmp/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/ && \
  cp include/* /usr/include/ && \
  cp aarch64/* /usr/lib

# rknn-toolkit2 python package on arm64
RUN apt-get install python3-pip -y && \
  pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
  pip install pip --upgrade

RUN cd /tmp/rknn-toolkit2-2.3.0/rknn-toolkit2/packages/arm64 && \
  pip install rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
