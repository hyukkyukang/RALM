# --------------------------------------------
# Stage 1: CUDA-independent base with Python, tools
# --------------------------------------------
FROM ubuntu:24.04 as base-env

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    tzdata gnupg git curl wget zip vim sudo tmux \
    make ninja-build g++ build-essential checkinstall \
    libssl-dev libsqlite3-dev libncursesw5-dev tk-dev \
    libgdbm-dev libc6-dev libbz2-dev libreadline-dev libffi-dev \
    liblzma-dev libgdm-dev zlib1g-dev \
    swig libblas-dev liblapack-dev libatlas-base-dev libgflags-dev \
    language-pack-en && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build Python 3.13
WORKDIR /usr/src
RUN curl -O https://www.python.org/ftp/python/3.13.3/Python-3.13.3.tgz && \
    tar -xvf Python-3.13.3.tgz && \
    cd Python-3.13.3 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-3.13.3 Python-3.13.3.tgz

# Set Python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.13 1 && \
    python3.13 -m ensurepip && \
    python3.13 -m pip install --upgrade pip setuptools numpy && \
    python3.13 -m pip cache purge

# --------------------------------------------
# Stage 2: Final image with CUDA + FAISS
# --------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Copy base environment
COPY --from=base-env / /

# Set environment variables
ENV PATH="${PATH}:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=./

# Add CMake from custom image
COPY --from=hyukkyukang/cmake:3.30.8-ubuntu24.04 /usr/local/bin/cmake /usr/local/bin/cmake
COPY --from=hyukkyukang/cmake:3.30.8-ubuntu24.04 /usr/local/share/cmake-3.30 /usr/local/share/cmake-3.30

RUN cmake --version

# Build and install FAISS
RUN git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . && make -C build -j faiss && \
    make -C build -j swigfaiss && \
    cd build/faiss/python && python setup.py install && \
    cd ../../.. && rm -rf faiss

# Create a non-root user
RUN useradd -ms /bin/bash user && \
    echo "user:user" | chpasswd && \
    usermod -aG sudo user

# Set working directory and user
USER user
WORKDIR /home/user
