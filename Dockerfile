# Multi-Stage Build: cmake
FROM hyukkyukang/cmake:3.30.8-ubuntu24.04 AS cmake-stage

# Multi-Stage Build: Main
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set timezone
ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install core utilities
RUN apt-get update && apt-get install -y \
    gnupg git curl wget zip vim sudo tmux && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install build tools
RUN apt-get update && apt-get install -y \
    make ninja-build g++ build-essential checkinstall && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies for building Python 3.13
RUN apt-get update && apt-get install -y \
    libssl-dev libsqlite3-dev libncursesw5-dev tk-dev \
    libgdbm-dev libc6-dev libbz2-dev libreadline-dev libffi-dev \
    liblzma-dev libgdm-dev zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build Python 3.13 from source
WORKDIR /usr/src
RUN curl -O https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tgz && \
    tar -xvf Python-3.13.2.tgz && \
    cd Python-3.13.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-3.13.2 Python-3.13.2.tgz

# Set default Python version
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.13 1

# Upgrade pip and setuptools
RUN python3.13 -m ensurepip && \
    python3.13 -m pip install --upgrade pip setuptools

# Install scientific computing dependencies
RUN apt-get update && apt-get install -y \
    swig libblas-dev liblapack-dev libatlas-base-dev libgflags-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install system locale
RUN apt-get update && apt-get install -y language-pack-en && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install NumPy
RUN pip install numpy

# Copy installed the CMake binary
COPY --from=cmake-stage /usr/local/bin/cmake /usr/local/bin/cmake
COPY --from=cmake-stage /usr/local/share/cmake-3.30 /usr/local/share/cmake-3.30

# Verify that CMake is correctly installed
RUN cmake --version

# Install faiss-gpu
RUN git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . && make -C build -j faiss && \
    make -C build -j swigfaiss && \
    cd build/faiss/python && python setup.py install && \
    cd ../../.. && rm -rf faiss

# Export environment variables
ENV PATH="${PATH}:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=./

# Fix permissions
RUN chmod 777 /root && echo "root:root" | chpasswd