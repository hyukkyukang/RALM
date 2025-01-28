# Multi Stage bulid: cmake
FROM hyukkyukang/cmake:latest AS cmake

# Multi Stage build: Main build
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make cmake g++ wget zip vim sudo tmux ninja-build -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Install prerequisites for python3.13
RUN apt install build-essential checkinstall libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y 
# Install python3.13
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.13 python3.13-dev -y
RUN apt-get -y install python3-pip python-is-python3 python3.13-venv
RUN echo "export PYTHONPATH=./" >> ~/.bashrc
RUN echo "export CONFIGPATH=./config.yml" >> ~/.bashrc
#RUN echo "export SETUPTOOLS_USE_DISTUTILS=stdlib" >> ~/.bashrc
# Set default python version to 3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Upgrade setuptools and pip
RUN python -m ensurepip --upgrade
RUN python3.13 -m pip install --upgrade setuptools
RUN pip install --upgrade pip

# Install prerequisites for faiss-gpu
RUN apt-get install swig libblas-dev liblapack-dev libatlas-base-dev -y

# Install Locale
RUN apt-get install language-pack-en -y

# Install numpy
RUN pip install numpy

RUN apt-get install -y libgflags-dev

# Install cmake
#RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3.tar.gz && tar -zxvf cmake-3.27.3.tar.gz && cd cmake-3.27.3 && ./bootstrap && make && make install && cd .. && rm -r cmake-3.27.3.tar.gz cmake-3.27.3

# Copy from cmake build
COPY --from=cmake /cmake-3.27.3 /cmake-3.27.3
RUN cd /cmake-3.27.3 && make install && rm -r /cmake-3.27.3

# Install faiss-gpu
RUN git clone https://github.com/facebookresearch/faiss.git
RUN cd faiss && cmake -B build . && make -C build -j faiss && make -C build -j swigfaiss && cd build/faiss/python && python setup.py install && cd ../../.. && rm -r faiss

# Export environment variables
RUN echo "export PATH=${PATH}:/usr/local/cuda/bin" >> /etc/environment
RUN echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> /etc/environment
RUN echo "export CUDA_HOME=/usr/local/cuda" >> /etc/environment

# Change directory permission
RUN chmod 777 /root
RUN echo "root:root" | chpasswd
