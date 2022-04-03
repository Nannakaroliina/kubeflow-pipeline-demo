FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y apt-utils build-essential ca-certificates checkinstall g++ git wget

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda --version

COPY . /app
WORKDIR /app

RUN git clone https://github.com/deephealthproject/eddl && \
    conda update conda && \
    conda env create -f eddl/environment.yml

SHELL [ "conda", "run", "-n", "eddl", "/bin/bash", "-c"]

RUN mkdir eddl/build

RUN cd eddl/build && \
    cmake .. \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DBUILD_SUPERBUILD=OFF \
    -DBUILD_PROTOBUF=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TARGET=CPU \
    -DBUILD_HPC=OFF -DBUILD_TESTS=OFF \
    -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF && \
    make -j$(nproc) && \
    make install

# delete build to avoid cache problem and copy protobuf file to shared lib folder
RUN rm -rf eddl/build/ && \
    cp $CONDA_PREFIX/lib/libprotobuf.so.22 /usr/lib/

# compile cpp file
RUN cmake . && \
    make

# run compiled file
CMD ./predict > log_p.txt
