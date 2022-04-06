FROM ubuntu:20.04
WORKDIR /app

COPY train.py ./train.py
COPY install.yml ./install.yml

RUN apt-get update && \
    apt-get install -y build-essential wget

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:$PATH"

RUN conda --version && \
    conda update conda &&\
    conda config --add channels dhealth && \ 
    conda config --add channels conda-forge && \
    conda env create -f install.yml
    
ENTRYPOINT [ "conda", "run", "-n", "pyeddlkube", "python", "train.py"]
