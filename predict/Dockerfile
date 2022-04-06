FROM continuumio/anaconda3
WORKDIR /app

COPY predict.py ./predict.py
COPY install.yml ./install.yml

RUN conda --version && \
    conda update conda &&\
    conda config --add channels dhealth && \ 
    conda config --add channels conda-forge && \
    conda env create -f install.yml

ENTRYPOINT [ "conda", "run", "-n", "pyeddlkube", "python", "predict.py"]
