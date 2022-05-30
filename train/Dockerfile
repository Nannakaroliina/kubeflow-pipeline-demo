FROM python:3.8-slim
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py && \
    mkdir breast_cancer/data && \
    mv *.parquet breast_cancer/data/ && \
    cd breast_cancer && \
    feast apply && \
    cd ../ && \
    python create_dataset.py && \
    python materialize.py && \
    python train.py