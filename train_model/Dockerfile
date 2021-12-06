FROM python:3.8-slim
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py ./train.py

ENTRYPOINT [ "python", "train.py" ]