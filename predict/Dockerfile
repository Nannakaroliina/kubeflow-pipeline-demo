FROM python:3.6-slim
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py ./predict.py

ENTRYPOINT [ "python", "predict.py" ]