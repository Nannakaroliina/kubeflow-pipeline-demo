FROM python:3.6-slim
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY preprocess_data.py ./preprocess_data.py

ENTRYPOINT [ "python", "preprocess_data.py" ]