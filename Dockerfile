FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -U setuptools

COPY requirements.txt requirements.txt
COPY requirements.torch.txt requirements.torch.txt

RUN pip install --no-cache-dir -r requirements.torch.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN jupyter nbextension install --py widgetsnbextension --user