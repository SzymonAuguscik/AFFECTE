FROM python:3.12-slim
# no gcc in python:3.12-slim
RUN apt-get update && \
    apt-get install -y gcc

WORKDIR /affecte

COPY requirements.txt ./
COPY affecte.py ./
COPY src ./src/

RUN pip3 install -r requirements.txt
