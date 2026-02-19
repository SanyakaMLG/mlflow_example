FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

RUN apt-get update && apt-get install -y git

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app
