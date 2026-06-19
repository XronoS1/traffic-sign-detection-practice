FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-docker.txt

COPY service /app/service
COPY src /app/src
COPY smoke_tests /app/smoke_tests
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

WORKDIR /app/service

EXPOSE 8000

CMD ["sh", "/app/docker-entrypoint.sh"]
