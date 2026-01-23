FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
LABEL authors="corazon008"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt update && apt upgrade -y && apt install -y unzip wget curl sudo git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . ./computer_vision
COPY ../../pyproject.toml ./pyproject.toml

RUN uv sync


CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5000", "computer_vision.api.app:app"]