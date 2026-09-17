FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 UV_PROJECT_ENVIRONMENT=/opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml README.md /app/
COPY continuum /app/continuum
RUN uv sync --extra agent
CMD ["continuum-mcp", "--host", "0.0.0.0"]
