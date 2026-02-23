# Multi-stage Dockerfile for IVOS Model TRE Deployment
# Build on M1 Mac, run on A10 GPU in TRE
# Following uv best practices: https://docs.astral.sh/uv/guides/integration/docker/

# Builder stage - install dependencies using uv
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PRIMITIVO_MODEL=0.1.dev0+docker

WORKDIR /app

# Copy project files for build
COPY pyproject.toml uv.lock ./
COPY primitivo_model ./primitivo_model

# Install dependencies and project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Remove the source code (will be mounted at runtime)
RUN rm -rf primitivo_model/*

# Runtime stage - Start from CUDA base and add Python + dependencies
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install system utilities and libgomp (required by LightGBM)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire Python installation from builder (Debian bookworm Python 3.12)
COPY --from=builder /usr/local /usr/local

# Copy the virtual environment (includes editable install pointing to /app/primitivo_model)
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data \
    MLFLOW_TRACKING_URI=file:///app/mlruns

# Create directories for data, MLflow, code, scripts, and notebooks mount points
RUN mkdir -p /app/data /app/mlruns /app/outputs /app/primitivo_model /app/scripts /app/lab

# Expose MLflow UI port
EXPOSE 5000

CMD ["/bin/bash"]
