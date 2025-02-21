# *** Base *** #
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install curl
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Setup workdir
WORKDIR /mrmustard
COPY pyproject.toml .
COPY uv.lock .

# Install uv, add to path
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /uv_bin/
ENV PATH="${PATH}:/uv_bin"

# Install all dependencies
RUN uv venv -p python${PYTHON_VERSION}
RUN uv sync --all-extras --group doc

ENV DEBIAN_FRONTEND=dialog

# Add source code, tests and configuration
COPY . .

CMD ["uv", "run", "python"]
