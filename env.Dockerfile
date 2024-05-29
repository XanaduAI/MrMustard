# *** Base *** #
FROM python:3.10 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install curl
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Setup workdir
WORKDIR /mrmustard
COPY pyproject.toml .
COPY poetry.lock .

# Upgrade pip and install package manager
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN poetry config virtualenvs.create false
RUN python -m pip install --no-cache-dir --upgrade pip

# Install all dependencies
RUN poetry install --no-root --all-extras --with dev,doc

# Install Julia
COPY juliapkg.json .
RUN poetry run python -c 'from juliacall import Main'

ENV DEBIAN_FRONTEND=dialog

# Add source code, tests and configuration
COPY . .
RUN poetry install --only-root --all-extras
