# *** Base *** #
FROM python:3.9 AS base

# Setup workdir
WORKDIR /mrmustard
COPY pyproject.toml .
COPY poetry.lock .
ENV PYTHONPATH "/mrmustard"

# Upgrade pip and install package manager
RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir poetry==1.4.0
RUN poetry config virtualenvs.create false

# Install all dependencies
RUN poetry install --all-extras --with dev,doc


# *** Testing *** #
FROM base AS testing
# Add source code, tests and configuration
COPY . .
