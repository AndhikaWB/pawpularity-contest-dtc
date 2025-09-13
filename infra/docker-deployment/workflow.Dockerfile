FROM prefecthq/prefect:3.4.13-python3.12

# Install to system instead of local .venv folder
# This is because Prefect will use the system Python
ENV UV_PROJECT_ENVIRONMENT="/usr/local"

# Use copy since we are using mounted cache volume
ENV UV_LINK_MODE="copy"
# # Don't sync unneeded dependencies when using "uv run"
# ENV UV_NO_SYNC="true"
# # Smaller overall size at the cost of re-download
# ENV UV_NO_CACHE="true"
# ENV UV_FROZEN="true"

WORKDIR /opt/prefect/pawpaw/

# Copy uv executables from the distroless image
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/
# Copy the data needed by the preprocess script
COPY data/raw/ /opt/prefect/pawpaw/data/raw/

COPY src/ /opt/prefect/pawpaw/src/
COPY Makefile /opt/prefect/pawpaw/
COPY pyproject.toml /opt/prefect/pawpaw/
COPY uv.lock /opt/prefect/pawpaw/

# Install curl for testing other host/container reachability
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt update && apt install -y curl

# This container uses the CPU version of PyTorch
# I don't have time to look up the CUDA version yet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --verbose --inexact --no-dev --group cpu --group monitoring