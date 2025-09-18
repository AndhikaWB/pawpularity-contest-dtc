FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Use copy since we are using mounted cache volume
ENV UV_LINK_MODE="copy"
# # Don't sync unneeded dependencies when using "uv run"
# ENV UV_NO_SYNC="true"
# # Smaller overall size at the cost of re-download
# ENV UV_NO_CACHE="true"
# ENV UV_FROZEN="true"

WORKDIR /app/

COPY src/ /app/src/
COPY Makefile /app/
COPY pyproject.toml /app/
COPY uv.lock /app/

EXPOSE 8765

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
    apt update && apt install -y make

# This container uses the CPU version of PyTorch
# I don't have time to look up the CUDA version yet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --verbose --no-dev --group cpu

CMD ["make", "serving"]