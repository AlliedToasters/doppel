# One image per backend service: docker build --build-arg PKG=doppel-api -f deploy/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv
ARG PKG
ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/venv PATH=/venv/bin:$PATH
WORKDIR /src
COPY . .
RUN uv sync --frozen --no-dev --package "$PKG" $EXTRAS
RUN useradd -m -u 10001 app && mkdir -p /data && chown app /data
USER app
WORKDIR /data
