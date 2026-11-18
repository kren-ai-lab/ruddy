# Runtime image of a released Ruddy version, installed from PyPI (no source copied), so any past
# release can be rebuilt. Built and pushed by .github/workflows/publish-container.yml.
#   docker build --build-arg RUDDY_VERSION=0.2.0 -t ruddy:0.2.0 .
#   docker build --build-arg RUDDY_VERSION=0.2.0 --build-arg EXTRAS=[manifold] -t ruddy:0.2.0-full .
FROM python:3.14-slim

ARG RUDDY_VERSION
ARG EXTRAS=""
LABEL org.opencontainers.image.source="https://github.com/kren-ai-lab/ruddy" \
      org.opencontainers.image.description="Ruddy: statistical EDA for tabular datasets and feature representations" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${RUDDY_VERSION}"

# procps: Nextflow reads task metrics with `ps` inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# typer<0.28: ruddy 0.2.0 allows typer 0.28+, which breaks `--config`; drop once 0.2.0 is rebuilt.
RUN test -n "${RUDDY_VERSION}" \
    && pip install --no-cache-dir "ruddy${EXTRAS}==${RUDDY_VERSION}" "typer>=0.27,<0.28"
