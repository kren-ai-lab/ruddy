# Runtime image of a released Ruddy version, installed from PyPI.
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

RUN test -n "${RUDDY_VERSION}" \
    && pip install --no-cache-dir "ruddy${EXTRAS}==${RUDDY_VERSION}"
