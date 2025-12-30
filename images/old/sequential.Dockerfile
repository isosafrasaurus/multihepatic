FROM ghcr.io/isosafrasaurus/ubuntu22.04-python3.12-graphnics:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN mkdir -p /tmp/matplotlib && chmod 1777 /tmp/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

# Create a writable workspace owned by the non-root user to avoid permission issues
RUN set -eux; \
    mkdir -p /workspace; \
    chown -R fenics:fenics /workspace; \
    chmod 775 /workspace

# Install nibabel with the same Python/pip from the base image
RUN python3 -m pip install --no-cache-dir nibabel

# Default to the non-root user
USER fenics
ENV HOME=/home/fenics
WORKDIR /workspace

