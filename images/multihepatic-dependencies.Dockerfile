FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

USER root

ARG USERNAME=dolfinx
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
      EXISTING_GROUP="$(getent group "${USER_GID}" | cut -d: -f1)"; \
      groupadd -f "${USERNAME}" || true; \
      useradd -m -u "${USER_UID}" -g "${EXISTING_GROUP}" -s /bin/bash "${USERNAME}" || true; \
    else \
      groupadd -g "${USER_GID}" "${USERNAME}"; \
      useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}"; \
    fi

WORKDIR /workspace

ENV HOME=/home/${USERNAME}
ENV XDG_CACHE_HOME=/home/${USERNAME}/.cache

RUN mkdir -p /tmp/src \
    /home/${USERNAME}/.cache \
    /home/${USERNAME}/.cache/fenics \
    /workspace && \
    chown -R ${USER_UID}:${USER_GID} /home/${USERNAME} /workspace /tmp/src

RUN SYSTEM_SITE_PACKAGES=$(python3 -c "import sys; print([p for p in sys.path if 'dist-packages' in p and 'local' in p][0])") && \
    echo "Found system packages at: $SYSTEM_SITE_PACKAGES" && \
    echo "$SYSTEM_SITE_PACKAGES" > /dolfinx-env/lib/python3.12/site-packages/system_packages.pth

RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/fenicsx_ii.git && \
    cd fenicsx_ii && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/networks_fenicsx.git && \
    cd networks_fenicsx && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

RUN /dolfinx-env/bin/python3 -m pip install \
    networkx vtk meshio nibabel pyvista tetgen pymeshfix h5py

RUN rm -rf /tmp/src

USER ${USERNAME}

ENV PYTHONUNBUFFERED=1
ENV HOME=/home/${USERNAME}
ENV XDG_CACHE_HOME=${HOME}/.cache
ENV MPLCONFIGDIR=${XDG_CACHE_HOME}/matplotlib