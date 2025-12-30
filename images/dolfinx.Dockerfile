FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

USER root

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN SYSTEM_SITE_PACKAGES=$(python3 -c "import sys; print([p for p in sys.path if 'dist-packages' in p and 'local' in p][0])") && \
    echo "Found system packages at: $SYSTEM_SITE_PACKAGES" && \
    echo "$SYSTEM_SITE_PACKAGES" > /dolfinx-env/lib/python3.12/site-packages/system_packages.pth

RUN groupadd -g 1001 dolfinx && \
    useradd -r -u 1001 -g dolfinx -m dolfinx

RUN mkdir -p /tmp/src && \
    mkdir -p /home/dolfinx

# Install fenicsx_ii
RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/fenicsx_ii.git && \
    cd fenicsx_ii && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

# Install networks_dolfinx
RUN cd /tmp/src && \
    git clone https://github.com/scientificcomputing/networks_fenicsx.git && \
    cd networks_fenicsx && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

RUN /dolfinx-env/bin/python3 -m pip install networkx vtk meshio

# Cleanup
RUN rm -rf /tmp/src

USER dolfinx
WORKDIR /home/dolfinx