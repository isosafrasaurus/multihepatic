FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN SYSTEM_SITE_PACKAGES=$(python3 -c "import sys; print([p for p in sys.path if 'dist-packages' in p and 'local' in p][0])") && \
    echo "Found system packages at: $SYSTEM_SITE_PACKAGES" && \
    echo "$SYSTEM_SITE_PACKAGES" > /dolfinx-env/lib/python3.12/site-packages/system_packages.pth

RUN mkdir -p tmp/

# Install fenicsx_ii
RUN cd tmp/ && \
    git clone https://github.com/scientificcomputing/fenicsx_ii.git && \
    cd fenicsx_ii && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

# Install networks_dolfinx
RUN cd tmp/ && \
    git clone https://github.com/scientificcomputing/networks_fenicsx.git && \
    cd networks_fenicsx && \
    /dolfinx-env/bin/python3 -m pip install . && \
    cd ..

# Cleanup
RUN rm -rf tmp/

# Install networkx
RUN /dolfinx-env/bin/python3 -m pip install networkx vtk