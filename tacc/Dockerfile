FROM tacc/tacc-ubuntu18-mvapich2.3-ib

RUN useradd -m -s /bin/bash -G sudo fenics && \
    echo "fenics:docker" | chpasswd
USER root
WORKDIR /tmp

ENV FENICS_HOME=/home/fenics
ENV FENICS_PREFIX=$FENICS_HOME/local \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

ENV PETSC_VERSION=3.20.2 \
    SLEPC_VERSION=3.20.1 \
    PYBIND11_VERSION=2.11.1  \
    MPI4PY_VERSION=3.1.5 \
    PETSC4PY_VERSION=3.20.2 \
    SLEPC4PY_VERSION=3.20.1 \
    TRILINOS_VERSION=12.10.1

ARG HDF5_VERSION=1.14.3
ARG H5PY_VERSION=3.12.1
ARG OCCT_TAG=V7_7_2
ARG GMSH_TAG=gmsh_4_11_1

ARG BUILD_JOBS=4
ENV MAKEFLAGS=-j${BUILD_JOBS} \
    CC=mpicc CXX=mpicxx MPICC=mpicc

RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install \
        vim ccache cmake doxygen flex git git-lfs graphviz \
        libboost-filesystem-dev libboost-iostreams-dev libboost-math-dev \
        libboost-program-options-dev libboost-system-dev libboost-thread-dev \
        libboost-timer-dev libeigen3-dev libfreetype6-dev liblapack-dev \
        libopenblas-dev libpcre3-dev libpng-dev libgmp-dev libcln-dev \
        libmpfr-dev man nano pkg-config wget bash-completion \
        # for Python build and various libs
        build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        gfortran \
        libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
        libffi-dev liblzma-dev python3-openssl \
        # OCC build deps (no GUI)
        libfontconfig1-dev libfreetype6-dev libx11-dev libxi-dev libxmu-dev \
        libgl1-mesa-dev mesa-common-dev \
        # runtime bits we will need later
        ca-certificates && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python 3.12
RUN apt-get -qq update && \
    wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar xvf Python-3.12.2.tgz && \
    cd Python-3.12.2 && \
    env -u CC -u CXX -u CFLAGS -u CPPFLAGS -u LDFLAGS \
        CC=gcc CXX=g++ \
        ./configure --enable-optimizations --with-lto && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd /tmp && rm -rf Python-3.12.2.tgz Python-3.12.2

# Update python3 alternative to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# Minimal Python runtime packages
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        python3-dev python3-flufl.lock python3-numpy python3-ply \
        python3-pytest python3-scipy python3-tk python3-urllib3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pip / setuptools for Python 3.12
RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
    python3 get-pip.py && \
    pip3 install --no-cache-dir setuptools wheel && \
    rm -f get-pip.py

# mpi4py against MVAPICH2
RUN MPICC=mpicc pip3 install --no-cache-dir --no-binary=mpi4py "mpi4py==${MPI4PY_VERSION}"

# Make sure MPI wrappers are visible to any build steps
ENV CC=mpicc CXX=mpicxx MPICC=mpicc

# External parallel HDF5
RUN set -eux; \
    wget -q "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-${HDF5_VERSION//./_}.tar.gz" -O hdf5-${HDF5_VERSION}.tar.gz; \
    mkdir -p /tmp/hdf5-src && tar -xf hdf5-${HDF5_VERSION}.tar.gz -C /tmp/hdf5-src --strip-components=1; \
    cd /tmp/hdf5-src && \
    ./configure --enable-parallel --enable-hl --enable-build-mode=production \
                --enable-shared --with-pic --prefix=/usr/local/hdf5 && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/hdf5-src /tmp/hdf5-${HDF5_VERSION}.tar.gz
ENV HDF5_ROOT=/usr/local/hdf5
ENV HDF5_DIR=/usr/local/hdf5

# preinstall cython+numpy in py3.12 so we can disable isolation for h5py
RUN pip3 install --no-cache-dir "numpy==1.26.4" "Cython>=3.0,<3.1"

# system is 3.10 on Ubuntu 18; PETSc TPLs need >=3.21
RUN python3 -m pip install --no-cache-dir "cmake>=3.26,<3.28"
RUN cmake --version

# h5py built against the external parallel HDF5
RUN HDF5_MPI=ON HDF5_DIR=/usr/local/hdf5 \
    pip3 install --no-cache-dir --no-binary=h5py --no-build-isolation "h5py==${H5PY_VERSION}"

# PETSc
RUN set -eux; \
    wget -q "https://gitlab.com/petsc/petsc/-/archive/v${PETSC_VERSION}/petsc-v${PETSC_VERSION}.tar.gz" -O petsc-${PETSC_VERSION}.tar.gz; \
    mkdir -p /tmp/petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C /tmp/petsc-src --strip-components=1; \
    cd /tmp/petsc-src && \
    ./configure CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" FFLAGS="-O2" \
        --with-fc=mpifort \
        --with-fortran-bindings=no \
        --with-debugging=0 \
        --with-hdf5-dir=/usr/local/hdf5 \
        --download-blacs \
        --download-hypre \
        --download-metis \
        --download-mumps \
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu \
        --download-strumpack \
        --download-ml \
        --with-make-np=${BUILD_JOBS} \
        --download-cmake=0 \
        --prefix=/usr/local/petsc-32 \
        --with-cc=mpicc \
        --with-cxx=mpicxx \
    && make ${MAKEFLAGS} \
    && make install \
    && cd / && rm -rf /tmp/petsc-src /tmp/petsc-${PETSC_VERSION}.tar.gz
ENV PETSC_DIR=/usr/local/petsc-32
ENV SCOTCH_DIR=/usr/local/petsc-32

# SLEPc 3.20.1
RUN set -eux; \
    wget -q "https://gitlab.com/slepc/slepc/-/archive/v${SLEPC_VERSION}/slepc-v${SLEPC_VERSION}.tar.gz" -O slepc-${SLEPC_VERSION}.tar.gz; \
    mkdir -p /tmp/slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C /tmp/slepc-src --strip-components=1; \
    cd /tmp/slepc-src && \
    ./configure --prefix=/usr/local/slepc-32 && \
    make SLEPC_DIR="$(pwd)" PETSC_DIR=/usr/local/petsc-32 && \
    make install && \
    rm -rf /tmp/slepc-src /tmp/slepc-${SLEPC_VERSION}.tar.gz
ENV SLEPC_DIR=/usr/local/slepc-32

# Python bindings
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir "Cython>=3.0,<3.1"
RUN PETSC_DIR=/usr/local/petsc-32 pip3 install --no-cache-dir --no-build-isolation "petsc4py==${PETSC4PY_VERSION}"
RUN SLEPC_DIR=/usr/local/slepc-32 pip3 install --no-cache-dir --no-build-isolation "slepc4py==${SLEPC4PY_VERSION}"

# pybind11 2.11.1
RUN set -eux; \
    wget -q "https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz" && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && cd build && \
    cmake -DPYBIND11_TEST=False .. && \
    make -j"$(nproc)" && make install && \
    cd /tmp && rm -rf v${PYBIND11_VERSION}.tar.gz pybind11-${PYBIND11_VERSION}
RUN python3 -m pip install --no-cache-dir "pybind11==${PYBIND11_VERSION}"

# OCC
RUN set -eux; \
    git clone --depth=1 --branch ${OCCT_TAG} https://github.com/Open-Cascade-SAS/OCCT.git /tmp/occt-src && \
    mkdir -p /tmp/occt-src/build && cd /tmp/occt-src/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_SKIP_RPATH=ON \
          -DBUILD_MODULE_ApplicationFramework=0 \
          -DBUILD_MODULE_Draw=0 \
          -DBUILD_MODULE_Visualization=0 \
          .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    cd /tmp && rm -rf /tmp/occt-src

# Runtime bits used by OCC at runtime
RUN apt-get -qq update && apt-get -y install --no-install-recommends libfontconfig1 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Gmsh
RUN set -eux; \
    git clone --depth=1 --branch ${GMSH_TAG} https://gitlab.onelab.info/gmsh/gmsh.git /tmp/gmsh-src && \
    mkdir -p /tmp/gmsh-src/build && cd /tmp/gmsh-src/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_SKIP_RPATH=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DENABLE_BUILD_DYNAMIC=1 \
          .. && \
    make -j"$(nproc)" && make install && \
    # move gmsh python bits to site-packages (as in fem-on-colab)
    PY_SITE=$(python3 -c "import site,sys; print(next(p for p in site.getsitepackages() if sys.version_info[:2]==tuple(map(int, p.split('python')[-1].split('/')[0].split('.')))))"); \
    if [ -f /usr/local/lib/gmsh.py ]; then mv /usr/local/lib/gmsh.py "$PY_SITE/"; fi; \
    shopt -s nullglob; \
    for d in /usr/local/lib/gmsh-*.dist-info; do mv "$d" "$PY_SITE/"; done; \
    ldconfig && \
    cd /tmp && rm -rf /tmp/gmsh-src

# Keep your scientific Python pins helpful for FFC/FIAT and docs
RUN pip3 install --no-cache-dir jupyter jupyterlab matplotlib scipy \
    "sympy>=1.12.1,<1.14" tqdm pkgconfig

# Make FEniCS user environment
WORKDIR $FENICS_HOME
COPY fenics.env.conf $FENICS_HOME/fenics.env.conf
COPY bin $FENICS_HOME/bin
RUN chmod +x bin/* && \
    PYTHON3_SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    PYTHON3_VERSION=$(python3 -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))') && \
    echo "$FENICS_HOME/local/lib/python$PYTHON3_VERSION/site-packages" >> "$PYTHON3_SITE_DIR/fenics-user.pth" && \
    chown -R fenics:fenics $FENICS_HOME

RUN echo 'source ~/.profile' >> $FENICS_HOME/.bash_profile && \
    echo '. ~/fenics.env.conf' >> $FENICS_HOME/.profile && \
    mkdir -p $FENICS_HOME/.config/matplotlib
COPY matplotlibrc $FENICS_HOME/.config/matplotlib/matplotlibrc

USER root
ENV FENICS_BUILD_TYPE=Release
ENV FENICS_SRC_DIR=/tmp/src
ENV FENICS_PREFIX=/usr/local

# Pull & build FEniCS after core deps exist
RUN /home/fenics/bin/fenics-pull && \
    PIP_NO_CACHE_DIR=off /home/fenics/bin/fenics-build

RUN ln -s /usr/local/lib/python3.12/site-packages/ufl_legacy \
           /usr/local/lib/python3.12/site-packages/ufl || true

RUN ldconfig

# Minimal GL for VTK runtime
RUN apt-get -qq update && \
    apt-get -y install --no-install-recommends libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Re‑install pybind11 wheel to ensure Python package visibility
RUN python3 -m pip install --no-cache-dir "pybind11==${PYBIND11_VERSION}"
RUN pip3 install --no-cache-dir networkx vtk
RUN pip3 install --no-cache-dir "git+https://bitbucket.org/fenics-apps/cbc.block.git@master"

RUN set -eux; \
    cd /tmp && \
    git clone --single-branch -b "collapse-iter-dev" "https://github.com/MiroK/fenics_ii" && \
    pip3 install --no-cache-dir fenics_ii/ && \
    rm -rf fenics_ii

RUN set -eux; \
    cd /tmp && \
    git clone "https://github.com/IngeborgGjerde/graphnics" && \
    pip3 install --no-cache-dir graphnics/ && \
    rm -rf graphnics

RUN pip3 install --no-cache-dir --upgrade "git+https://github.com/dolfin-adjoint/pyadjoint.git"

