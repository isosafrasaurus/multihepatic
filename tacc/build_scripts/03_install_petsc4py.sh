#!/bin/bash
set -e
set -x

# Install PETSc
git clone https://gitlab.com/petsc/petsc.git /tmp/petsc-src
cd /tmp/petsc-src
git checkout main
DOWNLOADS="\
    --download-metis \
    --download-parmetis \
    --download-superlu \
    --download-superlu_dist \
    --download-blacs \
    --download-scalapack \
    --download-mumps \
    --download-hwloc \
    --download-ptscotch \
    --download-suitesparse \
    --download-chaco \
    --download-triangle \
    --download-ctetgen \
    --download-exodusii \
    --download-netcdf \
    --download-pnetcdf \
    --download-eigen \
    --download-hypre \
    --download-spai \
    --download-ml \
"
./configure \
    --with-clanguage=cxx \
    --with-scalar-type=real \
    --with-debugging=0 \
    --with-hdf5-dir=$INSTALL_PREFIX \
    $DOWNLOADS \
    --prefix=$INSTALL_PREFIX \
    CPPFLAGS="-fPIC" \
    COPTFLAGS="-g -O3" \
    CXXOPTFLAGS="-g -O3" \
    FOPTFLAGS="-g -O3"
PETSC_ARCH=$(grep "^PETSC_ARCH" $PWD/lib/petsc/conf/petscvariables | sed "s/PETSC_ARCH=//")
make PETSC_DIR=$PWD PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PWD PETSC_ARCH=$PETSC_ARCH install

# Install petsc4py
cd /tmp/petsc-src/src/binding/petsc4py/
PETSC_DIR=$INSTALL_PREFIX PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --check-build-dependencies --no-build-isolation --user .
cd && rm -rf /tmp/petsc-src
