#!/bin/bash
set -e
set -x

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../ && pwd)"

# FIAT
git clone https://github.com/FEniCS/fiat.git /tmp/fiat-src
cd /tmp/fiat-src
patch -p1 < "$REPODIR/patches/fenics/06-pkg-resources-to-importlib-in-fiat.patch"
patch -p1 < "$REPODIR/patches/fenics/16-sympy-1-14-compatibility-fiat.patch"
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/fiat-src

# dijitso
git clone https://bitbucket.org/fenics-project/dijitso.git /tmp/dijitso-src
cd /tmp/dijitso-src
patch -p1 < "$REPODIR/patches/fenics/08-pkg-resources-to-importlib-in-dijitso.patch"
patch -p1 < "$REPODIR/patches/fenics/09-c++-14-in-dijitso.patch"
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/dijitso-src

# UFL (legacy)
git clone https://github.com/FEniCS/ufl-legacy.git /tmp/ufl-legacy-src
cd /tmp/ufl-legacy-src
patch -p1 < "$REPODIR/patches/fenics/13-pkg-resources-to-importlib-in-ufl-legacy.patch"
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/ufl-legacy-src

# FFC
git clone https://bitbucket.org/fenics-project/ffc.git /tmp/ffc-src
cd /tmp/ffc-src
patch -p1 < "$REPODIR/patches/fenics/14-pkg-resources-to-importlib-in-ffc.patch"
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/ffc-src

# dolfin
git clone https://bitbucket.org/fenics-project/dolfin.git /tmp/dolfin-src
cd /tmp/dolfin-src/
patch -p1 < "$REPODIR/patches/fenics/02-xdmf-checkpoint-fix.patch"
sed -i "s|INSTALL_PREFIX_IN|${INSTALL_PREFIX}|g" "$REPODIR/patches/fenics/03-add-pkg-config-path.patch"
patch -p1 < "$REPODIR/patches/fenics/03-add-pkg-config-path.patch"
patch -p1 < "$REPODIR/patches/fenics/04-deprecated-boost-filesystem.patch"
patch -p1 < "$REPODIR/patches/fenics/05-deprecated-std-bind2nd.patch"
patch -p1 < "$REPODIR/patches/fenics/07-deprecated-petsc.patch"
patch -p1 < "$REPODIR/patches/fenics/10-c++-14-in-dolfin.patch"
patch -p1 < "$REPODIR/patches/fenics/12-do-not-fiddle-with-dlopenflags-in-dolfin.patch"
mkdir -p /tmp/dolfin-src/build
cd /tmp/dolfin-src/build
export UFC_DIR=$INSTALL_PREFIX
export PETSC_DIR=$INSTALL_PREFIX
export SLEPC_DIR=$INSTALL_PREFIX
export BOOST_DIR=$INSTALL_PREFIX
cmake \
    -DCMAKE_C_COMPILER=$(which mpicc) \
    -DCMAKE_CXX_COMPILER=$(which mpicxx) \
    -DCMAKE_SKIP_RPATH:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
    ..
make -j $(nproc) install
cd /tmp/dolfin-src/python
export DOLFIN_DIR=$INSTALL_PREFIX
PYTHONUSERBASE=$INSTALL_PREFIX CXX="mpicxx" python3 -m pip install --user .
cd && rm -rf /tmp/dolfin-src/

# CGAL (required by mshr)
# You might need to load gmp and mpfr modules on TACC
git clone https://github.com/CGAL/cgal.git /tmp/cgal-src
cd /tmp/cgal-src
git checkout 5.6.x-branch
mkdir -p /tmp/cgal-src/build
cd /tmp/cgal-src/build
cmake \
    -DCMAKE_C_COMPILER=$(which mpicc) \
    -DCMAKE_CXX_COMPILER=$(which mpicxx) \
    -DCMAKE_CXX_FLAGS="-std=c++14 -fPIC" \
    -DCMAKE_SKIP_RPATH:BOOL=ON \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
    -DWITH_demos:BOOL=OFF -DWITH_examples:BOOL=OFF \
    ..
make -j $(nproc) install
cd && rm -rf /tmp/cgal-src

# mshr
git clone https://bitbucket.org/fenics-project/mshr.git /tmp/mshr-src
cd /tmp/mshr-src/
patch -p1 < "$REPODIR/patches/fenics/15-tetgen-increase-cmake-minim-required-version.patch"
mkdir -p /tmp/mshr-src/build
cd /tmp/mshr-src/build
cmake \
    -DCMAKE_C_COMPILER=$(which mpicc) \
    -DCMAKE_CXX_COMPILER=$(which mpicxx) \
    -DCMAKE_SKIP_RPATH:BOOL=ON \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
    -DUSE_SYSTEM_CGAL:BOOL=ON \
    ..
make -j $(nproc) install
cd /tmp/mshr-src/python
PYTHONUSERBASE=$INSTALL_PREFIX CXX="mpicxx" python3 -m pip install --user .
cd && rm -rf /tmp/mshr-src/
